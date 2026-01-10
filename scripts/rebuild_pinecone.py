#!/usr/bin/env python3
"""
Pinecone 데이터 재구축 스크립트

R2에서 원본 데이터를 읽어 새로운 스키마로 Pinecone에 재업로드합니다.
- 청킹: 120토큰, 오버랩 20
- ID 형식: {video_id}_chunk_{n}
- 필드명: chunk_text, start_time, end_time (구버전)
- 메타데이터: specialty, credentials 등 (신버전 추가)

사용법:
    python scripts/rebuild_pinecone.py              # 전체 실행
    python scripts/rebuild_pinecone.py --resume     # 중단 지점부터 재개
    python scripts/rebuild_pinecone.py --dry-run    # 테스트 (실제 저장 안함)
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

# 외부 라이브러리 (실제 실행 시 필요)
# boto3, pinecone, tiktoken은 Task 2에서 사용
try:
    import boto3
    from botocore.config import Config
    from pinecone import Pinecone
    import tiktoken
except ImportError as e:
    # --help 실행 시에는 필요 없음
    pass

# 프로젝트 루트를 path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 프로젝트 모듈 (실제 실행 시 필요)
try:
    from config.settings import settings
    from src.processors.gemini_processor import GeminiProcessor
except ImportError:
    # --help 실행 시에는 필요 없음
    pass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 상수
CHUNK_SIZE = 120          # 토큰
CHUNK_OVERLAP = 20        # 토큰
PINECONE_BATCH_SIZE = 100
PROGRESS_FILE = "data/rebuild_progress.json"


def log(msg: str = "") -> None:
    """실시간 로그 출력"""
    print(msg, flush=True)


class ProgressTracker:
    """진행 상태 추적 (중단/재개 지원)"""

    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = Path(progress_file)
        self.data = self._load()

    def _load(self) -> dict:
        """진행 상태 파일 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed_videos": [],
            "failed_videos": [],
            "total_chunks": 0,
            "started_at": None,
            "last_updated": None
        }

    def save(self) -> None:
        """진행 상태 저장"""
        self.data["last_updated"] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_completed(self, video_id: str) -> bool:
        """비디오가 이미 처리되었는지 확인"""
        return video_id in self.data["completed_videos"]

    def mark_completed(self, video_id: str, chunk_count: int) -> None:
        """비디오 처리 완료 표시"""
        if video_id not in self.data["completed_videos"]:
            self.data["completed_videos"].append(video_id)
            self.data["total_chunks"] += chunk_count
        self.save()

    def mark_failed(self, video_id: str, error: str) -> None:
        """비디오 처리 실패 표시"""
        self.data["failed_videos"].append({"video_id": video_id, "error": error})
        self.save()

    def start(self) -> None:
        """작업 시작 시간 기록"""
        if not self.data["started_at"]:
            self.data["started_at"] = datetime.now().isoformat()
        self.save()

    def reset(self) -> None:
        """진행 상태 초기화"""
        self.data = {
            "completed_videos": [],
            "failed_videos": [],
            "total_chunks": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": None
        }
        self.save()

    def get_summary(self) -> dict:
        """진행 상태 요약 반환"""
        return {
            "completed_count": len(self.data["completed_videos"]),
            "failed_count": len(self.data["failed_videos"]),
            "total_chunks": self.data["total_chunks"],
            "started_at": self.data["started_at"],
            "last_updated": self.data["last_updated"]
        }


class R2Client:
    """R2 스토리지 클라이언트"""

    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        self.bucket = settings.R2_BUCKET_NAME

    def get_json(self, key: str) -> dict | list | None:
        """R2에서 JSON 파일 읽기"""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.debug(f"R2 읽기 실패 ({key}): {e}")
            return None

    def list_video_ids(self) -> list[str]:
        """transcripts/ 폴더의 모든 비디오 ID 목록 반환"""
        video_ids = set()
        paginator = self.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix='transcripts/'):
            for obj in page.get('Contents', []):
                parts = obj['Key'].split('/')
                if len(parts) >= 2 and parts[1]:
                    video_ids.add(parts[1])

        return sorted(list(video_ids))

    def get_video_data(self, video_id: str) -> dict | None:
        """비디오의 raw.json과 refined.json 로드"""
        raw = self.get_json(f"transcripts/{video_id}/raw.json")
        refined = self.get_json(f"transcripts/{video_id}/refined.json")

        if not raw or not refined:
            return None

        return {
            "video_id": video_id,
            "raw_segments": raw,
            "refined_text": refined.get("text", "")
        }


def load_channels_metadata() -> dict[str, dict]:
    """channels.json에서 채널별 메타데이터 로드"""
    channels_path = Path("data/channels.json")
    if not channels_path.exists():
        logger.error("data/channels.json 파일이 없습니다")
        return {}

    with open(channels_path, 'r', encoding='utf-8') as f:
        channels = json.load(f)

    # channel_id를 키로 하는 딕셔너리로 변환
    return {ch['channel_id']: ch for ch in channels}


# tiktoken 인코더 (지연 초기화)
_TOKENIZER = None


def get_tokenizer():
    """토크나이저 인스턴스 반환 (지연 초기화)"""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = tiktoken.get_encoding("gpt2")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 반환"""
    return len(get_tokenizer().encode(text))


def chunk_with_timestamps(
    refined_text: str,
    raw_segments: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    refined_text를 청킹하고 raw_segments에서 타임스탬프 매핑

    Args:
        refined_text: 정제된 전체 텍스트
        raw_segments: 원본 세그먼트 리스트 [{"text": "...", "start": 0.0, "duration": 5.0}, ...]
        chunk_size: 청크 크기 (토큰)
        overlap: 오버랩 크기 (토큰)

    Returns:
        [{"text": "...", "start_time": 0.0, "end_time": 30.5, "chunk_index": 0}, ...]
    """
    if not refined_text or not raw_segments:
        return []

    # 1. raw_segments에서 타임스탬프 매핑 생성
    # 문자 위치 -> 시간 매핑을 위한 누적 텍스트 구축
    char_to_time = []
    current_pos = 0

    for seg in raw_segments:
        seg_text = seg.get('text', '')
        start = seg.get('start', 0)
        duration = seg.get('duration', 0)
        end = start + duration

        # 각 문자에 시간 할당 (선형 보간)
        for i, char in enumerate(seg_text):
            ratio = i / max(len(seg_text), 1)
            time_at_char = start + (duration * ratio)
            char_to_time.append(time_at_char)

        # 세그먼트 사이 공백
        char_to_time.append(end)

    # 2. refined_text를 토큰 단위로 청킹
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(refined_text)
    chunks = []

    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)

        # 3. 청크의 시작/종료 시간 추정
        # refined_text에서 청크의 대략적인 위치 계산
        text_start_ratio = start_idx / max(len(tokens), 1)
        text_end_ratio = end_idx / max(len(tokens), 1)

        # 전체 시간 범위
        if raw_segments:
            total_duration = raw_segments[-1].get('start', 0) + raw_segments[-1].get('duration', 0)
        else:
            total_duration = 0

        start_time = total_duration * text_start_ratio
        end_time = total_duration * text_end_ratio

        chunks.append({
            "text": chunk_text.strip(),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "chunk_index": len(chunks)
        })

        # 다음 청크 시작 (오버랩 적용)
        start_idx += (chunk_size - overlap)

    return chunks


def generate_context_and_topics(
    processor: GeminiProcessor,
    chunk_text: str,
    video_title: str,
    channel_name: str
) -> tuple[str, list[str]]:
    """
    Gemini API로 청크의 context와 topics 생성

    Args:
        processor: GeminiProcessor 인스턴스
        chunk_text: 청크 텍스트
        video_title: 비디오 제목
        channel_name: 채널 이름

    Returns:
        (context, topics)
    """
    # 비디오 요약 생성 (간단 버전)
    video_summary = f"'{video_title}' - {channel_name} 채널의 의료/건강 정보 영상"

    try:
        # GeminiProcessor의 기존 메서드 활용
        context, topics = processor.generate_chunk_context_and_topics(
            chunk_text, video_summary
        )

        # topics가 없거나 빈 경우 기본값
        if not topics or topics == ['NONE']:
            topics = ['NONE']

        return context, topics

    except Exception as e:
        logger.warning(f"Context/Topics 생성 실패: {e}")
        # 폴백: 기본 context 생성
        context = f"이 내용은 {channel_name} 채널의 '{video_title}' 영상에서 발췌한 것입니다."
        return context, ['NONE']


class RateLimiter:
    """API 호출 속도 제한"""

    def __init__(self, calls_per_minute: int = 15):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0

    def wait(self) -> None:
        """필요시 대기"""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def create_pinecone_vector(
    video_id: str,
    chunk_index: int,
    chunk_text: str,
    context: str,
    topics: list[str],
    start_time: float,
    end_time: float,
    video_title: str,
    channel_id: str,
    channel_name: str,
    channel_meta: dict,
    embedding: list[float],
    published_at: str = ""
) -> dict:
    """
    Pinecone 저장용 벡터 구조 생성

    최종 스키마:
    - ID: {video_id}_chunk_{n}
    - 구버전 필드명: chunk_text, start_time, end_time
    - 신버전 메타데이터: specialty, credentials 등
    """
    # specialty 정규화
    specialty = channel_meta.get('specialty', [])
    if isinstance(specialty, list):
        specialty = specialty[0] if specialty else 'NONE'

    return {
        'id': f"{video_id}_chunk_{chunk_index}",
        'values': embedding,
        'metadata': {
            # 구버전 필드명
            'chunk_text': chunk_text,
            'start_time': start_time,
            'end_time': end_time,

            # 공통 필드
            'video_id': video_id,
            'video_title': video_title,
            'channel_id': channel_id,
            'channel_name': channel_name,
            'chunk_index': chunk_index,
            'context': context,
            'topics': topics,
            'processed_at': datetime.now().isoformat(),

            # 신버전 추가 필드
            'is_verified_professional': channel_meta.get('is_verified_professional', False),
            'specialty': specialty,
            'credentials': channel_meta.get('credentials', ''),
            'video_url': f'https://www.youtube.com/watch?v={video_id}',
            'source_type': 'youtube',
            'published_at': published_at
        }
    }


def generate_embedding(
    processor: GeminiProcessor,
    chunk_text: str,
    context: str,
    rate_limiter: RateLimiter
) -> list[float] | None:
    """
    청크의 임베딩 벡터 생성

    context + chunk_text를 결합하여 임베딩
    """
    rate_limiter.wait()

    try:
        combined_text = f"{context}\n\n{chunk_text}"
        embedding = processor.get_embedding(combined_text)
        return embedding
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        return None


def delete_all_vectors(index) -> int:
    """Pinecone 인덱스의 모든 벡터 삭제"""
    try:
        # 전체 삭제
        index.delete(delete_all=True)
        log("  모든 벡터 삭제 완료")
        return 0
    except Exception as e:
        logger.error(f"벡터 삭제 실패: {e}")
        raise


def upsert_vectors(index, vectors: list[dict]) -> int:
    """벡터 배치 upsert"""
    if not vectors:
        return 0

    try:
        for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
            batch = vectors[i:i + PINECONE_BATCH_SIZE]
            index.upsert(vectors=batch)
        return len(vectors)
    except Exception as e:
        logger.error(f"벡터 upsert 실패: {e}")
        raise


def process_video(
    video_id: str,
    r2: R2Client,
    processor: GeminiProcessor,
    channels_meta: dict,
    rate_limiter: RateLimiter,
    dry_run: bool = False
) -> list[dict]:
    """
    단일 비디오 처리: R2 로드 → 청킹 → Context/Topics → 임베딩 → 벡터 생성

    Returns:
        생성된 벡터 리스트
    """
    # 1. R2에서 데이터 로드
    data = r2.get_video_data(video_id)
    if not data:
        raise ValueError(f"R2 데이터 없음: {video_id}")

    # 2. 메타데이터 조회 (R2에서 추가 정보 로드)
    metadata = r2.get_json(f"metadata/{video_id}/metadata.json") or {}
    video_title = metadata.get('video_title', f'Video {video_id}')
    channel_id = metadata.get('channel_id', '')
    channel_name = metadata.get('channel_name', 'Unknown')
    published_at = metadata.get('published_at', '')

    # 채널 메타데이터 조회
    channel_meta = channels_meta.get(channel_id, {})
    if not channel_meta:
        # channel_name으로 재시도
        for ch in channels_meta.values():
            if ch.get('name') == channel_name:
                channel_meta = ch
                break

    # 3. 청킹
    chunks = chunk_with_timestamps(
        data['refined_text'],
        data['raw_segments']
    )

    if not chunks:
        raise ValueError(f"청크 생성 실패: {video_id}")

    # 4. 각 청크 처리
    vectors = []
    for chunk in chunks:
        # Context & Topics 생성
        rate_limiter.wait()
        context, topics = generate_context_and_topics(
            processor,
            chunk['text'],
            video_title,
            channel_name
        )

        if dry_run:
            # dry-run 모드에서는 더미 임베딩 사용
            embedding = [0.0] * 1024
        else:
            # 임베딩 생성
            embedding = generate_embedding(
                processor,
                chunk['text'],
                context,
                rate_limiter
            )

            if not embedding:
                logger.warning(f"임베딩 실패, 스킵: {video_id}_chunk_{chunk['chunk_index']}")
                continue

        # 벡터 구조 생성
        vector = create_pinecone_vector(
            video_id=video_id,
            chunk_index=chunk['chunk_index'],
            chunk_text=chunk['text'],
            context=context,
            topics=topics,
            start_time=chunk['start_time'],
            end_time=chunk['end_time'],
            video_title=video_title,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_meta=channel_meta,
            embedding=embedding,
            published_at=published_at
        )

        vectors.append(vector)

    return vectors


def main():
    parser = argparse.ArgumentParser(description='Pinecone 데이터 재구축')
    parser.add_argument('--resume', action='store_true', help='중단 지점부터 재개')
    parser.add_argument('--dry-run', action='store_true', help='테스트 모드 (실제 저장 안함)')
    parser.add_argument('--limit', type=int, default=0, help='처리할 비디오 수 제한 (0=전체)')
    parser.add_argument('--skip-delete', action='store_true', help='기존 데이터 삭제 스킵')
    args = parser.parse_args()

    log("=" * 60)
    log("Pinecone 데이터 재구축")
    log("=" * 60)
    log(f"청크 크기: {CHUNK_SIZE}토큰, 오버랩: {CHUNK_OVERLAP}토큰")
    log(f"모드: {'DRY-RUN (테스트)' if args.dry_run else '실제 실행'}")
    log()

    # 초기화
    progress = ProgressTracker()
    if not args.resume:
        log("⚠️ 진행 상태 초기화")
        progress.reset()
    progress.start()

    r2 = R2Client()
    channels_meta = load_channels_metadata()
    processor = GeminiProcessor()
    rate_limiter = RateLimiter(calls_per_minute=15)

    # Pinecone 초기화
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    log(f"채널 메타데이터: {len(channels_meta)}개")
    log(f"Pinecone 인덱스: {settings.PINECONE_INDEX_NAME}")

    # 기존 데이터 삭제
    if not args.resume and not args.skip_delete and not args.dry_run:
        log("\n🗑️ 기존 Pinecone 데이터 삭제 중...")
        stats = index.describe_index_stats()
        log(f"  삭제 전 벡터 수: {stats.total_vector_count:,}")
        delete_all_vectors(index)
        time.sleep(2)  # 삭제 반영 대기

    # 비디오 목록 조회
    video_ids = r2.list_video_ids()
    log(f"\nR2 비디오 수: {len(video_ids)}개")

    # 이미 완료된 비디오 제외
    if args.resume:
        before = len(video_ids)
        video_ids = [v for v in video_ids if not progress.is_completed(v)]
        log(f"완료된 비디오 제외: {before} → {len(video_ids)}개")

    # 제한 적용
    if args.limit > 0:
        video_ids = video_ids[:args.limit]
        log(f"제한 적용: {len(video_ids)}개")

    # 처리 시작
    log("\n" + "=" * 60)
    log("비디오 처리 시작")
    log("=" * 60)

    total = len(video_ids)
    success = 0
    failed = 0
    total_chunks = 0

    for i, video_id in enumerate(video_ids, 1):
        try:
            log(f"\n[{i}/{total}] {video_id}")

            # 비디오 처리
            vectors = process_video(
                video_id=video_id,
                r2=r2,
                processor=processor,
                channels_meta=channels_meta,
                rate_limiter=rate_limiter,
                dry_run=args.dry_run
            )

            log(f"  청크 수: {len(vectors)}개")

            # Pinecone 저장
            if not args.dry_run and vectors:
                upsert_vectors(index, vectors)
                log(f"  ✅ Pinecone 저장 완료")

            # 진행 상태 업데이트
            progress.mark_completed(video_id, len(vectors))
            success += 1
            total_chunks += len(vectors)

        except Exception as e:
            logger.error(f"  ❌ 실패: {e}")
            progress.mark_failed(video_id, str(e))
            failed += 1

    # 최종 통계
    log("\n" + "=" * 60)
    log("완료!")
    log("=" * 60)
    log(f"성공: {success}개 비디오")
    log(f"실패: {failed}개 비디오")
    log(f"총 청크: {total_chunks}개")

    if not args.dry_run:
        stats = index.describe_index_stats()
        log(f"Pinecone 최종 벡터 수: {stats.total_vector_count:,}")


if __name__ == "__main__":
    main()
