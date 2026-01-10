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


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Pinecone 데이터 재구축')
    parser.add_argument('--resume', action='store_true', help='중단 지점부터 재개')
    parser.add_argument('--dry-run', action='store_true', help='테스트 모드 (실제 저장 안함)')
    parser.add_argument('--limit', type=int, default=0, help='처리할 비디오 수 제한 (0=전체)')
    args = parser.parse_args()

    log("=" * 60)
    log("Pinecone 데이터 재구축 스크립트")
    log("=" * 60)
    log(f"청크 크기: {CHUNK_SIZE}토큰, 오버랩: {CHUNK_OVERLAP}토큰")
    log(f"모드: {'DRY-RUN (테스트)' if args.dry_run else '실제 실행'}")
    log(f"재개 모드: {'예' if args.resume else '아니오'}")
    if args.limit > 0:
        log(f"처리 제한: {args.limit}개 비디오")
    log()

    # 초기화
    progress = ProgressTracker()
    if not args.resume:
        log("⚠️ 진행 상태 초기화")
        progress.reset()
    progress.start()

    r2 = R2Client()
    channels_meta = load_channels_metadata()

    log(f"채널 메타데이터 로드: {len(channels_meta)}개 채널")

    # 비디오 목록 조회
    video_ids = r2.list_video_ids()
    log(f"R2 비디오 수: {len(video_ids)}개")

    # 이미 완료된 비디오 제외
    if args.resume:
        video_ids = [v for v in video_ids if not progress.is_completed(v)]
        log(f"처리 대상: {len(video_ids)}개 (완료된 비디오 제외)")

    # 제한 적용
    if args.limit > 0:
        video_ids = video_ids[:args.limit]
        log(f"제한 적용: {len(video_ids)}개")

    # 샘플 데이터 로드 테스트
    if video_ids:
        sample = r2.get_video_data(video_ids[0])
        if sample:
            log(f"\n샘플 데이터 확인 ({video_ids[0]}):")
            log(f"  - raw_segments: {len(sample['raw_segments'])}개")
            log(f"  - refined_text: {len(sample['refined_text'])}자")

    log("\n✅ Task 2 완료: R2 클라이언트 및 데이터 로더")
    log("⚠️ Task 3부터 구현 필요")


if __name__ == "__main__":
    main()
