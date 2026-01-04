#!/usr/bin/env python3
"""
R2 청크 데이터 → 1024차원 임베딩 → Pinecone 저장 스크립트.

R2에 저장된 기존 청크 데이터를 가져와서 새로운 1024차원 임베딩을 생성하고
Pinecone에 저장합니다. YouTube API 호출 없이 임베딩만 재생성합니다.

사용법:
    python scripts/reembed_from_r2.py

기능:
    - R2에서 청크 데이터 로드
    - 1024차원 임베딩 생성 (Google API → OpenRouter 폴백)
    - Pinecone에 벡터 저장
    - 진행 상황 저장 (중단 후 재시작 가능)
    - Rate limit 고려한 배치 처리
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from config.settings import settings
from src.processors.gemini_processor import GeminiProcessor
from src.vector_db.pinecone_manager import PineconeManager


# 설정
BATCH_SIZE = 10  # Pinecone upsert 배치 크기
DELAY_BETWEEN_EMBEDDINGS = 0.5  # 임베딩 간 딜레이 (초) - rate limit 방지
PROGRESS_FILE = Path("data/reembed_progress.json")


def get_r2_client():
    """R2 클라이언트 생성."""
    return boto3.client(
        's3',
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )


def list_all_chunk_keys(s3_client, bucket: str) -> list[str]:
    """R2에서 모든 청크 파일 키 목록 조회."""
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix='chunks/'):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.json'):
                    keys.append(obj['Key'])

    return keys


def load_chunks_from_r2(s3_client, bucket: str, key: str) -> list[dict]:
    """R2에서 청크 데이터 로드."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data if isinstance(data, list) else [data]
    except Exception as e:
        log(f"  ⚠️ 청크 로드 실패 ({key}): {e}")
        return []


def load_progress() -> dict:
    """진행 상황 로드."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_keys": [], "failed_keys": [], "total_vectors": 0}


def save_progress(progress: dict):
    """진행 상황 저장."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def create_vector_from_chunk(
    chunk: dict,
    embedding: list[float],
    video_id: str
) -> dict:
    """청크와 임베딩으로 Pinecone 벡터 생성."""
    chunk_index = chunk.get('chunk_index', 0)

    # topics 정규화 (이전에 추가한 로직 반영)
    topics = chunk.get('topics', [])
    if not topics or topics == ['NONE']:
        topics = ['NONE']

    return {
        'id': f"{video_id}_chunk_{chunk_index}",
        'values': embedding,
        'metadata': {
            'video_id': video_id,
            'video_title': chunk.get('video_title', ''),
            'channel_id': chunk.get('channel_id', ''),
            'channel_name': chunk.get('channel_name', ''),
            'chunk_index': chunk_index,
            'chunk_text': chunk.get('text', ''),
            'context': chunk.get('context', ''),
            'topics': topics,
            'start_time': chunk.get('start_time', 0),
            'end_time': chunk.get('end_time', 0),
            'processed_at': datetime.now().isoformat(),
        }
    }


def log(msg: str = ""):
    """실시간 로그 출력."""
    print(msg, flush=True)


def main():
    """메인 실행 함수."""
    log("=" * 60)
    log("R2 청크 데이터 → 1024차원 임베딩 → Pinecone")
    log("=" * 60)
    log("")

    # 초기화
    log("🔧 초기화 중...")
    s3_client = get_r2_client()
    bucket = settings.R2_BUCKET_NAME
    processor = GeminiProcessor()
    pinecone_manager = PineconeManager()

    # 진행 상황 로드
    progress = load_progress()
    completed_keys = set(progress['completed_keys'])

    log(f"  - R2 버킷: {bucket}")
    log(f"  - 이전 완료: {len(completed_keys)}개 파일")
    log()

    # 청크 키 목록 조회
    log("📂 R2에서 청크 파일 목록 조회 중...")
    all_keys = list_all_chunk_keys(s3_client, bucket)
    remaining_keys = [k for k in all_keys if k not in completed_keys]

    log(f"  - 전체 파일: {len(all_keys)}개")
    log(f"  - 남은 파일: {len(remaining_keys)}개")
    log()

    if not remaining_keys:
        log("✅ 모든 파일 처리 완료!")
        return

    # 처리 시작
    log("🚀 임베딩 생성 시작...")
    log("-" * 60)

    total_vectors = progress['total_vectors']
    failed_keys = progress['failed_keys']
    vectors_batch = []

    start_time = time.time()

    for i, key in enumerate(remaining_keys):
        # 비디오 ID 추출 (chunks/VIDEO_ID/chunks.json)
        video_id = key.split('/')[1]

        log(f"\n[{i+1}/{len(remaining_keys)}] 처리 중: {video_id}")

        # 청크 로드
        chunks = load_chunks_from_r2(s3_client, bucket, key)
        if not chunks:
            failed_keys.append(key)
            continue

        log(f"  청크 수: {len(chunks)}개")

        # 각 청크에 대해 임베딩 생성
        chunk_vectors = []
        for j, chunk in enumerate(chunks):
            # 임베딩 텍스트 생성 (context + text)
            context = chunk.get('context', '')
            text = chunk.get('text', '')
            embedding_text = f"{context}\n\n{text}" if context else text

            # 임베딩 생성
            try:
                embedding = processor.get_embedding(embedding_text)
                if not embedding:
                    log(f"  ⚠️ 청크 {j} 임베딩 실패 (빈 결과)")
                    continue

                # 벡터 생성
                vector = create_vector_from_chunk(chunk, embedding, video_id)
                chunk_vectors.append(vector)

                # Rate limit 방지
                time.sleep(DELAY_BETWEEN_EMBEDDINGS)

            except Exception as e:
                log(f"  ⚠️ 청크 {j} 임베딩 실패: {e}")
                continue

        log(f"  ✅ 임베딩 생성: {len(chunk_vectors)}/{len(chunks)}개")

        # 배치에 추가
        vectors_batch.extend(chunk_vectors)

        # 배치 크기 도달 시 Pinecone에 저장
        if len(vectors_batch) >= BATCH_SIZE:
            try:
                upserted = pinecone_manager.upsert_vectors(vectors_batch)
                total_vectors += upserted
                log(f"  📤 Pinecone 저장: {upserted}개 (누적: {total_vectors}개)")
                vectors_batch = []
            except Exception as e:
                log(f"  ❌ Pinecone 저장 실패: {e}")

        # 완료 기록
        completed_keys.add(key)

        # 진행 상황 저장 (10개마다)
        if (i + 1) % 10 == 0:
            progress = {
                "completed_keys": list(completed_keys),
                "failed_keys": failed_keys,
                "total_vectors": total_vectors
            }
            save_progress(progress)

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60  # 분당 처리량
            log(f"\n  💾 진행 저장 (처리율: {rate:.1f} 파일/분)")

    # 남은 배치 저장
    if vectors_batch:
        try:
            upserted = pinecone_manager.upsert_vectors(vectors_batch)
            total_vectors += upserted
            log(f"\n  📤 최종 Pinecone 저장: {upserted}개")
        except Exception as e:
            log(f"\n  ❌ 최종 Pinecone 저장 실패: {e}")

    # 최종 진행 상황 저장
    progress = {
        "completed_keys": list(completed_keys),
        "failed_keys": failed_keys,
        "total_vectors": total_vectors,
        "completed_at": datetime.now().isoformat()
    }
    save_progress(progress)

    # 결과 출력
    elapsed = time.time() - start_time
    log()
    log("=" * 60)
    log("✅ 완료!")
    log("=" * 60)
    log(f"  - 처리 파일: {len(remaining_keys)}개")
    log(f"  - 실패 파일: {len(failed_keys)}개")
    log(f"  - 총 벡터: {total_vectors}개")
    log(f"  - 소요 시간: {elapsed/60:.1f}분")
    log()

    if failed_keys:
        log("⚠️ 실패한 파일:")
        for key in failed_keys[:10]:
            log(f"  - {key}")
        if len(failed_keys) > 10:
            log(f"  ... 외 {len(failed_keys) - 10}개")


if __name__ == "__main__":
    main()
