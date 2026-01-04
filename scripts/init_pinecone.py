#!/usr/bin/env python3
"""
Pinecone 벡터 데이터베이스 초기화 스크립트

이 스크립트는 Medical RAG 파이프라인을 위한 Pinecone 인덱스를 생성하고 설정합니다.

사용법:
    python scripts/init_pinecone.py

기능:
    1. Pinecone 연결 확인
    2. 인덱스 생성 (없는 경우)
    3. 인덱스 상태 확인
    4. 샘플 벡터 테스트 (선택)
"""

import sys
import time
from pathlib import Path
from typing import Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinecone import Pinecone, ServerlessSpec  # type: ignore[import-untyped]  # noqa: E402
from config.settings import settings  # noqa: E402


# =============================================================================
# 인덱스 설정 상수
# =============================================================================

# 인덱스 설정값을 명시적으로 타입 지정
INDEX_NAME: str = settings.PINECONE_INDEX_NAME  # "medical-rag-hybrid"
INDEX_DIMENSION: int = 768  # gemini-embedding-001 차원
INDEX_METRIC: str = "dotproduct"  # RAG 검색에 최적화
INDEX_CLOUD: str = "aws"
INDEX_REGION: str = settings.PINECONE_ENV  # "us-east-1"

# 메타데이터 스키마 (문서화 목적)
METADATA_SCHEMA: dict[str, str] = {
    "text": "str - 청크 원본 텍스트",
    "context": "str - Contextual Retrieval 컨텍스트",
    "video_id": "str - YouTube 비디오 ID (필터링 가능)",
    "video_title": "str - 비디오 제목",
    "channel_id": "str - YouTube 채널 ID (필터링 가능)",
    "channel_name": "str - 채널 이름",
    "chunk_index": "int - 청크 순서 인덱스",
    "source_type": "str - 소스 유형 (youtube)",
    "video_url": "str - 비디오 URL",
    "published_at": "str - 게시일 (ISO 8601)",
    "is_verified_professional": "bool - 의료 전문가 인증 여부 (필터링 가능)",
    "specialty": "str - 전문 분야 (예: 소아과)",
    "credentials": "str - 전문가 자격 정보",
    "timestamp_start": "str - 청크 시작 시간 (MM:SS)",
    "timestamp_end": "str - 청크 종료 시간 (MM:SS)",
    "topics": "list[str] - 의료 토픽 키워드",
    "processed_at": "str - 처리 시간 (ISO 8601)",
}


def check_api_key() -> bool:
    """API 키 존재 여부를 확인합니다."""
    if not settings.PINECONE_API_KEY:
        print("❌ 오류: PINECONE_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 PINECONE_API_KEY를 추가하세요.")
        print("   발급: https://app.pinecone.io/")
        return False
    print("✅ API 키 확인 완료")
    return True


def connect_pinecone() -> Pinecone | None:
    """Pinecone 클라이언트에 연결합니다."""
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        # 연결 테스트: 인덱스 목록 조회
        indexes = pc.list_indexes()
        print("✅ Pinecone 연결 성공")
        print(f"   기존 인덱스 수: {len(list(indexes))}")
        return pc
    except Exception as e:
        print(f"❌ Pinecone 연결 실패: {e}")
        return None


def create_index(pc: Pinecone) -> bool:
    """인덱스를 생성하거나 기존 인덱스를 확인합니다."""
    # 기존 인덱스 확인
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"ℹ️  인덱스 '{INDEX_NAME}'가 이미 존재합니다.")
        return True

    print(f"🔧 인덱스 '{INDEX_NAME}' 생성 중...")

    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(
                cloud=INDEX_CLOUD,
                region=INDEX_REGION
            )
        )

        # 인덱스가 준비될 때까지 대기
        print("   인덱스 준비 대기 중", end="", flush=True)
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print(".", end="", flush=True)
            time.sleep(1)
        print()

        print(f"✅ 인덱스 '{INDEX_NAME}' 생성 완료")
        return True

    except Exception as e:
        print(f"❌ 인덱스 생성 실패: {e}")
        return False


def get_index_stats(pc: Pinecone) -> dict[str, Any] | None:
    """인덱스 통계를 조회합니다."""
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        return stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
    except Exception as e:
        print(f"❌ 통계 조회 실패: {e}")
        return None


def print_index_info(pc: Pinecone) -> None:
    """인덱스 정보를 출력합니다."""
    try:
        index_info = pc.describe_index(INDEX_NAME)
        stats = get_index_stats(pc)

        print("\n" + "=" * 60)
        print("📊 Pinecone 인덱스 정보")
        print("=" * 60)
        print(f"  이름: {INDEX_NAME}")
        print(f"  차원: {INDEX_DIMENSION}")
        print(f"  메트릭: {INDEX_METRIC}")
        print(f"  클라우드: {INDEX_CLOUD}")
        print(f"  리전: {INDEX_REGION}")
        print(f"  상태: {'준비됨' if index_info.status['ready'] else '준비 중'}")

        if stats:
            print("\n📈 통계:")
            total_count = stats.get('total_vector_count', 0)
            namespaces = list(stats.get('namespaces', {}).keys()) or ['(기본)']
            print(f"  총 벡터 수: {total_count:,}")
            print(f"  네임스페이스: {namespaces}")

        print("=" * 60)

    except Exception as e:
        print(f"❌ 인덱스 정보 조회 실패: {e}")


def print_metadata_schema() -> None:
    """메타데이터 스키마를 출력합니다."""
    print("\n" + "=" * 60)
    print("📋 메타데이터 스키마")
    print("=" * 60)

    for field, description in METADATA_SCHEMA.items():
        print(f"  {field}: {description}")

    print("=" * 60)


def test_sample_vector(pc: Pinecone) -> bool:
    """샘플 벡터로 upsert/query 테스트를 수행합니다."""
    print("\n🧪 샘플 벡터 테스트 시작...")

    try:
        index = pc.Index(INDEX_NAME)

        # 테스트용 샘플 벡터 생성 (768차원)
        test_id = "test_vector_init"
        test_vector: list[float] = [0.1] * 768  # 768차원 더미 벡터
        test_metadata: dict[str, Any] = {
            "text": "테스트 텍스트입니다.",
            "video_id": "test_video",
            "channel_id": "test_channel",
            "is_verified_professional": True,
            "specialty": "테스트",
        }

        # Upsert 테스트
        index.upsert(
            vectors=[(test_id, test_vector, test_metadata)]
        )
        print("  ✅ Upsert 성공")

        # 짧은 대기 (인덱싱 시간)
        time.sleep(1)

        # Query 테스트
        results = index.query(
            vector=test_vector,
            top_k=1,
            include_metadata=True
        )

        if hasattr(results, 'matches') and results.matches and len(results.matches) > 0:
            print("  ✅ Query 성공")
            print(f"     매칭 ID: {results.matches[0].id}")
            print(f"     점수: {results.matches[0].score:.4f}")
        else:
            print("  ⚠️ Query 결과 없음 (인덱싱 지연일 수 있음)")

        # 테스트 벡터 삭제
        index.delete(ids=[test_id])
        print("  ✅ 테스트 벡터 삭제 완료")

        return True

    except Exception as e:
        print(f"  ❌ 테스트 실패: {e}")
        return False


def main() -> None:
    """메인 초기화 함수입니다."""
    print("\n" + "=" * 60)
    print("🚀 Pinecone 초기화 시작")
    print("   Medical RAG Pipeline Vector Database Setup")
    print("=" * 60 + "\n")

    # 1. API 키 확인
    if not check_api_key():
        sys.exit(1)

    # 2. Pinecone 연결
    pc = connect_pinecone()
    if not pc:
        sys.exit(1)

    # 3. 인덱스 생성/확인
    if not create_index(pc):
        sys.exit(1)

    # 4. 인덱스 정보 출력
    print_index_info(pc)

    # 5. 메타데이터 스키마 출력
    print_metadata_schema()

    # 6. 샘플 테스트 (선택)
    user_input = input("\n샘플 벡터 테스트를 수행하시겠습니까? (y/N): ").strip().lower()
    if user_input == 'y':
        test_sample_vector(pc)

    print("\n✅ Pinecone 초기화 완료!")
    print("   이제 main.py를 실행하여 파이프라인을 시작할 수 있습니다.\n")


if __name__ == "__main__":
    main()
