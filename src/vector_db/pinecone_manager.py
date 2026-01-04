"""
Pinecone 벡터 데이터베이스 관리자

Medical RAG 파이프라인을 위한 Pinecone 벡터 DB 연동 모듈입니다.

주요 기능:
    - 인덱스 생성 및 관리
    - 벡터 upsert/query/delete
    - 메타데이터 기반 필터링 검색
    - 인덱스 통계 조회

사용 예시:
    manager = PineconeManager()
    results = manager.query(embedding, top_k=5, filter={"specialty": "소아과"})
"""

from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
from typing import Any
import time
import logging

# 로거 설정
logger = logging.getLogger(__name__)


class PineconeManager:
    """
    Pinecone 벡터 데이터베이스를 관리합니다.

    Attributes:
        pc: Pinecone 클라이언트
        index: Pinecone Index 객체
        index_name: 인덱스 이름
    """

    # 인덱스 설정 상수
    DIMENSION = 1024  # gemini-embedding-001 차원 (MRL 지원)
    METRIC = "dotproduct"  # RAG 검색에 최적화
    BATCH_SIZE = 100  # upsert 배치 크기

    pc: Any  # Pinecone 클라이언트
    index: Any  # Pinecone Index

    def __init__(self) -> None:
        """PineconeManager를 초기화합니다."""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY가 설정되지 않았습니다.")

        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        logger.info(f"PineconeManager 초기화 완료: {self.index_name}")

    def _ensure_index_exists(self) -> None:
        """인덱스가 없으면 생성합니다."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]

        if self.index_name in existing_indexes:
            logger.info(f"기존 인덱스 사용: {self.index_name}")
            return

        logger.info(f"인덱스 생성 중: {self.index_name}")
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.DIMENSION,
                metric=self.METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENV
                )
            )
            # 인덱스 준비 대기
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            logger.info(f"인덱스 생성 완료: {self.index_name}")
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            raise

    def upsert_vectors(self, vectors: list[dict[str, Any]]) -> int:
        """
        벡터를 Pinecone에 upsert합니다.

        Args:
            vectors: 벡터 리스트. 각 벡터는 다음 형식:
                {
                    "id": "video_id_chunk_0",
                    "values": [0.1, ...],  # 1024차원
                    "metadata": {...}
                }

        Returns:
            성공적으로 upsert된 벡터 수
        """
        if not vectors:
            return 0

        total_upserted = 0

        try:
            for i in range(0, len(vectors), self.BATCH_SIZE):
                batch = vectors[i : i + self.BATCH_SIZE]
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.debug(f"Upsert 배치 완료: {i} ~ {i + len(batch)}")

            logger.info(f"총 {total_upserted}개 벡터 upsert 완료")
            return total_upserted

        except Exception as e:
            logger.error(f"벡터 upsert 실패: {e}")
            raise

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        namespace: str = ""
    ) -> list[dict[str, Any]]:
        """
        유사 벡터를 검색합니다.

        Args:
            vector: 쿼리 임베딩 벡터 (1024차원)
            top_k: 반환할 결과 수 (기본값: 5)
            filter: 메타데이터 필터 (예: {"specialty": "소아과"})
            include_metadata: 메타데이터 포함 여부
            namespace: 검색할 네임스페이스 (기본값: "")

        Returns:
            검색 결과 리스트. 각 결과는 다음 형식:
            {
                "id": "video_id_chunk_0",
                "score": 0.95,
                "metadata": {...}
            }

        사용 예시:
            # 기본 검색
            results = manager.query(embedding, top_k=5)

            # 전문가 인증 콘텐츠만 검색
            results = manager.query(
                embedding,
                top_k=10,
                filter={"is_verified_professional": True}
            )

            # 특정 채널의 콘텐츠만 검색
            results = manager.query(
                embedding,
                filter={"channel_id": "UC..."}
            )
        """
        try:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                include_metadata=include_metadata,
                namespace=namespace
            )

            results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                if include_metadata and match.metadata:
                    result["metadata"] = dict(match.metadata)
                results.append(result)

            logger.debug(f"쿼리 완료: {len(results)}개 결과")
            return results

        except Exception as e:
            logger.error(f"쿼리 실패: {e}")
            raise

    def query_by_video(
        self,
        vector: list[float],
        video_id: str,
        top_k: int = 10
    ) -> list[dict[str, Any]]:
        """
        특정 비디오 내에서 유사 청크를 검색합니다.

        Args:
            vector: 쿼리 임베딩 벡터
            video_id: YouTube 비디오 ID
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        return self.query(
            vector=vector,
            top_k=top_k,
            filter={"video_id": {"$eq": video_id}}
        )

    def query_verified_only(
        self,
        vector: list[float],
        top_k: int = 5,
        specialty: str | None = None
    ) -> list[dict[str, Any]]:
        """
        인증된 의료 전문가의 콘텐츠만 검색합니다.

        Args:
            vector: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수
            specialty: 특정 전문 분야 필터 (선택)

        Returns:
            검색 결과 리스트
        """
        filter_dict: dict[str, Any] = {"is_verified_professional": {"$eq": True}}

        if specialty:
            filter_dict["specialty"] = {"$eq": specialty}

        return self.query(
            vector=vector,
            top_k=top_k,
            filter=filter_dict
        )

    def delete_video_vectors(self, video_id: str) -> None:
        """
        특정 비디오의 모든 벡터를 삭제합니다.

        Args:
            video_id: 삭제할 비디오 ID
        """
        try:
            self.index.delete(
                filter={"video_id": {"$eq": video_id}}
            )
            logger.info(f"비디오 벡터 삭제 완료: {video_id}")
        except Exception as e:
            logger.error(f"벡터 삭제 실패 ({video_id}): {e}")
            raise

    def delete_channel_vectors(self, channel_id: str) -> None:
        """
        특정 채널의 모든 벡터를 삭제합니다.

        Args:
            channel_id: 삭제할 채널 ID
        """
        try:
            self.index.delete(
                filter={"channel_id": {"$eq": channel_id}}
            )
            logger.info(f"채널 벡터 삭제 완료: {channel_id}")
        except Exception as e:
            logger.error(f"채널 벡터 삭제 실패 ({channel_id}): {e}")
            raise

    def get_index_stats(self) -> dict[str, Any]:
        """
        인덱스 통계를 조회합니다.

        Returns:
            통계 딕셔너리:
            {
                "total_vector_count": 1000,
                "dimension": 1024,
                "namespaces": {...}
            }
        """
        try:
            stats = self.index.describe_index_stats()
            return stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            raise

    def is_ready(self) -> bool:
        """
        인덱스가 준비 상태인지 확인합니다.

        Returns:
            준비 상태 여부
        """
        try:
            info = self.pc.describe_index(self.index_name)
            return bool(info.status.get('ready', False))
        except Exception:
            return False

    def get_vector_count(self) -> int:
        """
        인덱스의 총 벡터 수를 반환합니다.

        Returns:
            총 벡터 수
        """
        stats = self.get_index_stats()
        count: int = stats.get('total_vector_count', 0)
        return count
