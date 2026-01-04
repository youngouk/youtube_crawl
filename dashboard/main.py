"""
Medical RAG 데이터 모니터링 대시보드 백엔드

이 모듈은 Medical RAG 파이프라인의 데이터 처리 현황을 모니터링하는
FastAPI 기반 웹 대시보드를 제공합니다.

주요 기능:
- R2 스토리지 데이터 현황 조회 (transcripts, chunks, metadata)
- R2에서 실제 메타데이터 로드 및 캐싱
- Pinecone 벡터 DB 통계 조회
- state.json 처리 상태 조회
- Jinja2 템플릿 기반 HTML 대시보드 렌더링
- 비디오 상세 정보 API

의존성:
- fastapi, uvicorn, jinja2
- boto3 (R2/S3 클라이언트)
- pinecone-client
- python-dotenv
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pinecone import Pinecone

import os


# ============================================
# 캐시 클래스
# ============================================
class TTLCache:
    """
    TTL(Time-To-Live) 기반 간단한 캐시

    지정된 시간 동안 데이터를 캐싱하여 R2 호출을 최소화합니다.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        """
        캐시를 초기화합니다.

        Args:
            ttl_seconds: 캐시 유효 시간 (초, 기본값 60초)
        """
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값을 조회합니다.

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None (만료되었거나 없는 경우)
        """
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            # TTL 만료
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """
        캐시에 값을 저장합니다.

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """캐시를 비웁니다."""
        self._cache.clear()

# ============================================
# 로깅 설정
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# 환경변수 로드
# ============================================
# 상위 디렉토리의 .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# R2 설정
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# Pinecone 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# 로컬 데이터 디렉토리 설정 (상위 디렉토리의 data 폴더 참조)
_local_dir = os.getenv("LOCAL_DATA_DIR", "./data")
if _local_dir.startswith("./"):
    # 상대 경로인 경우 상위 디렉토리 기준으로 변환
    LOCAL_DATA_DIR = Path(__file__).parent.parent / _local_dir[2:]
else:
    LOCAL_DATA_DIR = Path(_local_dir)

# ============================================
# FastAPI 앱 초기화
# ============================================
app = FastAPI(
    title="Medical RAG 데이터 모니터링",
    description="Medical RAG 파이프라인 데이터 처리 현황 대시보드",
    version="1.0.0"
)

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ============================================
# R2 스토리지 클라이언트
# ============================================
class R2Client:
    """
    Cloudflare R2 스토리지 클라이언트

    R2 버킷에서 파일 목록 조회 및 JSON 데이터 로드 기능을 제공합니다.
    """

    def __init__(self) -> None:
        """R2 클라이언트를 초기화합니다."""
        self.client: Optional[Any] = None
        self.bucket = R2_BUCKET_NAME
        self._initialize_client()

    def _initialize_client(self) -> None:
        """boto3 S3 클라이언트를 생성합니다."""
        if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
            logger.warning("R2 환경변수가 설정되지 않았습니다.")
            return

        try:
            # 재시도 설정
            config = Config(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=30
            )

            self.client = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT_URL,
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                config=config
            )
            logger.info("R2 클라이언트 초기화 성공")
        except Exception as e:
            logger.error(f"R2 클라이언트 초기화 실패: {e}")
            self.client = None

    def count_objects_in_folder(self, prefix: str) -> int:
        """
        지정된 폴더(prefix) 내의 객체 수를 반환합니다.

        Args:
            prefix: 조회할 폴더 경로 (예: 'transcripts/')

        Returns:
            해당 폴더 내 객체 수
        """
        if not self.client:
            logger.warning("R2 클라이언트가 초기화되지 않았습니다.")
            return 0

        try:
            count = 0
            paginator = self.client.get_paginator("list_objects_v2")

            # 페이지네이션으로 모든 객체 조회
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    # 폴더 자체는 제외하고 파일만 카운트
                    count += sum(
                        1 for obj in page["Contents"]
                        if not obj["Key"].endswith("/")
                    )

            logger.info(f"R2 폴더 '{prefix}' 객체 수: {count}")
            return count
        except Exception as e:
            logger.error(f"R2 객체 수 조회 실패 (prefix={prefix}): {e}")
            return 0

    def get_all_video_ids(self) -> list[str]:
        """
        R2의 metadata 폴더를 스캔하여 모든 비디오 ID를 반환합니다.

        Returns:
            비디오 ID 리스트
        """
        if not self.client:
            logger.warning("R2 클라이언트가 초기화되지 않았습니다.")
            return []

        try:
            video_ids = []
            paginator = self.client.get_paginator("list_objects_v2")

            # 페이지네이션으로 모든 메타데이터 파일 조회
            for page in paginator.paginate(Bucket=self.bucket, Prefix="metadata/"):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # key format: metadata/{video_id}.json
                        if key.endswith(".json"):
                            # "metadata/" 접두사와 ".json" 확장자 제거
                            video_id = key.split("/")[-1].replace(".json", "")
                            video_ids.append(video_id)

            logger.info(f"R2에서 {len(video_ids)}개의 비디오 ID 발견")
            return video_ids
        except Exception as e:
            logger.error(f"비디오 ID 목록 조회 실패: {e}")
            return []

    def load_json(self, key: str) -> Optional[Any]:
        """
        R2에서 JSON 파일을 로드합니다.

        Args:
            key: 객체 키 (파일 경로)

        Returns:
            파싱된 JSON 데이터 또는 None
        """
        if not self.client:
            return None

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except self.client.exceptions.NoSuchKey:
            logger.warning(f"R2 객체를 찾을 수 없습니다: {key}")
            return None
        except Exception as e:
            logger.error(f"R2 JSON 로드 실패 ({key}): {e}")
            return None

    def get_video_metadata(self, video_id: str) -> Optional[dict[str, Any]]:
        """
        R2에서 비디오 메타데이터를 가져옵니다.

        Args:
            video_id: 비디오 ID

        Returns:
            메타데이터 딕셔너리 또는 None
        """
        return self.load_json(f"metadata/{video_id}.json")

    def get_chunk_count(self, video_id: str) -> int:
        """
        R2에서 특정 비디오의 청크 수를 가져옵니다.

        Args:
            video_id: 비디오 ID

        Returns:
            청크 수 (없으면 0)
        """
        chunks = self.load_json(f"chunks/{video_id}/chunks.json")
        return len(chunks) if chunks else 0

    def get_video_chunks(self, video_id: str) -> Optional[list[dict[str, Any]]]:
        """
        R2에서 비디오 청크 데이터를 가져옵니다.

        Args:
            video_id: 비디오 ID

        Returns:
            청크 리스트 또는 None
        """
        return self.load_json(f"chunks/{video_id}/chunks.json")

    def get_refined_transcript(self, video_id: str) -> Optional[dict[str, Any]]:
        """
        R2에서 정제된 트랜스크립트를 가져옵니다.

        Args:
            video_id: 비디오 ID

        Returns:
            트랜스크립트 데이터 또는 None
        """
        return self.load_json(f"transcripts/{video_id}/refined_transcript.json")

    def get_all_video_metadata(
        self,
        video_ids: list[str],
        cache: Optional[TTLCache] = None
    ) -> dict[str, dict[str, Any]]:
        """
        여러 비디오의 메타데이터를 일괄 조회합니다.

        캐싱을 사용하여 R2 호출을 최소화합니다.

        Args:
            video_ids: 비디오 ID 목록
            cache: TTL 캐시 인스턴스 (선택적)

        Returns:
            video_id를 키로 하는 메타데이터 딕셔너리
        """
        result: dict[str, dict[str, Any]] = {}

        for video_id in video_ids:
            # 캐시 확인
            cache_key = f"metadata:{video_id}"
            if cache:
                cached = cache.get(cache_key)
                if cached is not None:
                    result[video_id] = cached
                    continue

            # R2에서 조회
            metadata = self.get_video_metadata(video_id)
            if metadata:
                result[video_id] = metadata
                # 캐시에 저장
                if cache:
                    cache.set(cache_key, metadata)

        return result


# ============================================
# Pinecone 클라이언트
# ============================================
class PineconeClient:
    """
    Pinecone 벡터 DB 클라이언트

    Pinecone 인덱스의 벡터 통계를 조회합니다.
    """

    def __init__(self) -> None:
        """Pinecone 클라이언트를 초기화합니다."""
        self.client: Optional[Pinecone] = None
        self.index_name = "medical-rag"  # 기본 인덱스 이름
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Pinecone 클라이언트를 생성합니다."""
        if not PINECONE_API_KEY:
            logger.warning("PINECONE_API_KEY가 설정되지 않았습니다.")
            return

        try:
            self.client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone 클라이언트 초기화 성공")
        except Exception as e:
            logger.error(f"Pinecone 클라이언트 초기화 실패: {e}")
            self.client = None

    def get_vector_count(self) -> int:
        """
        인덱스의 총 벡터 수를 반환합니다.

        Returns:
            총 벡터 수
        """
        if not self.client:
            return 0

        try:
            # 인덱스 목록에서 사용 가능한 인덱스 확인
            indexes = self.client.list_indexes()

            # 인덱스가 존재하는지 확인
            index_names = [idx.name for idx in indexes]
            if self.index_name not in index_names:
                # 대체 인덱스 이름 시도
                if index_names:
                    self.index_name = index_names[0]
                    logger.info(f"대체 인덱스 사용: {self.index_name}")
                else:
                    logger.warning("사용 가능한 Pinecone 인덱스가 없습니다.")
                    return 0

            # 인덱스 통계 조회
            index = self.client.Index(self.index_name)
            stats = index.describe_index_stats()

            total_vectors = stats.total_vector_count
            logger.info(f"Pinecone 총 벡터 수: {total_vectors}")
            return total_vectors
        except Exception as e:
            logger.error(f"Pinecone 벡터 수 조회 실패: {e}")
            return 0


# ============================================
# 상태 관리
# ============================================
class StateManager:
    """
    처리 상태 관리자

    state.json 파일에서 비디오 처리 상태를 로드하고,
    R2에서 실제 메타데이터를 가져와 풍부한 정보를 제공합니다.
    """

    def __init__(
        self,
        data_dir: Path,
        r2_client: Optional["R2Client"] = None
    ) -> None:
        """
        상태 관리자를 초기화합니다.

        Args:
            data_dir: state.json이 위치한 디렉토리
            r2_client: R2 클라이언트 인스턴스 (메타데이터 조회용)
        """
        self.state_file = data_dir / "state.json"
        self.r2_client = r2_client
        # 메타데이터 캐시 (TTL 60초)
        self._metadata_cache = TTLCache(ttl_seconds=60)
        # 통계 캐시 (TTL 60초)
        self._stats_cache = TTLCache(ttl_seconds=60)

    def set_r2_client(self, r2_client: "R2Client") -> None:
        """
        R2 클라이언트를 설정합니다.

        Args:
            r2_client: R2 클라이언트 인스턴스
        """
        self.r2_client = r2_client

    def load_state(self) -> dict[str, Any]:
        """
        state.json 파일을 로드합니다.
        파일이 없거나 로드 실패 시 R2에서 상태를 복구합니다.

        Returns:
            상태 데이터 딕셔너리 (video_id를 키로 하는 딕셔너리)
        """
        # 1. 파일이 존재하면 로드 시도
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"state.json 로드 실패: {e}")

        # 2. 파일이 없거나 실패 시 R2에서 복구
        if self.r2_client:
            logger.info("state.json이 없거나 손상됨. R2에서 상태를 복구합니다...")
            return self._restore_state_from_r2()

        return {}

    def _restore_state_from_r2(self) -> dict[str, Any]:
        """
        R2에 저장된 메타데이터를 기반으로 state.json을 복구합니다.

        Returns:
            복구된 상태 딕셔너리
        """
        if not self.r2_client:
            return {}

        video_ids = self.r2_client.get_all_video_ids()
        restored_state = {}
        now_iso = datetime.now().isoformat()

        for video_id in video_ids:
            # 기본 상태 생성 (이미 R2에 메타데이터가 있으므로 completed로 가정)
            restored_state[video_id] = {
                "status": "completed",
                "updated_at": now_iso,
                "current_step": "completed",
                "retry_count": 0,
                "restored": True  # 복구됨 표시
            }

        # 복구된 상태 저장
        try:
            # 디렉토리가 없으면 생성
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(restored_state, f, ensure_ascii=False, indent=2)
            logger.info(f"R2에서 {len(restored_state)}개 비디오 상태 복구 및 저장 완료")
        except Exception as e:
            logger.error(f"복구된 상태 저장 실패: {e}")
            # 저장 실패해도 메모리 상의 상태는 반환

        return restored_state

    def _truncate_text(self, text: str, max_length: int = 50) -> str:
        """
        텍스트를 지정된 길이로 자릅니다.

        Args:
            text: 원본 텍스트
            max_length: 최대 길이 (기본값 50)

        Returns:
            잘린 텍스트 (초과 시 '...' 추가)
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _format_updated_at(self, updated_at: str) -> str:
        """
        ISO 포맷의 시간을 HH:MM 형식으로 변환합니다.

        Args:
            updated_at: ISO 포맷 시간 문자열

        Returns:
            HH:MM 형식 시간 문자열
        """
        if updated_at == "-" or "T" not in updated_at:
            return updated_at

        try:
            return updated_at.split("T")[1][:5]
        except (IndexError, AttributeError):
            return updated_at

    def _build_video_info(
        self,
        video_id: str,
        video_data: dict[str, Any],
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        비디오 정보를 구성합니다.

        Args:
            video_id: 비디오 ID
            video_data: state.json의 비디오 데이터
            metadata: R2 메타데이터

        Returns:
            비디오 정보 딕셔너리
        """
        status = video_data.get("status", "pending")

        # 시간 포맷팅
        updated_at = self._format_updated_at(
            video_data.get("updated_at", "-")
        )

        # 청크 수: R2에서 가져오거나 state.json 값 사용
        chunk_count = video_data.get("chunk_count", "-")
        if chunk_count == "-" and self.r2_client and status == "completed":
            cache_key = f"chunk_count:{video_id}"
            cached_count = self._metadata_cache.get(cache_key)
            if cached_count is not None:
                chunk_count = cached_count
            else:
                chunk_count = self.r2_client.get_chunk_count(video_id)
                if chunk_count > 0:
                    self._metadata_cache.set(cache_key, chunk_count)

        # 제목: R2 메타데이터 > 기본값
        title = metadata.get("title", f"Video {video_id[:8]}...")

        # 요약: R2 메타데이터 (50자 제한)
        summary = metadata.get("summary", "")
        if summary:
            summary = self._truncate_text(summary, 50)

        return {
            "video_id": video_id,
            "title": title,
            "channel_name": metadata.get("channel_title", "-"),
            "status": status,
            "chunk_count": chunk_count,
            "summary": summary,
            "thumbnail_url": metadata.get("thumbnail_url", ""),
            "updated_at": updated_at
        }

    def get_all_channels(self) -> list[str]:
        """
        모든 고유 채널 목록을 반환합니다.

        Returns:
            채널 이름 리스트 (정렬됨)
        """
        # 캐시 확인
        cached_channels = self._stats_cache.get("channels")
        if cached_channels is not None:
            return cached_channels

        videos = self.load_state()
        video_ids = list(videos.keys())

        # R2에서 메타데이터 조회
        r2_metadata: dict[str, dict[str, Any]] = {}
        if self.r2_client:
            r2_metadata = self.r2_client.get_all_video_metadata(
                video_ids,
                cache=self._metadata_cache
            )

        # 고유 채널 추출
        channels = set()
        for metadata in r2_metadata.values():
            channel = metadata.get("channel_title", "")
            if channel:
                channels.add(channel)

        result = sorted(list(channels))

        # 캐싱
        self._stats_cache.set("channels", result)

        return result

    def get_videos_paginated(
        self,
        page: int = 1,
        per_page: int = 20,
        channel_filter: Optional[str] = None
    ) -> dict[str, Any]:
        """
        페이지네이션된 비디오 목록을 반환합니다.

        Args:
            page: 페이지 번호 (1부터 시작)
            per_page: 페이지당 항목 수
            channel_filter: 채널 필터 (None이면 전체)

        Returns:
            페이지네이션 결과 딕셔너리
        """
        videos = self.load_state()
        video_ids = list(videos.keys())

        # R2에서 메타데이터 일괄 조회
        r2_metadata: dict[str, dict[str, Any]] = {}
        if self.r2_client:
            r2_metadata = self.r2_client.get_all_video_metadata(
                video_ids,
                cache=self._metadata_cache
            )

        # 모든 비디오 정보 구성
        all_videos: list[dict[str, Any]] = []
        for video_id, video_data in videos.items():
            metadata = r2_metadata.get(video_id, {})
            video_info = self._build_video_info(video_id, video_data, metadata)
            all_videos.append(video_info)

        # 채널 필터 적용
        if channel_filter:
            all_videos = [
                v for v in all_videos
                if v.get("channel_name") == channel_filter
            ]

        # 최근 업데이트 순 정렬
        all_videos.sort(
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )

        # 전체 개수
        total_count = len(all_videos)
        total_pages = (total_count + per_page - 1) // per_page if per_page > 0 else 1

        # 페이지네이션 적용
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_videos = all_videos[start_idx:end_idx]

        return {
            "videos": paginated_videos,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages
            }
        }

    def get_statistics(self) -> dict[str, Any]:
        """
        처리 상태 통계를 계산합니다.

        R2에서 실제 메타데이터를 가져와 풍부한 정보를 제공합니다.

        Returns:
            통계 딕셔너리 (total_videos, completed, failed, processing, recent_videos)
        """
        # 캐시된 결과 확인
        cached_stats = self._stats_cache.get("statistics")
        if cached_stats is not None:
            logger.debug("캐시된 통계 데이터 반환")
            return cached_stats

        # state.json은 video_id를 키로 하는 딕셔너리 형태
        videos = self.load_state()

        # 상태별 카운트
        total = len(videos)
        completed = 0
        failed = 0
        processing = 0

        # 최근 비디오 목록 (최대 10개)
        recent_videos: list[dict[str, Any]] = []

        # video_id 목록 추출
        video_ids = list(videos.keys())

        # R2에서 메타데이터 일괄 조회 (캐싱 적용)
        r2_metadata: dict[str, dict[str, Any]] = {}
        if self.r2_client:
            r2_metadata = self.r2_client.get_all_video_metadata(
                video_ids,
                cache=self._metadata_cache
            )
            logger.info(f"R2에서 {len(r2_metadata)}개 비디오 메타데이터 로드")

        for video_id, video_data in videos.items():
            status = video_data.get("status", "pending")

            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
            elif status in ("processing", "in_progress"):
                processing += 1

            # R2 메타데이터 가져오기
            metadata = r2_metadata.get(video_id, {})

            # 비디오 정보 구성
            video_info = self._build_video_info(video_id, video_data, metadata)
            recent_videos.append(video_info)

        # 최근 업데이트 순으로 정렬 (updated_at 기준)
        recent_videos.sort(
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )

        result = {
            "total_videos": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "recent_videos": recent_videos[:10]  # 최근 10개만
        }

        # 결과 캐싱
        self._stats_cache.set("statistics", result)

        return result


# ============================================
# 클라이언트 인스턴스 생성
# ============================================
r2_client = R2Client()
pinecone_client = PineconeClient()
# StateManager에 R2 클라이언트를 전달하여 메타데이터 조회 가능하게 함
state_manager = StateManager(LOCAL_DATA_DIR, r2_client=r2_client)


# ============================================
# 데이터 수집 함수
# ============================================
def collect_dashboard_data() -> dict[str, Any]:
    """
    대시보드에 표시할 모든 데이터를 수집합니다.

    Returns:
        템플릿에 전달할 데이터 딕셔너리
    """
    logger.info("대시보드 데이터 수집 시작")

    # 1. R2 스토리지 통계
    r2_stats = {
        "transcripts": r2_client.count_objects_in_folder("transcripts/"),
        "chunks": r2_client.count_objects_in_folder("chunks/"),
        "metadata": r2_client.count_objects_in_folder("metadata/")
    }

    # 2. Pinecone 통계
    pinecone_stats = {
        "total_vectors": pinecone_client.get_vector_count()
    }

    # 3. 처리 상태 통계
    state_stats = state_manager.get_statistics()

    # 4. 마지막 업데이트 시간
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 통합
    data = {
        "total_videos": state_stats["total_videos"],
        "completed": state_stats["completed"],
        "failed": state_stats["failed"],
        "processing": state_stats["processing"],
        "recent_videos": state_stats["recent_videos"],
        "r2_stats": r2_stats,
        "pinecone_stats": pinecone_stats,
        "last_updated": last_updated
    }

    logger.info(f"대시보드 데이터 수집 완료: {data}")
    return data


# ============================================
# API 엔드포인트
# ============================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """
    HTML 대시보드 페이지를 렌더링합니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        렌더링된 HTML 응답
    """
    try:
        data = collect_dashboard_data()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, **data}
        )
    except Exception as e:
        logger.error(f"대시보드 렌더링 실패: {e}")
        # 에러 발생 시 기본값으로 렌더링
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "total_videos": 0,
                "completed": 0,
                "failed": 0,
                "processing": 0,
                "recent_videos": [],
                "r2_stats": {"transcripts": 0, "chunks": 0, "metadata": 0},
                "pinecone_stats": {"total_vectors": 0},
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
        )


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """
    JSON 형식으로 전체 상태를 반환합니다.

    Returns:
        상태 데이터 딕셔너리
    """
    try:
        return collect_dashboard_data()
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        return {
            "error": str(e),
            "total_videos": 0,
            "completed": 0,
            "failed": 0,
            "processing": 0,
            "recent_videos": [],
            "r2_stats": {"transcripts": 0, "chunks": 0, "metadata": 0},
            "pinecone_stats": {"total_vectors": 0},
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    헬스체크 엔드포인트입니다.

    Returns:
        상태 메시지
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/video/{video_id}")
async def get_video_detail(video_id: str) -> dict[str, Any]:
    """
    특정 비디오의 상세 정보를 반환합니다.

    R2에서 메타데이터, 청크, 트랜스크립트 정보를 가져와
    풍부한 비디오 상세 정보를 제공합니다.

    Args:
        video_id: 비디오 ID

    Returns:
        비디오 상세 정보 딕셔너리

    Raises:
        HTTPException: 비디오를 찾을 수 없는 경우 (404)
    """
    try:
        # state.json에서 상태 확인
        state = state_manager.load_state()
        video_state = state.get(video_id)

        if not video_state:
            logger.warning(f"비디오를 찾을 수 없습니다: {video_id}")
            raise HTTPException(
                status_code=404,
                detail=f"비디오를 찾을 수 없습니다: {video_id}"
            )

        # R2에서 메타데이터 조회
        metadata = r2_client.get_video_metadata(video_id) or {}

        # R2에서 청크 데이터 조회
        chunks = r2_client.get_video_chunks(video_id) or []

        # R2에서 정제된 트랜스크립트 조회
        transcript = r2_client.get_refined_transcript(video_id) or {}

        # 상세 정보 구성
        result: dict[str, Any] = {
            "video_id": video_id,
            # 메타데이터 정보
            "title": metadata.get("title", f"Video {video_id}"),
            "channel_title": metadata.get("channel_title", "-"),
            "thumbnail_url": metadata.get("thumbnail_url", ""),
            "published_at": metadata.get("published_at", "-"),
            "summary": metadata.get("summary", ""),
            # 상태 정보 (state.json에서)
            "status": video_state.get("status", "pending"),
            "current_step": video_state.get("current_step", "-"),
            "updated_at": video_state.get("updated_at", "-"),
            "processed_at": metadata.get("processed_at", "-"),
            # 청크 정보
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_index": chunk.get("chunk_index", i),
                    "text": chunk.get("text", "")[:200] + "..."
                           if len(chunk.get("text", "")) > 200
                           else chunk.get("text", ""),
                    "context": chunk.get("context", ""),
                    "topics": chunk.get("topics", []),
                    "start_time": chunk.get("start_time", 0),
                    "end_time": chunk.get("end_time", 0)
                }
                for i, chunk in enumerate(chunks[:10])  # 처음 10개만
            ],
            # 트랜스크립트 정보 (요약만)
            "has_transcript": bool(transcript),
            "transcript_length": len(transcript.get("text", "")) if transcript else 0
        }

        logger.info(f"비디오 상세 정보 조회 성공: {video_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"비디오 상세 정보 조회 실패 ({video_id}): {e}")
        raise HTTPException(
            status_code=500,
            detail=f"비디오 정보 조회 중 오류 발생: {str(e)}"
        )


@app.delete("/api/cache")
async def clear_cache() -> dict[str, str]:
    """
    모든 캐시를 비웁니다.

    Returns:
        성공 메시지
    """
    state_manager._metadata_cache.clear()
    state_manager._stats_cache.clear()
    logger.info("캐시가 비워졌습니다.")
    return {"status": "success", "message": "캐시가 비워졌습니다."}


@app.get("/api/videos")
async def get_videos(
    page: int = 1,
    per_page: int = 20,
    channel: Optional[str] = None
) -> dict[str, Any]:
    """
    페이지네이션된 비디오 목록을 반환합니다.

    Args:
        page: 페이지 번호 (1부터 시작, 기본값 1)
        per_page: 페이지당 항목 수 (기본값 20, 최대 100)
        channel: 채널 필터 (선택적)

    Returns:
        페이지네이션된 비디오 목록과 메타데이터
    """
    try:
        # per_page 제한
        per_page = min(max(1, per_page), 100)
        page = max(1, page)

        result = state_manager.get_videos_paginated(
            page=page,
            per_page=per_page,
            channel_filter=channel
        )

        logger.info(
            f"비디오 목록 조회: page={page}, per_page={per_page}, "
            f"channel={channel}, total={result['pagination']['total_count']}"
        )
        return result

    except Exception as e:
        logger.error(f"비디오 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"비디오 목록 조회 중 오류 발생: {str(e)}"
        )


@app.get("/api/channels")
async def get_channels() -> dict[str, Any]:
    """
    모든 고유 채널 목록을 반환합니다.

    Returns:
        채널 목록과 개수
    """
    try:
        channels = state_manager.get_all_channels()
        logger.info(f"채널 목록 조회: {len(channels)}개")
        return {
            "channels": channels,
            "count": len(channels)
        }
    except Exception as e:
        logger.error(f"채널 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"채널 목록 조회 중 오류 발생: {str(e)}"
        )


# ============================================
# 서버 실행
# ============================================
if __name__ == "__main__":
    import uvicorn

    logger.info("Medical RAG 대시보드 서버 시작")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
