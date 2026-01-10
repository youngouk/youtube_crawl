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
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

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
# 글로벌 캐시 (성능 최적화)
# ============================================
# R2 스토리지 통계 캐시 (5분 TTL)
_r2_stats_cache = TTLCache(ttl_seconds=300)
# 대시보드 데이터 캐시 (60초 TTL)
_dashboard_cache = TTLCache(ttl_seconds=60)
# 백그라운드 태스크 참조
_background_task: asyncio.Task | None = None


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

    def get_folder_stats(self, prefix: str) -> dict[str, Any]:
        """
        지정된 폴더(prefix) 내의 객체 수와 총 용량을 반환합니다.

        Args:
            prefix: 조회할 폴더 경로 (예: 'transcripts/')

        Returns:
            {'count': 객체 수, 'size_bytes': 총 용량, 'size_human': 읽기 쉬운 용량}
        """
        if not self.client:
            return {"count": 0, "size_bytes": 0, "size_human": "0 B"}

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            total_count = 0
            total_size = 0

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if not obj["Key"].endswith("/"):
                            total_count += 1
                            total_size += obj.get("Size", 0)

            return {
                "count": total_count,
                "size_bytes": total_size,
                "size_human": self._format_bytes(total_size)
            }
        except Exception as e:
            logger.error(f"R2 폴더 통계 조회 실패 ({prefix}): {e}")
            return {"count": 0, "size_bytes": 0, "size_human": "0 B"}

    def _format_bytes(self, size_bytes: int) -> str:
        """바이트 단위를 읽기 쉬운 형식으로 변환합니다."""
        if size_bytes == 0:
            return "0 B"
        units = ("B", "KB", "MB", "GB", "TB")
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"

    def count_objects_in_folder(self, prefix: str) -> int:
        """이전 버전과의 호환성을 위해 유지합니다."""
        stats = self.get_folder_stats(prefix)
        return stats["count"]

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

    def enrich_video_metadata(self, limit: int = 10) -> int:
        """
        메타데이터가 부족한 비디오(복구된 상태 등)를 찾아 R2에서 정보를 보강합니다.
        
        Args:
            limit: 한 번에 조회할 비디오 수
            
        Returns:
            업데이트된 비디오 수
        """
        if not self.r2_client:
            return 0

        # 상태 로드
        videos = self.load_state()
        if not videos:
            return 0

        # 보강이 필요한 비디오 식별 (채널명이 없거나 restored 플래그가 있는 경우)
        candidates = []
        for video_id, data in videos.items():
            if data.get("restored") or "channel_name" not in data or data.get("channel_name") == "Unknown":
                candidates.append(video_id)
        
        # 대상이 없으면 종료
        if not candidates:
            return 0

        # 제한된 수만큼 선택
        targets = candidates[:limit]
        updated_count = 0

        for video_id in targets:
            try:
                # R2에서 메타데이터 조회
                meta = self.r2_client.get_video_metadata(video_id)
                if meta:
                    # state 업데이트
                    videos[video_id]["title"] = meta.get("title", videos[video_id].get("title", ""))
                    videos[video_id]["channel_name"] = meta.get("channel_title", "Unknown")
                    videos[video_id]["summary"] = self._truncate_text(meta.get("summary", ""), 100)
                    videos[video_id]["thumbnail_url"] = meta.get("thumbnail_url", "")
                    videos[video_id]["chunk_count"] = meta.get("chunk_count", videos[video_id].get("chunk_count", 0))
                    
                    # restored 플래그 제거 및 업데이트 시간 갱신
                    if "restored" in videos[video_id]:
                        del videos[video_id]["restored"]
                    
                    updated_count += 1
            except Exception as e:
                logger.warning(f"비디오 {video_id} 메타데이터 보강 실패: {e}")

        # 변경사항 저장
        if updated_count > 0:
            try:
                # 백업용으로 기존 파일을 .bak으로 저장하지 않고 바로 덮어씀 (성능 우선)
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(videos, f, ensure_ascii=False, indent=2)
                logger.info(f"메타데이터 보강 완료: {updated_count}개 비디오 업데이트")
            except Exception as e:
                logger.error(f"state.json 업데이트 실패: {e}")

        return updated_count

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
        성능 최적화를 위해 메타데이터는 필요한 경우(최근 목록 등)에만 부분적으로 조회합니다.

        Returns:
            통계 딕셔너리
        """
        # 캐시된 결과 확인
        cached_stats = self._stats_cache.get("statistics")
        if cached_stats is not None:
            return cached_stats

        videos = self.load_state()

        # 상태별 카운트
        total = len(videos)
        completed = 0
        failed = 0
        processing = 0

        # 트렌드 및 분포 분석
        trends_map = defaultdict(int)
        distribution_map = defaultdict(int)
        health_summary = {"S": 0, "A": 0, "B": 0, "F": 0}

        # 최근 비디오 후보 리스트
        recent_candidates: list[tuple[str, dict[str, Any]]] = []

        for video_id, video_data in videos.items():
            status = video_data.get("status", "pending")
            
            # 상태 카운팅
            if status == "completed":
                completed += 1
                # 트렌드 (날짜별)
                updated_at = video_data.get("updated_at", "")
                if updated_at and "T" in updated_at:
                    trends_map[updated_at.split("T")[0]] += 1
                
                # 채널 분포 (state에 정보가 있을 경우만)
                ch_name = video_data.get("channel_name", "Unknown")
                if ch_name != "Unknown":
                    distribution_map[ch_name] += 1

                # 헬스 점수 (약식 계산: 메타데이터 로드 없이 state 정보로 판단)
                chunks = video_data.get("chunk_count", 0)
                if chunks != "-" and int(chunks) > 0:
                    # chunk가 있으면 최소 A, 요약까지 있으면 S라고 가정할 수 있으나
                    # 여기서는 안전하게 chunk 존재 시 A로 분류
                    health_summary["A"] += 1 
                else:
                    health_summary["B"] += 1

            elif status == "failed":
                failed += 1
                health_summary["F"] += 1
            elif status in ("processing", "in_progress"):
                processing += 1

            # 최근 비디오 후보 추가
            recent_candidates.append((video_id, video_data))

        # 최근 업데이트 순으로 정렬하여 상위 10개 추출
        recent_candidates.sort(
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True
        )
        top_10_candidates = recent_candidates[:10]

        # Top 10에 대해서만 R2 메타데이터 상세 조회
        recent_videos = []
        for video_id, video_data in top_10_candidates:
            # 기본 정보 구성
            video_info = {
                "video_id": video_id,
                "title": video_data.get("title", f"Video {video_id}"),
                "channel_name": video_data.get("channel_name", "Unknown"),
                "status": video_data.get("status", "pending"),
                "chunk_count": video_data.get("chunk_count", 0),
                "summary": video_data.get("summary", ""),
                "thumbnail_url": video_data.get("thumbnail_url", ""),
                "updated_at": self._format_updated_at(video_data.get("updated_at", "-"))
            }

            # 상세 메타데이터가 없는 경우(예: 복구된 상태) R2에서 조회 시도
            if self.r2_client and (video_info["title"].startswith("Video ") or not video_info["summary"]):
                meta = self.r2_client.get_video_metadata(video_id)
                if meta:
                    video_info["title"] = meta.get("title", video_info["title"])
                    video_info["channel_name"] = meta.get("channel_title", video_info["channel_name"])
                    video_info["summary"] = self._truncate_text(meta.get("summary", ""), 60)
                    video_info["thumbnail_url"] = meta.get("thumbnail_url", video_info["thumbnail_url"])
                    
                    # 배포 차트용 데이터 보정 (상위 10개에 대해서라도)
                    if meta.get("channel_title"):
                         distribution_map[meta.get("channel_title")] += 1

            recent_videos.append(video_info)

        # 정렬된 트렌드 데이터 (최근 14일)
        sorted_trends = sorted(trends_map.items())[-14:]
        # 정렬된 분포 데이터
        sorted_distribution = sorted(distribution_map.items(), key=lambda x: x[1], reverse=True)

        result = {
            "total_videos": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "recent_videos": recent_videos,
            "trends": [{"day": d, "count": c} for d, c in sorted_trends],
            "distribution": [{"channel": c, "count": v} for c, v in sorted_distribution],
            "health": health_summary
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
# 데이터 수집 함수 (성능 최적화)
# ============================================
def _fetch_r2_stats_uncached() -> dict[str, Any]:
    """R2 스토리지 통계를 가져옵니다 (캐시 없이)."""
    return {
        "transcripts": r2_client.get_folder_stats("transcripts/"),
        "chunks": r2_client.get_folder_stats("chunks/"),
        "metadata": r2_client.get_folder_stats("metadata/")
    }


def get_cached_r2_stats() -> dict[str, Any]:
    """캐시된 R2 스토리지 통계를 반환합니다."""
    cached = _r2_stats_cache.get("r2_stats")
    if cached is not None:
        return cached
    
    # 캐시 미스 시 새로 가져와서 캐싱
    stats = _fetch_r2_stats_uncached()
    _r2_stats_cache.set("r2_stats", stats)
    return stats


def collect_dashboard_data(force_refresh: bool = False) -> dict[str, Any]:
    """
    대시보드에 표시할 모든 데이터를 수집합니다.
    성능 최적화를 위해 캐시를 우선 사용합니다.

    Args:
        force_refresh: True면 캐시를 무시하고 새로 수집

    Returns:
        템플릿에 전달할 데이터 딕셔너리
    """
    # 캐시 우선 조회
    if not force_refresh:
        cached = _dashboard_cache.get("dashboard_data")
        if cached is not None:
            logger.debug("캐시된 대시보드 데이터 반환")
            return cached

    logger.info("대시보드 데이터 수집 시작")

    # 1. R2 스토리지 통계 (5분 캐시 사용)
    r2_stats = get_cached_r2_stats()

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

    # 결과 캐싱
    _dashboard_cache.set("dashboard_data", data)
    logger.info(f"대시보드 데이터 수집 완료")
    return data


# ============================================
# 백그라운드 캐시 워밍 (Prefetch)
# ============================================
async def background_cache_refresh():
    """
    백그라운드에서 주기적으로 캐시를 갱신하고 메타데이터를 보강합니다.
    """
    while True:
        try:
            await asyncio.sleep(60)  # 60초마다 갱신
            loop = asyncio.get_event_loop()
            
            # 1. 대시보드 데이터 갱신 (캐시 워밍)
            await loop.run_in_executor(None, collect_dashboard_data, True)
            
            # 2. 메타데이터 보강 (Enrichment)
            # 한 번에 20개씩 천천히 보강하여 서버 부하 방지
            updated = await loop.run_in_executor(None, state_manager.enrich_video_metadata, 20)
            if updated > 0:
                logger.info(f"백그라운드: {updated}개 비디오 메타데이터 보강됨")
            
            logger.debug("백그라운드 작업 완료")
            
        except asyncio.CancelledError:
            logger.info("백그라운드 태스크 종료")
            break
        except Exception as e:
            logger.error(f"백그라운드 작업 실패: {e}")
            await asyncio.sleep(30)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 캐시를 미리 워밍합니다."""
    global _background_task
    logger.info("서버 시작: 캐시 워밍 중...")
    try:
        # 초기 캐시 워밍 (동기 함수를 별도 스레드에서 실행)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, collect_dashboard_data, True)
        logger.info("초기 캐시 워밍 완료")
    except Exception as e:
        logger.error(f"초기 캐시 워밍 실패: {e}")
    
    # 백그라운드 갱신 태스크 시작
    _background_task = asyncio.create_task(background_cache_refresh())
    logger.info("백그라운드 캐시 갱신 태스크 시작")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 백그라운드 태스크를 정리합니다."""
    global _background_task
    if _background_task:
        _background_task.cancel()
        try:
            await _background_task
        except asyncio.CancelledError:
            pass
        logger.info("백그라운드 태스크 정리 완료")


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


@app.get("/api/stats/analysis")
async def get_analysis_stats() -> dict[str, Any]:
    """분석용 상세 통계를 반환합니다."""
    try:
        stats = state_manager.get_statistics()
        return {
            "trends": stats.get("trends", []),
            "distribution": stats.get("distribution", [])
        }
    except Exception as e:
        logger.error(f"분석 통계 조회 실패: {e}")
        return {"trends": [], "distribution": []}


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
