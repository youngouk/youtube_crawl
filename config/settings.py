import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """애플리케이션 설정을 관리합니다."""

    # YouTube Data API
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")

    # Google Gemini API
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # OpenRouter API (폴백용)
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # Pinecone
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV", "us-east-1")
    PINECONE_INDEX_NAME: str = "medical-rag-hybrid"

    # Storage (R2 / Local)
    R2_ENDPOINT_URL: Optional[str] = os.getenv("R2_ENDPOINT_URL")
    R2_ACCESS_KEY_ID: Optional[str] = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY: Optional[str] = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "medical-rag-bucket")

    # Defaults to True if R2 credentials are missing
    USE_LOCAL_STORAGE: bool = os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true"
    LOCAL_DATA_DIR: Path = Path(os.getenv("LOCAL_DATA_DIR", "./data"))

    # Processing limits
    MAX_RETRIES: int = 3

    # 비디오 수집 설정
    # VIDEO_SORT_BY: "recent" (최신순) 또는 "views" (조회수 높은순)
    VIDEO_SORT_BY: str = os.getenv("VIDEO_SORT_BY", "recent")
    # 조회수 정렬 시 후보 비디오 수 (이 중에서 상위 N개 선택)
    VIDEO_FETCH_POOL: int = int(os.getenv("VIDEO_FETCH_POOL", "200"))

    @classmethod
    def validate(cls) -> None:
        """Simple validation to check if critical keys are missing when needed."""
        if not cls.USE_LOCAL_STORAGE:
            if not all([cls.R2_ENDPOINT_URL, cls.R2_ACCESS_KEY_ID, cls.R2_SECRET_ACCESS_KEY]):
                raise ValueError("R2 credentials are required when USE_LOCAL_STORAGE is false.")

        if not cls.GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY is missing. Gemini features will fail.")


settings = Settings()
