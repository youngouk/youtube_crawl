import json
import boto3  # type: ignore[import-untyped]
from pathlib import Path
from typing import Any, Optional
from config.settings import settings


class StorageInterface:
    """스토리지 인터페이스 기본 클래스입니다."""
    def save_json(self, path: str, data: Any) -> None:
        raise NotImplementedError

    def load_json(self, path: str) -> Optional[Any]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

class LocalStorage(StorageInterface):
    """로컬 파일시스템 기반 스토리지입니다."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, path: str) -> Path:
        return self.base_dir / path

    def save_json(self, path: str, data: Any) -> None:
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[LocalStorage] Saved: {path}")

    def load_json(self, path: str) -> Optional[Any]:
        full_path = self._get_full_path(path)
        if not full_path.exists():
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def exists(self, path: str) -> bool:
        return self._get_full_path(path).exists()

class R2Storage(StorageInterface):
    """Cloudflare R2 기반 스토리지입니다."""

    s3: Any  # boto3 S3 클라이언트

    def __init__(self) -> None:
        self.s3 = boto3.client(
            's3',
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY
        )
        self.bucket = settings.R2_BUCKET_NAME

    def save_json(self, path: str, data: Any) -> None:
        try:
            body = json.dumps(data, ensure_ascii=False)
            self.s3.put_object(Bucket=self.bucket, Key=path, Body=body, ContentType='application/json')
            print(f"[R2Storage] Saved: {path}")
        except Exception as e:
            print(f"[R2Storage] Error saving {path}: {e}")
            raise

    def load_json(self, path: str) -> Optional[Any]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"[R2Storage] Error loading {path}: {e}")
            return None

    def exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

def get_storage() -> StorageInterface:
    if settings.USE_LOCAL_STORAGE:
        return LocalStorage(settings.LOCAL_DATA_DIR)
    else:
        return R2Storage()
