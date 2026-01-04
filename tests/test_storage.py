import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from src.storage.r2_storage import LocalStorage, R2Storage, get_storage


def test_local_storage_save_load(tmp_path: Path) -> None:
    storage = LocalStorage(base_dir=tmp_path)
    data = {"key": "value"}
    path = "test/file.json"
    
    # Save
    storage.save_json(path, data)
    
    # Check physical file exists
    assert (tmp_path / path).exists()
    
    # Load
    loaded = storage.load_json(path)
    assert loaded == data

def test_local_storage_exists(tmp_path: Path) -> None:
    storage = LocalStorage(base_dir=tmp_path)
    path = "test/file.json"
    
    assert not storage.exists(path)
    storage.save_json(path, {})
    assert storage.exists(path)

def test_r2_storage_save(mocker: Any) -> None:
    mock_boto = mocker.patch('src.storage.r2_storage.boto3.client')
    mock_s3 = mock_boto.return_value
    
    storage = R2Storage()
    # Mock settings to be sure? Assuming imported settings usage
    
    data = {"key": "value"}
    storage.save_json("test.json", data)
    
    mock_s3.put_object.assert_called_once()
    call_args = mock_s3.put_object.call_args[1]
    assert call_args['Key'] == "test.json"
    assert json.loads(call_args['Body']) == data
    
def test_r2_storage_load_success(mocker: Any) -> None:
    mock_boto = mocker.patch('src.storage.r2_storage.boto3.client')
    mock_s3 = mock_boto.return_value
    
    # Mock get_object response body
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"key": "val"}).encode('utf-8')
    mock_s3.get_object.return_value = {"Body": mock_body}
    
    storage = R2Storage()
    result = storage.load_json("test.json")
    
    assert result == {"key": "val"}

def test_get_storage_factory(mocker: Any) -> None:
    # Test valid environment switch
    mock_settings = mocker.patch('src.storage.r2_storage.settings')
    
    mock_settings.USE_LOCAL_STORAGE = True
    mock_settings.LOCAL_DATA_DIR = Path("/tmp")
    
    # Set R2 settings to strings to avoid boto3/urllib issues with Mocks
    mock_settings.R2_ENDPOINT_URL = "https://r2.example.com"
    mock_settings.R2_ACCESS_KEY_ID = "key"
    mock_settings.R2_SECRET_ACCESS_KEY = "secret"
    mock_settings.R2_BUCKET_NAME = "bucket"
    
    s = get_storage()
    assert isinstance(s, LocalStorage)
    
    mock_settings.USE_LOCAL_STORAGE = False
    
    # We also need to mock boto3 inside get_storage -> R2Storage
    mocker.patch('src.storage.r2_storage.boto3.client')
    
    s = get_storage()
    assert isinstance(s, R2Storage)
