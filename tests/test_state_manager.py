import json
from pathlib import Path
from typing import Any
from src.storage.state_manager import StateManager


def test_state_manager_init(tmp_path: Path, mocker: Any) -> None:
    # Mock settings to point to tmp_path
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    # StateManager 생성 시 state.json 파일이 자동 생성됨
    _manager = StateManager()  # noqa: F841
    state_file = tmp_path / "state.json"

    assert state_file.exists()
    assert json.loads(state_file.read_text()) == {}

def test_update_video_status(tmp_path: Path, mocker: Any) -> None:
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    manager = StateManager()

    manager.update_video_status("vid1", "processing", "step1")

    status = manager.get_video_status("vid1")
    assert status is not None
    assert status['status'] == "processing"
    assert status['current_step'] == "step1"
    assert "updated_at" in status

    # Update error
    manager.update_video_status("vid1", "failed", error="Failure")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status['status'] == "failed"
    assert status['error_message'] == "Failure"


def test_retry_count_tracking(tmp_path: Path, mocker: Any) -> None:
    """실패 시 retry_count가 증가하는지 검증"""
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    manager = StateManager()

    # 첫 번째 시도 실패
    manager.update_video_status("vid1", "failed", error="Error 1")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status.get('retry_count', 0) == 1

    # 두 번째 시도 실패
    manager.update_video_status("vid1", "failed", error="Error 2")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status['retry_count'] == 2

    # 성공 시 retry_count 유지 (리셋하지 않음)
    manager.update_video_status("vid1", "completed")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status['retry_count'] == 2


def test_retry_count_not_incremented_on_success(tmp_path: Path, mocker: Any) -> None:
    """성공 상태에서는 retry_count가 증가하지 않음"""
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    manager = StateManager()

    manager.update_video_status("vid1", "processing")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status.get('retry_count', 0) == 0

    manager.update_video_status("vid1", "completed")
    status = manager.get_video_status("vid1")
    assert status is not None
    assert status.get('retry_count', 0) == 0
