from typing import Any, Optional
import json
from datetime import datetime, timezone
from config.settings import settings


class StateManager:
    """비디오 처리 상태를 로컬 JSON 파일로 관리합니다."""

    def __init__(self) -> None:
        # For simplicity in this PoC, we will primarily use local state tracking
        # Ideally this would be DynamoDB
        self.local_state_file = settings.LOCAL_DATA_DIR / "state.json"

        if not self.local_state_file.exists():
            self._save_state({})

    def _load_state(self) -> dict[str, Any]:
        with open(self.local_state_file, 'r', encoding='utf-8') as f:
            result: dict[str, Any] = json.load(f)
            return result

    def _save_state(self, state: dict[str, Any]) -> None:
        with open(self.local_state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def update_video_status(
        self,
        video_id: str,
        status: str,
        step: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Updates the processing status of a video.

        Args:
            video_id: YouTube 비디오 ID
            status: 상태 (pending, processing, completed, failed)
            step: 현재 처리 단계 (선택)
            error: 에러 메시지 (선택, failed 상태 시)
        """
        state = self._load_state()

        if video_id not in state:
            state[video_id] = {'retry_count': 0}

        current = state[video_id]
        current['status'] = status

        if step:
            current['current_step'] = step

        if error:
            current['error_message'] = error

        # 실패 시 retry_count 증가
        if status == 'failed':
            current['retry_count'] = current.get('retry_count', 0) + 1

        current['updated_at'] = datetime.now(timezone.utc).isoformat()

        state[video_id] = current
        self._save_state(state)

    def get_video_status(self, video_id: str) -> Optional[dict[str, Any]]:
        state = self._load_state()
        return state.get(video_id)
