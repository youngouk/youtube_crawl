from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime, timezone
from config.settings import settings

class StateManager:
    def __init__(self):
        # For simplicity in this PoC, we will primarily use local state tracking
        # Ideally this would be DynamoDB
        self.local_state_file = settings.LOCAL_DATA_DIR / "state.json"
        
        if not self.local_state_file.exists():
            self._save_state({})

    def _load_state(self) -> Dict[str, Any]:
        with open(self.local_state_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_state(self, state: Dict[str, Any]):
        with open(self.local_state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def update_video_status(self, video_id: str, status: str, step: str = None, error: str = None):
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

    def get_video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        state = self._load_state()
        return state.get(video_id)
