"""
채널별 처리 상태를 관리하는 모듈입니다.

각 채널의 처리 진행 상황, 완료된 비디오 수, 실패 비디오 수 등을
추적하고 다중 채널 자동 순회 처리를 지원합니다.
"""
from typing import Any, Optional
import json
from datetime import datetime, timezone
from config.settings import settings


class ChannelStateManager:
    """
    채널별 처리 상태를 로컬 JSON 파일로 관리합니다.

    상태 파일 구조:
    {
        "channel_id": {
            "name": "채널 이름",
            "status": "pending|processing|completed|failed",
            "total_videos": 50,
            "processed_videos": 30,
            "failed_videos": 5,
            "no_transcript_videos": 10,
            "skipped_videos": 5,
            "started_at": "2025-01-01T00:00:00Z",
            "completed_at": "2025-01-01T01:00:00Z",
            "updated_at": "2025-01-01T01:00:00Z",
            "current_video_index": 35,
            "error_summary": {"No transcript": 10, "Refinement failed": 3}
        }
    }
    """

    def __init__(self) -> None:
        """채널 상태 관리자를 초기화합니다."""
        self.state_file = settings.LOCAL_DATA_DIR / "channel_state.json"

        if not self.state_file.exists():
            self._save_state({})

    def _load_state(self) -> dict[str, Any]:
        """상태 파일을 로드합니다."""
        with open(self.state_file, 'r', encoding='utf-8') as f:
            result: dict[str, Any] = json.load(f)
            return result

    def _save_state(self, state: dict[str, Any]) -> None:
        """상태를 파일에 저장합니다."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def init_channel(
        self,
        channel_id: str,
        channel_name: str,
        total_videos: int
    ) -> None:
        """
        채널 처리를 시작할 때 초기 상태를 설정합니다.

        Args:
            channel_id: YouTube 채널 ID
            channel_name: 채널 이름
            total_videos: 처리할 총 비디오 수
        """
        state = self._load_state()

        # 이미 완료된 채널은 초기화하지 않음
        if channel_id in state and state[channel_id].get('status') == 'completed':
            return

        state[channel_id] = {
            'name': channel_name,
            'status': 'processing',
            'total_videos': total_videos,
            'processed_videos': 0,
            'failed_videos': 0,
            'no_transcript_videos': 0,
            'skipped_videos': 0,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'completed_at': None,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'current_video_index': 0,
            'error_summary': {}
        }

        self._save_state(state)

    def update_video_result(
        self,
        channel_id: str,
        success: bool,
        skipped: bool = False,
        error_type: Optional[str] = None
    ) -> None:
        """
        비디오 처리 결과를 업데이트합니다.

        Args:
            channel_id: YouTube 채널 ID
            success: 처리 성공 여부
            skipped: 이미 처리된 비디오로 스킵되었는지 여부
            error_type: 실패 시 에러 유형 (예: "No transcript", "Refinement failed")
        """
        state = self._load_state()

        if channel_id not in state:
            return

        channel_state = state[channel_id]
        channel_state['current_video_index'] += 1

        if skipped:
            channel_state['skipped_videos'] += 1
        elif success:
            channel_state['processed_videos'] += 1
        else:
            channel_state['failed_videos'] += 1

            # 에러 유형별 집계
            if error_type:
                if error_type == 'No transcript':
                    channel_state['no_transcript_videos'] += 1

                error_summary = channel_state.get('error_summary', {})
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
                channel_state['error_summary'] = error_summary

        channel_state['updated_at'] = datetime.now(timezone.utc).isoformat()

        self._save_state(state)

    def complete_channel(self, channel_id: str) -> None:
        """
        채널 처리를 완료로 표시합니다.

        Args:
            channel_id: YouTube 채널 ID
        """
        state = self._load_state()

        if channel_id not in state:
            return

        state[channel_id]['status'] = 'completed'
        state[channel_id]['completed_at'] = datetime.now(timezone.utc).isoformat()
        state[channel_id]['updated_at'] = datetime.now(timezone.utc).isoformat()

        self._save_state(state)

    def get_channel_status(self, channel_id: str) -> Optional[dict[str, Any]]:
        """
        특정 채널의 상태를 반환합니다.

        Args:
            channel_id: YouTube 채널 ID

        Returns:
            채널 상태 딕셔너리 또는 None
        """
        state = self._load_state()
        return state.get(channel_id)

    def is_channel_completed(self, channel_id: str) -> bool:
        """
        채널이 이미 완료되었는지 확인합니다.

        Args:
            channel_id: YouTube 채널 ID

        Returns:
            완료 여부
        """
        status = self.get_channel_status(channel_id)
        return status is not None and status.get('status') == 'completed'

    def get_all_channels_status(self) -> dict[str, Any]:
        """
        모든 채널의 상태를 반환합니다.

        Returns:
            전체 채널 상태 딕셔너리
        """
        return self._load_state()

    def get_summary(self) -> dict[str, Any]:
        """
        전체 처리 현황 요약을 반환합니다.

        Returns:
            요약 정보:
            - total_channels: 전체 채널 수
            - completed_channels: 완료된 채널 수
            - processing_channels: 처리 중 채널 수
            - total_videos_processed: 전체 처리된 비디오 수
            - total_videos_failed: 전체 실패 비디오 수
            - channels: 채널별 간단 현황
        """
        state = self._load_state()

        total_channels = len(state)
        completed_channels = sum(1 for c in state.values() if c.get('status') == 'completed')
        processing_channels = sum(1 for c in state.values() if c.get('status') == 'processing')

        total_videos_processed = sum(c.get('processed_videos', 0) for c in state.values())
        total_videos_failed = sum(c.get('failed_videos', 0) for c in state.values())
        total_no_transcript = sum(c.get('no_transcript_videos', 0) for c in state.values())

        channels_summary = []
        for channel_id, channel_state in state.items():
            channels_summary.append({
                'channel_id': channel_id,
                'name': channel_state.get('name', 'Unknown'),
                'status': channel_state.get('status', 'unknown'),
                'progress': f"{channel_state.get('current_video_index', 0)}/{channel_state.get('total_videos', 0)}",
                'processed': channel_state.get('processed_videos', 0),
                'failed': channel_state.get('failed_videos', 0),
                'no_transcript': channel_state.get('no_transcript_videos', 0),
                'skipped': channel_state.get('skipped_videos', 0)
            })

        return {
            'total_channels': total_channels,
            'completed_channels': completed_channels,
            'processing_channels': processing_channels,
            'total_videos_processed': total_videos_processed,
            'total_videos_failed': total_videos_failed,
            'total_no_transcript': total_no_transcript,
            'channels': channels_summary
        }

    def reset_channel(self, channel_id: str) -> bool:
        """
        특정 채널의 상태를 리셋합니다 (재처리를 위해).

        Args:
            channel_id: YouTube 채널 ID

        Returns:
            리셋 성공 여부
        """
        state = self._load_state()

        if channel_id in state:
            del state[channel_id]
            self._save_state(state)
            return True

        return False
