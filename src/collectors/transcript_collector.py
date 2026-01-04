from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from typing import Any, Optional
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class TranscriptCollector:
    """YouTube 자막을 수집합니다."""

    def __init__(self, use_proxy: bool = True) -> None:
        """
        TranscriptCollector 초기화.

        Args:
            use_proxy: True이면 Webshare 프록시 사용 (환경변수 필요)
        """
        proxy_config = None

        # Webshare 프록시 설정 (환경변수에서 읽기)
        if use_proxy:
            username = os.getenv("WEBSHARE_PROXY_USERNAME")
            password = os.getenv("WEBSHARE_PROXY_PASSWORD")

            if username and password:
                try:
                    from youtube_transcript_api.proxies import WebshareProxyConfig
                    proxy_config = WebshareProxyConfig(
                        proxy_username=username,
                        proxy_password=password,
                    )
                    print("[TranscriptCollector] Webshare 프록시 활성화됨")
                except ImportError:
                    print("[TranscriptCollector] 프록시 모듈을 찾을 수 없음, 직접 연결 사용")
            else:
                print("[TranscriptCollector] 프록시 인증 정보 없음, 직접 연결 사용")

        self.api = YouTubeTranscriptApi(proxy_config=proxy_config)

    def get_transcript(self, video_id: str) -> Optional[list[dict[str, Any]]]:
        """
        Fetches transcript for a video.
        우선순위:
        1. 수동 업로드 자막 (ko -> en)
        2. 자동 생성 자막 (ko -> en)

        Returns list of dicts with 'text', 'start', 'duration' keys.
        """
        try:
            # 1차: 수동 업로드 자막 시도 (한국어 우선)
            transcript = self.api.fetch(video_id, languages=['ko', 'en'])
            print(f"[{video_id}] Transcript fetched successfully (manual).")
            return self._convert_to_dict(transcript)

        except NoTranscriptFound:
            # 2차: 자동 생성 자막 시도
            return self._try_auto_generated(video_id)
        except TranscriptsDisabled:
            print(f"[{video_id}] Transcripts are disabled.")
            return None
        except Exception as e:
            print(f"[{video_id}] Error fetching transcript: {e}")
            return None

    def _try_auto_generated(self, video_id: str) -> Optional[list[dict[str, Any]]]:
        """
        자동 생성 자막을 가져옵니다.
        우선순위: 한국어 자동 자막 -> 영어 자동 자막
        """
        try:
            # 사용 가능한 모든 자막 목록 조회
            transcript_list = self.api.list(video_id)

            # 자동 생성 자막 찾기 (한국어 우선)
            for lang in ['ko', 'en']:
                for transcript_info in transcript_list:
                    if transcript_info.is_generated and transcript_info.language_code == lang:
                        transcript = transcript_info.fetch()
                        print(f"[{video_id}] Transcript fetched successfully (auto-generated, {lang}).")
                        return self._convert_to_dict(transcript)

            # 아무 자동 생성 자막이라도 있으면 사용
            for transcript_info in transcript_list:
                if transcript_info.is_generated:
                    transcript = transcript_info.fetch()
                    lang = transcript_info.language_code
                    print(f"[{video_id}] Transcript fetched successfully (auto-generated, {lang}).")
                    return self._convert_to_dict(transcript)

            print(f"[{video_id}] No suitable transcript found (including auto-generated).")
            return None

        except Exception as e:
            print(f"[{video_id}] Error fetching auto-generated transcript: {e}")
            return None

    def _convert_to_dict(self, transcript: Any) -> list[dict[str, Any]]:
        """FetchedTranscriptSnippet 객체를 dict 리스트로 변환합니다."""
        return [
            {
                'text': snippet.text,
                'start': snippet.start,
                'duration': snippet.duration
            }
            for snippet in transcript
        ]
