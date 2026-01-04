from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from typing import Any, Optional, cast
from config.settings import settings


class YouTubeCollector:
    """YouTube Data API를 사용하여 채널 및 비디오 정보를 수집합니다."""

    youtube: Any  # Resource 타입

    def __init__(self) -> None:
        self.youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)

    def get_channel_id_by_handle(self, handle: str) -> Optional[str]:
        """
        채널 핸들(@username)로 채널 ID를 조회합니다.

        Args:
            handle: 채널 핸들 (예: "@Dr.HaJungHoon119" 또는 "Dr.HaJungHoon119")

        Returns:
            채널 ID (예: "UC...") 또는 None
        """
        # @ 제거
        clean_handle = handle.lstrip('@')

        try:
            request = self.youtube.channels().list(
                part="id",
                forHandle=clean_handle
            )
            response = request.execute()

            if not response.get('items'):
                print(f"채널을 찾을 수 없습니다: @{clean_handle}")
                return None

            channel_id: str = response['items'][0]['id']
            print(f"채널 ID 조회 성공: @{clean_handle} → {channel_id}")
            return channel_id

        except HttpError as e:
            print(f"HTTP 오류 {e.resp.status}: {e.content}")
            return None

    def get_channel_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """Retrieves the 'uploads' playlist ID for a given channel."""
        try:
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            )
            response = request.execute()

            # items 키가 없거나 비어있는 경우 처리
            if not response.get('items'):
                print(f"채널을 찾을 수 없습니다: {channel_id} (삭제됨/비공개/ID 오류)")
                return None

            playlist_id: str = cast(
                str,
                response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            )
            return playlist_id
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            return None

    def get_videos_from_playlist(self, playlist_id: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Retrieves video metadata from a playlist."""
        videos: list[dict[str, Any]] = []
        next_page_token = None
        
        while True:
            try:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=50, # Max allowed by API per page
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    snippet = item['snippet']
                    video_id = snippet['resourceId']['videoId']
                    
                    video_data = {
                        "video_id": video_id,
                        "title": snippet['title'],
                        "description": snippet['description'],
                        "published_at": snippet['publishedAt'],
                        "channel_id": snippet['channelId'],
                        "channel_title": snippet['channelTitle'],
                        "thumbnail_url": snippet['thumbnails'].get('high', {}).get('url')
                    }
                    videos.append(video_data)
                    
                    if len(videos) >= max_results:
                        return videos

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred: {e.content}")
                break
                
        return videos

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Wrapper to get videos for a channel."""
        playlist_id = self.get_channel_uploads_playlist_id(channel_id)
        if not playlist_id:
            return []

        return self.get_videos_from_playlist(playlist_id, max_results)

    def get_video_statistics(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        비디오 ID 목록에 대한 통계 정보(조회수 등)를 조회합니다.

        Args:
            video_ids: 비디오 ID 목록 (최대 50개씩 처리)

        Returns:
            {video_id: {"view_count": int, "like_count": int, ...}} 형태의 딕셔너리
        """
        stats: dict[str, dict[str, Any]] = {}

        # API는 한 번에 50개까지만 조회 가능
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            try:
                request = self.youtube.videos().list(
                    part="statistics",
                    id=",".join(batch)
                )
                response = request.execute()

                for item in response.get('items', []):
                    video_id = item['id']
                    statistics = item.get('statistics', {})
                    stats[video_id] = {
                        "view_count": int(statistics.get('viewCount', 0)),
                        "like_count": int(statistics.get('likeCount', 0)),
                        "comment_count": int(statistics.get('commentCount', 0))
                    }
            except HttpError as e:
                print(f"통계 조회 HTTP 오류 {e.resp.status}: {e.content}")

        return stats

    def get_channel_videos_sorted(
        self,
        channel_id: str,
        max_results: int = 50,
        sort_by: str = "recent",
        fetch_pool: int = 200
    ) -> list[dict[str, Any]]:
        """
        채널의 비디오를 정렬 기준에 따라 가져옵니다.

        Args:
            channel_id: YouTube 채널 ID
            max_results: 최종 반환할 비디오 수
            sort_by: 정렬 기준 ("recent": 최신순, "views": 조회수순)
            fetch_pool: 조회수 정렬 시 후보 비디오 수

        Returns:
            정렬된 비디오 메타데이터 목록
        """
        playlist_id = self.get_channel_uploads_playlist_id(channel_id)
        if not playlist_id:
            return []

        if sort_by == "recent":
            # 최신순: 기존 로직 그대로
            return self.get_videos_from_playlist(playlist_id, max_results)

        elif sort_by == "views":
            # 조회수순: 더 많은 비디오를 가져온 후 정렬
            print(f"[YouTubeCollector] 조회수 기준 정렬 - {fetch_pool}개 후보 중 상위 {max_results}개 선택")

            # 1. 후보 비디오 가져오기
            videos = self.get_videos_from_playlist(playlist_id, fetch_pool)
            if not videos:
                return []

            # 2. 조회수 통계 조회
            video_ids = [v['video_id'] for v in videos]
            stats = self.get_video_statistics(video_ids)

            # 3. 각 비디오에 조회수 추가
            for video in videos:
                video_stat = stats.get(video['video_id'], {})
                video['view_count'] = video_stat.get('view_count', 0)
                video['like_count'] = video_stat.get('like_count', 0)

            # 4. 조회수 기준 내림차순 정렬
            videos.sort(key=lambda x: x.get('view_count', 0), reverse=True)

            # 5. 상위 N개 반환
            return videos[:max_results]

        else:
            print(f"[YouTubeCollector] 알 수 없는 정렬 기준: {sort_by}, 기본값(recent) 사용")
            return self.get_videos_from_playlist(playlist_id, max_results)
