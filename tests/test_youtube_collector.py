from typing import Any
from src.collectors.youtube_collector import YouTubeCollector


def test_get_channel_uploads_playlist_id_success(mocker: Any) -> None:
    # Mock the build function
    mock_build = mocker.patch('src.collectors.youtube_collector.build')
    
    # Mock method chains: youtube.channels().list().execute()
    mock_youtube = mock_build.return_value
    mock_channels = mock_youtube.channels.return_value
    mock_list = mock_channels.list.return_value
    
    # Set return value
    mock_list.execute.return_value = {
        "items": [
            {
                "contentDetails": {
                    "relatedPlaylists": {
                        "uploads": "UU12345"
                    }
                }
            }
        ]
    }
    
    collector = YouTubeCollector()
    playlist_id = collector.get_channel_uploads_playlist_id("UC12345")
    
    assert playlist_id == "UU12345"
    mock_channels.list.assert_called_with(part="contentDetails", id="UC12345")

def test_get_channel_uploads_playlist_id_not_found(mocker: Any) -> None:
    mock_build = mocker.patch('src.collectors.youtube_collector.build')
    mock_youtube = mock_build.return_value
    mock_youtube.channels.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    
    collector = YouTubeCollector()
    playlist_id = collector.get_channel_uploads_playlist_id("UC_INVALID")
    
    assert playlist_id is None

def test_get_videos_from_playlist(mocker: Any) -> None:
    mock_build = mocker.patch('src.collectors.youtube_collector.build')
    mock_youtube = mock_build.return_value
    
    # Mock playlistItems().list().execute()
    mock_playlist = mock_youtube.playlistItems.return_value
    mock_list = mock_playlist.list.return_value
    
    # Mock response with one video
    mock_list.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "resourceId": {"videoId": "vid1"},
                    "title": "Test Video",
                    "description": "Desc",
                    "publishedAt": "2024-01-01",
                    "channelId": "UC123",
                    "channelTitle": "Test Channel",
                    "thumbnails": {"high": {"url": "http://thumb"}}
                }
            }
        ],
        "nextPageToken": None
    }
    
    collector = YouTubeCollector()
    videos = collector.get_videos_from_playlist("UU123")
    
    assert len(videos) == 1
    assert videos[0]['video_id'] == "vid1"
    assert videos[0]['title'] == "Test Video"
