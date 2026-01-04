from typing import Any
from unittest.mock import MagicMock
from src.collectors.transcript_collector import TranscriptCollector


def test_get_transcript_success(mocker: Any) -> None:
    """자막 조회 성공 테스트"""
    # YouTubeTranscriptApi 클래스 자체를 모킹
    mock_api_class = mocker.patch('src.collectors.transcript_collector.YouTubeTranscriptApi')
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    # fetch 결과 모킹 (FetchedTranscriptSnippet 객체 형태)
    mock_snippet = MagicMock()
    mock_snippet.text = "안녕하세요"
    mock_snippet.start = 0.0
    mock_snippet.duration = 1.5
    mock_api_instance.fetch.return_value = [mock_snippet]

    collector = TranscriptCollector()
    result = collector.get_transcript("vid1")

    assert result is not None
    assert len(result) == 1
    assert result[0]['text'] == "안녕하세요"
    assert result[0]['start'] == 0.0
    assert result[0]['duration'] == 1.5

    # fetch가 올바르게 호출되었는지 검증
    mock_api_instance.fetch.assert_called_once_with("vid1", languages=['ko', 'en'])


def test_get_transcript_transcripts_disabled(mocker: Any) -> None:
    """자막 비활성화 시 None 반환 테스트"""
    from youtube_transcript_api._errors import TranscriptsDisabled

    mock_api_class = mocker.patch('src.collectors.transcript_collector.YouTubeTranscriptApi')
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    # TranscriptsDisabled 예외 발생
    mock_api_instance.fetch.side_effect = TranscriptsDisabled("vid1")

    collector = TranscriptCollector()
    result = collector.get_transcript("vid1")

    assert result is None


def test_get_transcript_no_transcript_found(mocker: Any) -> None:
    """자막이 없을 때 None 반환 테스트"""
    from youtube_transcript_api._errors import NoTranscriptFound

    mock_api_class = mocker.patch('src.collectors.transcript_collector.YouTubeTranscriptApi')
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    # NoTranscriptFound 예외 발생
    mock_api_instance.fetch.side_effect = NoTranscriptFound("vid1", ["ko", "en"], None)

    collector = TranscriptCollector()
    result = collector.get_transcript("vid1")

    assert result is None


def test_get_transcript_generic_error(mocker: Any) -> None:
    """일반 오류 시 None 반환 테스트"""
    mock_api_class = mocker.patch('src.collectors.transcript_collector.YouTubeTranscriptApi')
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    # 일반 예외 발생
    mock_api_instance.fetch.side_effect = Exception("Network error")

    collector = TranscriptCollector()
    result = collector.get_transcript("vid1")

    assert result is None


def test_get_transcript_multiple_segments(mocker: Any) -> None:
    """여러 세그먼트 자막 처리 테스트"""
    mock_api_class = mocker.patch('src.collectors.transcript_collector.YouTubeTranscriptApi')
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance

    # 여러 세그먼트 모킹
    segments = []
    for i in range(3):
        mock_snippet = MagicMock()
        mock_snippet.text = f"세그먼트 {i}"
        mock_snippet.start = float(i * 5)
        mock_snippet.duration = 4.5
        segments.append(mock_snippet)

    mock_api_instance.fetch.return_value = segments

    collector = TranscriptCollector()
    result = collector.get_transcript("vid1")

    assert result is not None
    assert len(result) == 3
    assert result[0]['text'] == "세그먼트 0"
    assert result[1]['text'] == "세그먼트 1"
    assert result[2]['text'] == "세그먼트 2"
    assert result[1]['start'] == 5.0
