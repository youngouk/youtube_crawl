from typing import Any
import main


def test_pipeline_execution_success(mocker: Any) -> None:
    # Mock settings
    mocker.patch('main.settings.YOUTUBE_API_KEY', 'fake_yt_key')
    mocker.patch('main.settings.GOOGLE_API_KEY', 'fake_gemini_key')
    mocker.patch('main.settings.PINECONE_API_KEY', 'fake_pc_key')

    # Mock components
    mock_storage = mocker.patch('main.get_storage').return_value
    mock_state_mgr = mocker.patch('main.StateManager').return_value
    mock_channel_state_mgr = mocker.patch('main.ChannelStateManager').return_value
    mock_yt = mocker.patch('main.YouTubeCollector').return_value
    mock_tr = mocker.patch('main.TranscriptCollector').return_value
    mock_gemini = mocker.patch('main.GeminiProcessor').return_value
    mock_chunker = mocker.patch('main.Chunker').return_value
    mock_pinecone = mocker.patch('main.PineconeManager').return_value

    # ChannelStateManager 반환값 설정
    mock_channel_state_mgr.is_channel_completed.return_value = False

    # Setup Data
    mock_storage.exists.return_value = True
    mock_storage.load_json.return_value = [{"channel_id": "UC123", "name": "TestChannel"}]

    # get_channel_videos_sorted 사용 (조회수/최신순 정렬 지원)
    mock_yt.get_channel_videos_sorted.return_value = [{"video_id": "vid1", "title": "Test Video"}]

    mock_state_mgr.get_video_status.return_value = None  # Not completed

    # 타임스탬프 포함된 트랜스크립트
    mock_tr.get_transcript.return_value = [
        {"text": "Hello world", "start": 0.0, "duration": 5.0}
    ]

    mock_gemini.refine_transcript.return_value = "Refined Hello World"
    mock_gemini.summarize_video.return_value = "Summary"
    # 통합 함수 모킹 (context, topics를 한 번에 반환)
    mock_gemini.generate_chunk_context_and_topics.return_value = ("Context", ["발열", "해열제"])
    mock_gemini.get_embedding.return_value = [0.1, 0.2]

    # split_transcript_with_timestamps 사용 (타임스탬프 포함 청킹)
    mock_chunker.split_transcript_with_timestamps.return_value = [
        {"text": "Refined Hello World", "start_time": 0.0, "end_time": 5.0}
    ]

    # Execute
    main.main()

    # Assertions
    # 1. Processing flow
    # get_channel_videos_sorted 호출 확인 (정렬 지원)
    mock_yt.get_channel_videos_sorted.assert_called()
    mock_tr.get_transcript.assert_called_with("vid1")
    mock_gemini.refine_transcript.assert_called()
    mock_gemini.summarize_video.assert_called()
    # 통합 함수 호출 확인 (context + topics 한 번에)
    mock_gemini.generate_chunk_context_and_topics.assert_called()

    # 2. Storage
    # videos list
    mock_storage.save_json.assert_any_call(
        "raw/videos/UC123/list.json", [{"video_id": "vid1", "title": "Test Video"}]
    )
    # transcript raw
    mock_storage.save_json.assert_any_call(
        "transcripts/vid1/raw.json",
        [{"text": "Hello world", "start": 0.0, "duration": 5.0}]
    )
    # refined
    mock_storage.save_json.assert_any_call(
        "transcripts/vid1/refined.json", {"text": "Refined Hello World"}
    )

    # 3. Pinecone
    mock_pinecone.upsert_vectors.assert_called()

    # 4. State updates
    assert mock_state_mgr.update_video_status.call_count >= 4
    mock_state_mgr.update_video_status.assert_any_call("vid1", "completed")

def test_pipeline_skip_completed(mocker: Any) -> None:
    mocker.patch('main.settings.YOUTUBE_API_KEY', 'fake_yt_key')
    mock_storage = mocker.patch('main.get_storage').return_value
    mock_state_mgr = mocker.patch('main.StateManager').return_value
    mock_yt = mocker.patch('main.YouTubeCollector').return_value
    
    mock_storage.exists.return_value = True
    mock_storage.load_json.return_value = [{"channel_id": "UC123", "name": "TestChannel"}]
    mock_yt.get_channel_videos.return_value = [{"video_id": "vid1", "title": "Test Video"}]
    
    # Status is completed
    mock_state_mgr.get_video_status.return_value = {'status': 'completed'}
    
    mock_tr = mocker.patch('main.TranscriptCollector').return_value
    
    main.main()
    
    # Should skip transcript collection
    mock_tr.get_transcript.assert_not_called()
