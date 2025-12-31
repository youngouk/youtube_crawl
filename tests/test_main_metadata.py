"""
메타데이터 생성 함수 테스트
Pinecone에 저장될 청크 메타데이터의 필수 필드를 검증합니다.
"""
import pytest
from datetime import datetime


def test_chunk_metadata_has_required_fields():
    """청크 메타데이터에 필수 필드가 포함되어 있는지 검증"""
    required_fields = [
        'text',
        'context',
        'video_id',
        'video_title',
        'channel_name',
        'chunk_index',
        # 추가 필수 필드
        'source_type',
        'video_url',
        'published_at',
        'is_verified_professional',
        'specialty',
        'processed_at'
    ]

    # 테스트용 메타데이터 생성 함수 import
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트 청크",
        context="테스트 컨텍스트",
        video_id="test123",
        video_title="테스트 비디오",
        channel_name="테스트 채널",
        chunk_index=0,
        published_at="2024-01-01T00:00:00Z",
        is_verified_professional=True,
        specialty="소아과"
    )

    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"


def test_chunk_metadata_source_type():
    """source_type이 'youtube'로 설정되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="abc123",
        video_title="제목",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과"
    )

    assert metadata['source_type'] == 'youtube'


def test_chunk_metadata_video_url_format():
    """video_url이 올바른 형식인지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="dQw4w9WgXcQ",
        video_title="제목",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=False,
        specialty=""
    )

    assert metadata['video_url'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'


def test_chunk_metadata_processed_at_is_set():
    """processed_at이 ISO 형식으로 설정되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="test",
        video_title="제목",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과"
    )

    # ISO 8601 형식인지 확인 (예: 2024-12-31T10:30:00+00:00)
    assert 'processed_at' in metadata
    assert 'T' in metadata['processed_at']  # ISO 형식에는 T가 포함됨


def test_chunk_metadata_preserves_input_values():
    """입력값이 올바르게 보존되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="원본 텍스트",
        context="원본 컨텍스트",
        video_id="vid123",
        video_title="비디오 제목",
        channel_name="채널 이름",
        chunk_index=5,
        published_at="2024-06-15T12:00:00Z",
        is_verified_professional=True,
        specialty="소아과"
    )

    assert metadata['text'] == "원본 텍스트"
    assert metadata['context'] == "원본 컨텍스트"
    assert metadata['video_id'] == "vid123"
    assert metadata['video_title'] == "비디오 제목"
    assert metadata['channel_name'] == "채널 이름"
    assert metadata['chunk_index'] == 5
    assert metadata['published_at'] == "2024-06-15T12:00:00Z"
    assert metadata['is_verified_professional'] is True
    assert metadata['specialty'] == "소아과"
