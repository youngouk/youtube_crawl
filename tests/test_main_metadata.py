"""
메타데이터 생성 함수 테스트
Pinecone에 저장될 청크 메타데이터의 필수 필드를 검증합니다.
"""


def test_chunk_metadata_has_required_fields() -> None:
    """청크 메타데이터에 필수 필드가 포함되어 있는지 검증"""
    required_fields = [
        'text',
        'context',
        'video_id',
        'video_title',
        'channel_id',
        'channel_name',
        'chunk_index',
        # 추가 필수 필드
        'source_type',
        'video_url',
        'published_at',
        'is_verified_professional',
        'specialty',
        'credentials',
        'timestamp_start',
        'timestamp_end',
        'topics',
        'processed_at'
    ]

    # 테스트용 메타데이터 생성 함수 import
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트 청크",
        context="테스트 컨텍스트",
        video_id="test123",
        video_title="테스트 비디오",
        channel_id="UC123456",
        channel_name="테스트 채널",
        chunk_index=0,
        published_at="2024-01-01T00:00:00Z",
        is_verified_professional=True,
        specialty="소아과",
        credentials="소아청소년과 전문의",
        timestamp_start="02:30",
        timestamp_end="04:15",
        topics=["발열", "해열제"]
    )

    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"


def test_chunk_metadata_source_type() -> None:
    """source_type이 'youtube'로 설정되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="abc123",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의"
    )

    assert metadata['source_type'] == 'youtube'


def test_chunk_metadata_video_url_format() -> None:
    """video_url이 올바른 형식인지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="dQw4w9WgXcQ",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=False,
        specialty="",
        credentials=""
    )

    assert metadata['video_url'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'


def test_chunk_metadata_processed_at_is_set() -> None:
    """processed_at이 ISO 형식으로 설정되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="test",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의"
    )

    # ISO 8601 형식인지 확인 (예: 2024-12-31T10:30:00+00:00)
    assert 'processed_at' in metadata
    assert 'T' in metadata['processed_at']  # ISO 형식에는 T가 포함됨


def test_chunk_metadata_preserves_input_values() -> None:
    """입력값이 올바르게 보존되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="원본 텍스트",
        context="원본 컨텍스트",
        video_id="vid123",
        video_title="비디오 제목",
        channel_id="UC_channel_123",
        channel_name="채널 이름",
        chunk_index=5,
        published_at="2024-06-15T12:00:00Z",
        is_verified_professional=True,
        specialty="소아과",
        credentials="소아청소년과 전문의",
        timestamp_start="05:30",
        timestamp_end="07:45",
        topics=["발열", "응급실", "수분섭취"]
    )

    assert metadata['text'] == "원본 텍스트"
    assert metadata['context'] == "원본 컨텍스트"
    assert metadata['video_id'] == "vid123"
    assert metadata['video_title'] == "비디오 제목"
    assert metadata['channel_id'] == "UC_channel_123"
    assert metadata['channel_name'] == "채널 이름"
    assert metadata['chunk_index'] == 5
    assert metadata['published_at'] == "2024-06-15T12:00:00Z"
    assert metadata['is_verified_professional'] is True
    assert metadata['specialty'] == "소아과"
    assert metadata['credentials'] == "소아청소년과 전문의"
    assert metadata['timestamp_start'] == "05:30"
    assert metadata['timestamp_end'] == "07:45"
    assert metadata['topics'] == ["발열", "응급실", "수분섭취"]


def test_chunk_metadata_channel_id() -> None:
    """channel_id가 올바르게 저장되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="vid123",
        video_title="제목",
        channel_id="UCabcdef123456",
        channel_name="테스트 채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의"
    )

    assert metadata['channel_id'] == "UCabcdef123456"


def test_chunk_metadata_timestamps() -> None:
    """타임스탬프가 올바르게 저장되는지 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="vid123",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의",
        timestamp_start="10:30",
        timestamp_end="15:45"
    )

    assert metadata['timestamp_start'] == "10:30"
    assert metadata['timestamp_end'] == "15:45"


def test_chunk_metadata_topics() -> None:
    """topics 배열이 올바르게 저장되는지 검증"""
    from main import create_chunk_metadata

    topics_list = ["발열", "해열제", "응급실", "수분 섭취"]
    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="vid123",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의",
        topics=topics_list
    )

    assert metadata['topics'] == topics_list
    assert len(metadata['topics']) == 4


def test_chunk_metadata_default_values() -> None:
    """선택적 필드의 기본값 검증"""
    from main import create_chunk_metadata

    metadata = create_chunk_metadata(
        chunk_text="테스트",
        context="컨텍스트",
        video_id="vid123",
        video_title="제목",
        channel_id="UC123",
        channel_name="채널",
        chunk_index=0,
        published_at="2024-01-01",
        is_verified_professional=True,
        specialty="소아과",
        credentials="전문의"
        # timestamp_start, timestamp_end, topics 생략
    )

    # 기본값 검증
    assert metadata['timestamp_start'] == ''
    assert metadata['timestamp_end'] == ''
    assert metadata['topics'] == []


def test_normalize_specialty_with_list() -> None:
    """배열 형태의 specialty가 첫 번째 값으로 정규화되는지 검증"""
    from main import normalize_specialty

    # 배열인 경우 첫 번째 값 반환
    assert normalize_specialty(["pediatrics", "nutrition"]) == "pediatrics"
    assert normalize_specialty(["소아과"]) == "소아과"


def test_normalize_specialty_with_string() -> None:
    """문자열 형태의 specialty가 그대로 유지되는지 검증"""
    from main import normalize_specialty

    assert normalize_specialty("pediatrics") == "pediatrics"
    assert normalize_specialty("소아과") == "소아과"


def test_normalize_specialty_with_empty() -> None:
    """빈 값 처리 검증"""
    from main import normalize_specialty

    assert normalize_specialty([]) == ''
    assert normalize_specialty('') == ''
    assert normalize_specialty(None) == ''


def test_format_timestamp() -> None:
    """초를 MM:SS 형식으로 변환하는 함수 검증"""
    from main import format_timestamp

    assert format_timestamp(0) == "00:00"
    assert format_timestamp(30) == "00:30"
    assert format_timestamp(60) == "01:00"
    assert format_timestamp(90) == "01:30"
    assert format_timestamp(150) == "02:30"
    assert format_timestamp(3661) == "61:01"  # 1시간 1분 1초
    assert format_timestamp(150.5) == "02:30"  # 소수점 무시
