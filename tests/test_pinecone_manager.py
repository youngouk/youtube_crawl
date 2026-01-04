"""
PineconeManager 테스트

Pinecone 벡터 데이터베이스 관리자의 단위 테스트입니다.
"""

from typing import Any
from unittest.mock import MagicMock
from src.vector_db.pinecone_manager import PineconeManager


def _create_mock_manager(mocker: Any, index_exists: bool = True) -> tuple[Any, Any, PineconeManager]:
    """
    테스트용 PineconeManager 목업을 생성합니다.

    Args:
        mocker: pytest-mock mocker
        index_exists: 인덱스 존재 여부

    Returns:
        (mock_pc_instance, mock_index, manager) 튜플
    """
    mock_pinecone_cls = mocker.patch('src.vector_db.pinecone_manager.Pinecone')
    mock_pc_instance = mock_pinecone_cls.return_value

    if index_exists:
        mock_index_obj = MagicMock()
        mock_index_obj.name = "medical-rag-hybrid"
        mock_pc_instance.list_indexes.return_value = [mock_index_obj]
    else:
        mock_pc_instance.list_indexes.return_value = []
        mock_pc_instance.create_index = MagicMock()
        mock_pc_instance.describe_index.return_value = MagicMock(status={'ready': True})

    mock_index = mock_pc_instance.Index.return_value
    manager = PineconeManager()

    return mock_pc_instance, mock_index, manager


def test_pinecone_init_create_index(mocker: Any) -> None:
    """인덱스가 없을 때 생성되는지 테스트합니다."""
    mock_pc_instance, _, manager = _create_mock_manager(mocker, index_exists=False)

    mock_pc_instance.create_index.assert_called_once()
    assert manager.index is not None


def test_pinecone_init_existing_index(mocker: Any) -> None:
    """기존 인덱스가 있을 때 생성하지 않는지 테스트합니다."""
    mock_pc_instance, _, manager = _create_mock_manager(mocker, index_exists=True)

    mock_pc_instance.create_index.assert_not_called()
    assert manager.index is not None


def test_pinecone_upsert_vectors(mocker: Any) -> None:
    """벡터 upsert가 올바르게 동작하는지 테스트합니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    vectors = [{"id": "1", "values": [0.1]}]
    result = manager.upsert_vectors(vectors)

    mock_index.upsert.assert_called_with(vectors=vectors)
    assert result == 1  # 반환값이 upsert된 벡터 수


def test_pinecone_upsert_empty_vectors(mocker: Any) -> None:
    """빈 벡터 리스트 upsert 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    result = manager.upsert_vectors([])

    mock_index.upsert.assert_not_called()
    assert result == 0


def test_pinecone_upsert_batch(mocker: Any) -> None:
    """100개 이상 벡터 배치 upsert 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    # 150개 벡터 생성
    vectors = [{"id": str(i), "values": [0.1]} for i in range(150)]
    result = manager.upsert_vectors(vectors)

    # 2번 호출되어야 함 (100 + 50)
    assert mock_index.upsert.call_count == 2
    assert result == 150


def test_pinecone_query(mocker: Any) -> None:
    """벡터 쿼리가 올바르게 동작하는지 테스트합니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    # Mock 쿼리 응답
    mock_match = MagicMock()
    mock_match.id = "video1_0"
    mock_match.score = 0.95
    mock_match.metadata = {"text": "테스트 텍스트", "video_id": "video1"}

    mock_response = MagicMock()
    mock_response.matches = [mock_match]
    mock_index.query.return_value = mock_response

    query_vector = [0.1] * 768
    results = manager.query(query_vector, top_k=5)

    mock_index.query.assert_called_once()
    assert len(results) == 1
    assert results[0]["id"] == "video1_0"
    assert results[0]["score"] == 0.95
    assert results[0]["metadata"]["video_id"] == "video1"


def test_pinecone_query_with_filter(mocker: Any) -> None:
    """필터가 있는 쿼리 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    mock_response = MagicMock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    query_vector = [0.1] * 768
    manager.query(
        query_vector,
        top_k=10,
        filter={"is_verified_professional": True}
    )

    # 필터가 전달되었는지 확인
    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["filter"] == {"is_verified_professional": True}


def test_pinecone_query_verified_only(mocker: Any) -> None:
    """인증된 전문가만 검색하는 메서드 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    mock_response = MagicMock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    query_vector = [0.1] * 768
    manager.query_verified_only(query_vector, top_k=5, specialty="소아과")

    call_kwargs = mock_index.query.call_args.kwargs
    expected_filter = {
        "is_verified_professional": {"$eq": True},
        "specialty": {"$eq": "소아과"}
    }
    assert call_kwargs["filter"] == expected_filter


def test_pinecone_query_by_video(mocker: Any) -> None:
    """특정 비디오 내 검색 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    mock_response = MagicMock()
    mock_response.matches = []
    mock_index.query.return_value = mock_response

    query_vector = [0.1] * 768
    manager.query_by_video(query_vector, video_id="abc123")

    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["filter"] == {"video_id": {"$eq": "abc123"}}


def test_pinecone_delete_video(mocker: Any) -> None:
    """비디오 벡터 삭제 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    manager.delete_video_vectors("vid1")

    mock_index.delete.assert_called_with(filter={"video_id": {"$eq": "vid1"}})


def test_pinecone_delete_channel(mocker: Any) -> None:
    """채널 벡터 삭제 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    manager.delete_channel_vectors("channel123")

    mock_index.delete.assert_called_with(filter={"channel_id": {"$eq": "channel123"}})


def test_pinecone_get_index_stats(mocker: Any) -> None:
    """인덱스 통계 조회 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {
        "total_vector_count": 1000,
        "dimension": 768,
        "namespaces": {}
    }
    mock_index.describe_index_stats.return_value = mock_stats

    stats = manager.get_index_stats()

    assert stats["total_vector_count"] == 1000
    assert stats["dimension"] == 768


def test_pinecone_get_vector_count(mocker: Any) -> None:
    """총 벡터 수 조회 테스트입니다."""
    _, mock_index, manager = _create_mock_manager(mocker)

    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {"total_vector_count": 500}
    mock_index.describe_index_stats.return_value = mock_stats

    count = manager.get_vector_count()

    assert count == 500


def test_pinecone_is_ready(mocker: Any) -> None:
    """인덱스 준비 상태 확인 테스트입니다."""
    mock_pc_instance, _, manager = _create_mock_manager(mocker)

    mock_pc_instance.describe_index.return_value = MagicMock(status={'ready': True})

    assert manager.is_ready() is True


def test_pinecone_constants(mocker: Any) -> None:
    """상수 값 테스트입니다."""
    _, _, manager = _create_mock_manager(mocker)

    # 1024차원 임베딩 사용 (gemini-embedding-001 MRL 지원)
    assert manager.DIMENSION == 1024
    assert manager.METRIC == "dotproduct"
    assert manager.BATCH_SIZE == 100
