"""
GeminiProcessor 테스트 모듈.

GeminiProcessor의 자막 정제, 요약, 토픽 추출, 임베딩 기능을 검증합니다.
새 LLMManager 아키텍처를 사용하는 버전입니다.
"""
import pytest
from typing import Any
from unittest.mock import MagicMock

from src.processors.gemini_processor import GeminiProcessor
from src.processors.llm_provider import (
    LLMManager,
    RateLimiter,
    LLMResponse,
    ProviderType,
    ProviderError,
)


def test_gemini_init_missing_keys(mocker: Any) -> None:
    """Google과 OpenRouter 키가 모두 없으면 에러 발생 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', None)
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY 또는 OPENROUTER_API_KEY"):
        GeminiProcessor()


def test_gemini_init_with_google_only(mocker: Any) -> None:
    """Google 키만 있어도 초기화 가능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_google_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)
    mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

    processor = GeminiProcessor()
    assert processor.llm_manager is not None


def test_gemini_init_with_openrouter_only(mocker: Any) -> None:
    """OpenRouter 키만 있어도 초기화 가능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', None)
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', "fake_openrouter_key")

    processor = GeminiProcessor()
    assert processor.llm_manager is not None


def test_gemini_refine_transcript(mocker: Any) -> None:
    """자막 정제 기능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    # LLMManager 모킹
    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="Refined Text",
        provider=ProviderType.GOOGLE,
        model="gemini"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.refine_transcript("Raw Text")

    assert result == "Refined Text"
    mock_manager.generate.assert_called_once()


def test_gemini_summarize_video(mocker: Any) -> None:
    """영상 요약 기능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="Video summary",
        provider=ProviderType.GOOGLE,
        model="gemini"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.summarize_video("Full video text")

    assert result == "Video summary"


def test_gemini_extract_topics(mocker: Any) -> None:
    """토픽 추출 기능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="발열, 해열제, 응급실, 수분 섭취",
        provider=ProviderType.GOOGLE,
        model="gemini"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.extract_topics("아이가 열이 나면 해열제를 먹이세요.")

    assert isinstance(result, list)
    assert len(result) == 4
    assert "발열" in result
    assert "해열제" in result


def test_gemini_extract_topics_max_5(mocker: Any) -> None:
    """토픽 추출 시 최대 5개 제한 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="발열, 해열제, 응급실, 수분, 병원, 진료, 약국",
        provider=ProviderType.GOOGLE,
        model="gemini"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.extract_topics("테스트 텍스트")

    assert len(result) == 5


def test_gemini_context_and_topics_combined(mocker: Any) -> None:
    """컨텍스트와 토픽 통합 추출 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="""CONTEXT: 이 청크는 아이의 발열 증상에 대한 대처법을 설명합니다.
TOPICS: [발열, 해열제, 수분 섭취, 체온 측정]""",
        provider=ProviderType.GOOGLE,
        model="gemini"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    context, topics = processor.generate_chunk_context_and_topics(
        "아이가 열이 나면 해열제를 먹이세요.",
        "소아 발열 대처법 영상 요약"
    )

    assert "발열 증상" in context or "대처법" in context
    assert isinstance(topics, list)
    assert len(topics) >= 3
    assert "발열" in topics
    # API 한 번만 호출
    assert mock_manager.generate.call_count == 1


def test_gemini_context_and_topics_empty_response(mocker: Any) -> None:
    """통합 함수에서 빈 응답 처리 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)
    mocker.patch('src.processors.gemini_processor.settings.MAX_RETRIES', 1)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.side_effect = ProviderError("API Error")
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    context, topics = processor.generate_chunk_context_and_topics("테스트", "요약")

    assert context == ""
    # 빈 응답도 'NONE'으로 정규화됨 (의료 키워드 필터링 표준화)
    assert topics == ['NONE']


def test_gemini_get_embedding(mocker: Any) -> None:
    """임베딩 생성 기능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    # 정규화된 벡터 반환 [0.6, 0.8] (norm = 1.0)
    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.get_embedding.return_value = [0.6, 0.8]
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.get_embedding("test text")

    assert len(result) == 2
    assert abs(result[0] - 0.6) < 0.0001
    assert abs(result[1] - 0.8) < 0.0001


def test_gemini_fallback_to_openrouter(mocker: Any) -> None:
    """Google 할당량 초과 시 OpenRouter로 폴백 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_google")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', "fake_openrouter")

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.generate.return_value = LLMResponse(
        text="OpenRouter response",
        provider=ProviderType.OPENROUTER,
        model="gpt"
    )
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    result = processor.refine_transcript("테스트")

    assert result == "OpenRouter response"


def test_gemini_get_stats(mocker: Any) -> None:
    """통계 조회 기능 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.get_stats.return_value = {
        "google_calls": 5,
        "google_success": 4,
        "openrouter_calls": 1,
        "openrouter_success": 1,
        "fallback_count": 1
    }
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()
    stats = processor.get_stats()

    assert stats["google_calls"] == 5
    assert stats["fallback_count"] == 1


def test_gemini_fallback_mode_toggle(mocker: Any) -> None:
    """폴백 모드 수동 토글 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', "fake_openrouter")

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.google_provider = MagicMock()
    mock_manager.google_provider.rate_limiter = RateLimiter(rpm=5)

    mocker.patch('src.processors.gemini_processor.LLMManager', return_value=mock_manager)

    processor = GeminiProcessor()

    processor.enable_fallback_mode()
    mock_manager.enable_fallback_mode.assert_called_once()

    processor.disable_fallback_mode()
    mock_manager.disable_fallback_mode.assert_called_once()


def test_gemini_rate_limiter_backward_compatibility(mocker: Any) -> None:
    """하위 호환성을 위한 rate_limiter 속성 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)
    mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

    processor = GeminiProcessor(rpm=10)

    # rate_limiter 속성이 존재하고 올바른 설정인지 확인
    assert processor.rate_limiter is not None
    assert processor.rate_limiter.rpm == 10
    assert processor.rate_limiter.min_interval == 6.0


def test_gemini_processor_default_rpm(mocker: Any) -> None:
    """기본 RPM이 5인지 검증."""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mocker.patch('src.processors.gemini_processor.settings.OPENROUTER_API_KEY', None)
    mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

    processor = GeminiProcessor()

    assert processor.rate_limiter.rpm == 5
    assert processor.rate_limiter.min_interval == 12.0
