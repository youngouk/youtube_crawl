"""
LLM Provider 테스트 모듈.

GoogleProvider, OpenRouterProvider, LLMManager의 동작을 검증합니다.
"""
import pytest
from typing import Any
from unittest.mock import MagicMock

from src.processors.llm_provider import (
    GoogleProvider,
    OpenRouterProvider,
    LLMManager,
    RateLimiter,
    ProviderType,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    LLMResponse,
)


class TestRateLimiter:
    """RateLimiter 클래스 테스트."""

    def test_first_call_no_wait(self, mocker: Any) -> None:
        """첫 호출 시 대기하지 않는지 검증."""
        mock_sleep = mocker.patch('src.processors.llm_provider.time.sleep')
        mock_time = mocker.patch('src.processors.llm_provider.time.time')
        mock_time.return_value = 0

        limiter = RateLimiter(rpm=5)
        limiter.wait_if_needed()

        assert mock_sleep.call_count == 0

    def test_wait_when_interval_not_met(self, mocker: Any) -> None:
        """간격이 충족되지 않으면 대기하는지 검증."""
        mock_sleep = mocker.patch('src.processors.llm_provider.time.sleep')
        mock_time = mocker.patch('src.processors.llm_provider.time.time')

        # 첫 호출: 0초, 두 번째 호출: 5초 (12초 간격 필요)
        mock_time.side_effect = [0, 5, 12]

        limiter = RateLimiter(rpm=5)  # 12초 간격
        limiter.wait_if_needed()  # 첫 호출
        limiter.wait_if_needed()  # 두 번째 호출

        assert mock_sleep.call_count == 1
        # 12 - 5 = 7초 대기
        mock_sleep.assert_called_with(pytest.approx(7.0, abs=0.1))

    def test_no_wait_after_interval(self, mocker: Any) -> None:
        """충분한 시간이 지나면 대기하지 않는지 검증."""
        mock_sleep = mocker.patch('src.processors.llm_provider.time.sleep')
        mock_time = mocker.patch('src.processors.llm_provider.time.time')

        mock_time.side_effect = [0, 15, 15]  # 15초 후 호출

        limiter = RateLimiter(rpm=5)
        limiter.wait_if_needed()
        limiter.wait_if_needed()

        assert mock_sleep.call_count == 0

    def test_wait_for_retry(self, mocker: Any) -> None:
        """wait_for_retry 메서드 동작 검증."""
        mock_sleep = mocker.patch('src.processors.llm_provider.time.sleep')
        mock_time = mocker.patch('src.processors.llm_provider.time.time')
        mock_time.return_value = 100

        limiter = RateLimiter(rpm=5)
        limiter.wait_for_retry(45.0)

        mock_sleep.assert_called_once_with(45.0)


class TestGoogleProvider:
    """GoogleProvider 클래스 테스트."""

    def test_not_available_without_key(self, mocker: Any) -> None:
        """API 키 없으면 사용 불가 상태인지 검증."""
        # settings에서 API 키 로드 방지
        mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')
        provider = GoogleProvider(api_key=None)
        # api_key가 None이면 _initialize가 호출되지 않아 _model이 None
        provider._model = None
        assert provider.is_available is False

    def test_provider_type(self, mocker: Any) -> None:
        """프로바이더 타입이 GOOGLE인지 검증."""
        mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')
        provider = GoogleProvider(api_key="fake_key")
        assert provider.provider_type == ProviderType.GOOGLE

    def test_generate_success(self, mocker: Any) -> None:
        """성공적인 텍스트 생성 검증."""
        mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="fake_key")
        provider._model = mock_model
        provider._genai = MagicMock()

        response = provider.generate("Test prompt")

        assert response.text == "Generated text"
        assert response.provider == ProviderType.GOOGLE

    def test_quota_exceeded_error(self, mocker: Any) -> None:
        """할당량 초과 시 QuotaExceededError 발생 검증."""
        mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

        mock_model = MagicMock()
        # 429 에러 with quota 키워드
        mock_model.generate_content.side_effect = Exception(
            "429 quota exceeded, please retry in 45.5s FreeTier"
        )

        provider = GoogleProvider(api_key="fake_key")
        provider._model = mock_model
        provider._genai = MagicMock()

        with pytest.raises(QuotaExceededError) as exc_info:
            provider.generate("Test prompt")

        assert exc_info.value.retry_after == pytest.approx(47.5, abs=0.1)

    def test_rate_limit_error(self, mocker: Any) -> None:
        """Rate limit 초과 시 RateLimitError 발생 검증."""
        mocker.patch('src.processors.llm_provider.GoogleProvider._initialize')

        mock_model = MagicMock()
        # 429 에러 without quota 키워드
        mock_model.generate_content.side_effect = Exception(
            "429 Too Many Requests, please retry in 30s"
        )

        provider = GoogleProvider(api_key="fake_key")
        provider._model = mock_model
        provider._genai = MagicMock()

        with pytest.raises(RateLimitError) as exc_info:
            provider.generate("Test prompt")

        assert exc_info.value.retry_after == pytest.approx(32.0, abs=0.1)


class TestOpenRouterProvider:
    """OpenRouterProvider 클래스 테스트."""

    def test_not_available_without_key(self) -> None:
        """API 키 없으면 사용 불가 상태인지 검증."""
        # 명시적으로 빈 키 전달하여 테스트
        provider = OpenRouterProvider(api_key="")
        provider._api_key = None  # 강제로 None 설정
        assert provider.is_available is False

    def test_provider_type(self) -> None:
        """프로바이더 타입이 OPENROUTER인지 검증."""
        provider = OpenRouterProvider(api_key="fake_key")
        assert provider.provider_type == ProviderType.OPENROUTER

    def test_generate_success(self, mocker: Any) -> None:
        """성공적인 텍스트 생성 검증."""
        mock_post = mocker.patch('src.processors.llm_provider.requests.post')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'OpenRouter response'}}],
            'usage': {'total_tokens': 100}
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="fake_key")
        response = provider.generate("Test prompt")

        assert response.text == "OpenRouter response"
        assert response.provider == ProviderType.OPENROUTER
        assert response.tokens_used == 100


class TestLLMManager:
    """LLMManager 클래스 테스트."""

    def test_fallback_on_quota_exceeded(self, mocker: Any) -> None:
        """Google 할당량 초과 시 OpenRouter로 폴백하는지 검증."""
        # Google 프로바이더 모킹
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = True
        mock_google.generate.side_effect = QuotaExceededError("Quota exceeded")

        # OpenRouter 프로바이더 모킹
        mock_openrouter = MagicMock(spec=OpenRouterProvider)
        mock_openrouter.is_available = True
        mock_openrouter.generate.return_value = LLMResponse(
            text="Fallback response",
            provider=ProviderType.OPENROUTER,
            model="test-model"
        )

        manager = LLMManager(google_api_key="fake", openrouter_api_key="fake")
        manager.google_provider = mock_google
        manager.openrouter_provider = mock_openrouter

        response = manager.generate("Test prompt")

        assert response.text == "Fallback response"
        assert response.provider == ProviderType.OPENROUTER
        assert manager.stats["fallback_count"] == 1
        assert manager._fallback_mode is True

    def test_retry_on_rate_limit(self, mocker: Any) -> None:
        """Rate limit 시 재시도하는지 검증."""
        mock_sleep = mocker.patch('src.processors.llm_provider.time.sleep')

        # Google 프로바이더 모킹 - 첫 호출 rate limit, 두 번째 성공
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = True
        mock_google.rate_limiter = RateLimiter(rpm=5)
        mock_google.generate.side_effect = [
            RateLimitError("Rate limited", retry_after=5.0),
            LLMResponse(text="Success", provider=ProviderType.GOOGLE, model="test")
        ]

        manager = LLMManager(google_api_key="fake", max_retries=3)
        manager.google_provider = mock_google

        response = manager.generate("Test prompt")

        assert response.text == "Success"
        assert mock_google.generate.call_count == 2

    def test_google_only_mode(self, mocker: Any) -> None:
        """OpenRouter 없이 Google만 사용하는 경우 검증."""
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = True
        mock_google.generate.return_value = LLMResponse(
            text="Google response",
            provider=ProviderType.GOOGLE,
            model="gemini"
        )

        manager = LLMManager(google_api_key="fake", openrouter_api_key=None)
        manager.google_provider = mock_google
        manager.openrouter_provider = OpenRouterProvider(api_key=None)

        response = manager.generate("Test prompt")

        assert response.text == "Google response"
        assert response.provider == ProviderType.GOOGLE

    def test_openrouter_only_mode(self, mocker: Any) -> None:
        """Google 없이 OpenRouter만 사용하는 경우 검증."""
        # Google 프로바이더 모킹 (사용 불가 상태)
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = False

        mock_openrouter = MagicMock(spec=OpenRouterProvider)
        mock_openrouter.is_available = True
        mock_openrouter.generate.return_value = LLMResponse(
            text="OpenRouter response",
            provider=ProviderType.OPENROUTER,
            model="gpt"
        )

        manager = LLMManager(google_api_key=None, openrouter_api_key="fake")
        manager.google_provider = mock_google
        manager.openrouter_provider = mock_openrouter

        response = manager.generate("Test prompt")

        assert response.text == "OpenRouter response"
        assert response.provider == ProviderType.OPENROUTER

    def test_fallback_mode_toggle(self) -> None:
        """폴백 모드 토글 검증."""
        manager = LLMManager(google_api_key="fake", openrouter_api_key="fake")

        assert manager._fallback_mode is False
        assert manager.is_google_available is True

        manager.enable_fallback_mode()
        assert manager._fallback_mode is True
        assert manager.is_google_available is False

        manager.disable_fallback_mode()
        assert manager._fallback_mode is False
        assert manager.is_google_available is True

    def test_stats_tracking(self, mocker: Any) -> None:
        """호출 통계 추적 검증."""
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = True
        mock_google.generate.return_value = LLMResponse(
            text="Response",
            provider=ProviderType.GOOGLE,
            model="gemini"
        )

        manager = LLMManager(google_api_key="fake")
        manager.google_provider = mock_google

        manager.generate("Test 1")
        manager.generate("Test 2")

        stats = manager.get_stats()
        assert stats["google_calls"] == 2
        assert stats["google_success"] == 2
        assert stats["fallback_count"] == 0

    def test_all_providers_fail(self, mocker: Any) -> None:
        """모든 프로바이더 실패 시 에러 발생 검증."""
        mock_google = MagicMock(spec=GoogleProvider)
        mock_google.is_available = True
        mock_google.generate.side_effect = QuotaExceededError("Quota exceeded")

        mock_openrouter = MagicMock(spec=OpenRouterProvider)
        mock_openrouter.is_available = True
        mock_openrouter.generate.side_effect = ProviderError("API Error")

        mocker.patch('src.processors.llm_provider.time.sleep')

        manager = LLMManager(google_api_key="fake", openrouter_api_key="fake", max_retries=1)
        manager.google_provider = mock_google
        manager.openrouter_provider = mock_openrouter

        with pytest.raises(ProviderError, match="모든 LLM 프로바이더 호출 실패"):
            manager.generate("Test prompt")


class TestLLMResponse:
    """LLMResponse 데이터 클래스 테스트."""

    def test_response_creation(self) -> None:
        """응답 객체 생성 검증."""
        response = LLMResponse(
            text="Test text",
            provider=ProviderType.GOOGLE,
            model="gemini-2.5-flash",
            tokens_used=100
        )

        assert response.text == "Test text"
        assert response.provider == ProviderType.GOOGLE
        assert response.model == "gemini-2.5-flash"
        assert response.tokens_used == 100

    def test_response_without_tokens(self) -> None:
        """토큰 정보 없는 응답 검증."""
        response = LLMResponse(
            text="Test",
            provider=ProviderType.OPENROUTER,
            model="test"
        )

        assert response.tokens_used is None
