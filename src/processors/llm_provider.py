"""
LLM 프로바이더 추상 인터페이스 및 구현체.

Google Gemini와 OpenRouter를 지원하며, 폴백 로직을 통해
Google API 실패 시 OpenRouter로 자동 전환합니다.

주요 클래스:
- LLMProvider: 추상 기반 클래스
- GoogleProvider: Google Gemini API 직접 호출
- OpenRouterProvider: OpenRouter API 호출 (유료)
- LLMManager: 폴백 로직이 포함된 프로바이더 관리자
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import time
import re
import requests


class ProviderType(Enum):
    """LLM 프로바이더 유형."""
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ProviderError(Exception):
    """프로바이더 관련 에러의 기본 클래스."""
    pass


class QuotaExceededError(ProviderError):
    """API 할당량 초과 에러."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """Rate limit 초과 에러."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스."""
    text: str
    provider: ProviderType
    model: str
    tokens_used: Optional[int] = None


class RateLimiter:
    """
    API Rate Limit 관리자.

    요청 간 최소 간격을 유지하여 rate limit을 준수합니다.
    """

    def __init__(self, rpm: int = 5) -> None:
        """
        Args:
            rpm: 분당 최대 요청 수 (기본값: 5, gemini-2.5-flash 무료 티어)
        """
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self._first_call = True
        self.last_request_time: float = 0.0

    def wait_if_needed(self) -> None:
        """Rate limit을 준수하기 위해 필요한 경우 대기합니다."""
        current_time = time.time()

        if self._first_call:
            self._first_call = False
            self.last_request_time = current_time
            return

        elapsed = current_time - self.last_request_time

        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"[RateLimiter] Waiting {wait_time:.1f}s for rate limit...")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def wait_for_retry(self, seconds: float) -> None:
        """에러 발생 시 지정된 시간만큼 대기합니다."""
        print(f"[RateLimiter] 에러 발생 - {seconds:.0f}초 대기 중...")
        time.sleep(seconds)
        self.last_request_time = time.time()


class LLMProvider(ABC):
    """LLM 프로바이더 추상 기반 클래스."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """프로바이더 유형을 반환합니다."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """프로바이더 사용 가능 여부를 반환합니다."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """
        텍스트를 생성합니다.

        Args:
            prompt: 입력 프롬프트

        Returns:
            LLMResponse 객체

        Raises:
            QuotaExceededError: 할당량 초과 시
            RateLimitError: Rate limit 초과 시
            ProviderError: 기타 에러 시
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """
        텍스트의 임베딩 벡터를 생성합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (1024차원)
        """
        pass


class GoogleProvider(LLMProvider):
    """
    Google Gemini API 프로바이더.

    gemini-2.5-flash (텍스트 생성)와 gemini-embedding-001 (임베딩)을 사용합니다.
    """

    def __init__(self, api_key: Optional[str] = None, rpm: int = 5) -> None:
        """
        Args:
            api_key: Google API 키 (None이면 환경변수에서 로드)
            rpm: 분당 최대 요청 수
        """
        # 환경변수에서 API 키 로드 (지연 임포트)
        if api_key is None:
            from config.settings import settings
            api_key = settings.GOOGLE_API_KEY

        self._api_key = api_key
        self._model: Any = None
        self._genai: Any = None
        self.rate_limiter = RateLimiter(rpm=rpm)

        if self._api_key:
            self._initialize()

    def _initialize(self) -> None:
        """Gemini API를 초기화합니다."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._genai = genai
            self._model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            print(f"[GoogleProvider] 초기화 실패: {e}")

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    @property
    def is_available(self) -> bool:
        return self._api_key is not None and self._model is not None

    def generate(self, prompt: str) -> LLMResponse:
        """Gemini를 사용하여 텍스트를 생성합니다."""
        if not self.is_available:
            raise ProviderError("GoogleProvider가 초기화되지 않았습니다.")

        self.rate_limiter.wait_if_needed()

        try:
            response = self._model.generate_content(prompt)
            return LLMResponse(
                text=response.text.strip(),
                provider=self.provider_type,
                model="gemini-2.5-flash"
            )
        except Exception as e:
            self._handle_error(e)
            raise  # _handle_error에서 처리되지 않은 경우

    def _handle_error(self, error: Exception) -> None:
        """에러를 분석하고 적절한 예외를 발생시킵니다."""
        error_str = str(error)

        # 429 에러 (Rate limit 또는 Quota 초과)
        if "429" in error_str:
            # "Please retry in X seconds" 파싱
            match = re.search(r'retry in (\d+\.?\d*)', error_str.lower())
            retry_after = float(match.group(1)) + 2 if match else 60.0

            # 일일 할당량 초과 vs Rate limit 구분
            if "quota" in error_str.lower() or "FreeTier" in error_str:
                raise QuotaExceededError(
                    f"Google API 할당량 초과: {error_str}",
                    retry_after=retry_after
                )
            else:
                raise RateLimitError(
                    f"Google API rate limit 초과: {error_str}",
                    retry_after=retry_after
                )

        raise ProviderError(f"Google API 에러: {error_str}")

    def get_embedding(self, text: str) -> list[float]:
        """Gemini 임베딩 API를 사용하여 벡터를 생성합니다."""
        if not self.is_available:
            raise ProviderError("GoogleProvider가 초기화되지 않았습니다.")

        try:
            result = self._genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document",
                output_dimensionality=1024
            )
            embedding: list[float] = result['embedding']

            # 1024차원은 정규화 필요
            import math
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            return embedding
        except Exception as e:
            print(f"[GoogleProvider] 임베딩 생성 실패: {e}")
            return []


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter API 프로바이더.

    다양한 LLM 모델에 접근할 수 있으며, 유료 구독 시 할당량 제한이 없습니다.
    """

    # OpenRouter API 엔드포인트
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

    # 기본 모델 설정 (Gemini 대비 가격 효율적인 모델)
    DEFAULT_MODEL = "google/gemini-2.5-flash"  # OpenRouter의 Gemini 2.5 Flash
    EMBED_MODEL = "google/gemini-embedding-001"  # Gemini 임베딩 모델 (1024차원)

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Args:
            api_key: OpenRouter API 키 (None이면 환경변수에서 로드)
        """
        if api_key is None:
            from config.settings import settings
            api_key = getattr(settings, 'OPENROUTER_API_KEY', None)

        self._api_key = api_key

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENROUTER

    @property
    def is_available(self) -> bool:
        return self._api_key is not None

    def generate(self, prompt: str) -> LLMResponse:
        """OpenRouter를 사용하여 텍스트를 생성합니다."""
        if not self.is_available:
            raise ProviderError("OpenRouterProvider가 초기화되지 않았습니다.")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://medical-rag-pipeline.local",
            "X-Title": "Medical RAG Pipeline"
        }

        data = {
            "model": self.DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            text = result['choices'][0]['message']['content'].strip()
            tokens = result.get('usage', {}).get('total_tokens')

            return LLMResponse(
                text=text,
                provider=self.provider_type,
                model=self.DEFAULT_MODEL,
                tokens_used=tokens
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"OpenRouter rate limit 초과: {e}")
            raise ProviderError(f"OpenRouter API 에러: {e}")
        except Exception as e:
            raise ProviderError(f"OpenRouter API 에러: {e}")

    def get_embedding(self, text: str) -> list[float]:
        """OpenRouter 임베딩 API를 사용하여 벡터를 생성합니다."""
        if not self.is_available:
            raise ProviderError("OpenRouterProvider가 초기화되지 않았습니다.")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.EMBED_MODEL,
            "input": text,
            "dimensions": 1024  # MRL로 1024차원 출력
        }

        try:
            response = requests.post(
                self.EMBED_URL,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            embedding: list[float] = result['data'][0]['embedding']

            # 정규화 (1024차원은 MRL로 생성되므로 정규화 필요)
            import math
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            return embedding
        except Exception as e:
            print(f"[OpenRouterProvider] 임베딩 생성 실패: {e}")
            return []


class LLMManager:
    """
    LLM 프로바이더 관리자.

    Google API를 1차로 시도하고, 실패 시 OpenRouter로 폴백합니다.
    할당량 초과 시에만 폴백하며, 일반 에러는 재시도합니다.
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        google_rpm: int = 5,
        max_retries: int = 3
    ) -> None:
        """
        Args:
            google_api_key: Google API 키
            openrouter_api_key: OpenRouter API 키
            google_rpm: Google API 분당 최대 요청 수
            max_retries: 최대 재시도 횟수
        """
        self.google_provider = GoogleProvider(api_key=google_api_key, rpm=google_rpm)
        self.openrouter_provider = OpenRouterProvider(api_key=openrouter_api_key)
        self.max_retries = max_retries

        # 통계 추적
        self.stats = {
            "google_calls": 0,
            "google_success": 0,
            "openrouter_calls": 0,
            "openrouter_success": 0,
            "fallback_count": 0
        }

        # 폴백 모드 (True이면 Google 건너뛰고 바로 OpenRouter 사용)
        self._fallback_mode = False

    @property
    def is_google_available(self) -> bool:
        """Google 프로바이더 사용 가능 여부."""
        return self.google_provider.is_available and not self._fallback_mode

    @property
    def is_openrouter_available(self) -> bool:
        """OpenRouter 프로바이더 사용 가능 여부."""
        return self.openrouter_provider.is_available

    def enable_fallback_mode(self) -> None:
        """폴백 모드를 활성화합니다 (할당량 초과 시 호출)."""
        if not self._fallback_mode:
            print("[LLMManager] 폴백 모드 활성화 - OpenRouter로 전환합니다.")
            self._fallback_mode = True

    def disable_fallback_mode(self) -> None:
        """폴백 모드를 비활성화합니다 (다음 날 할당량 리셋 시)."""
        if self._fallback_mode:
            print("[LLMManager] 폴백 모드 비활성화 - Google API로 복귀합니다.")
            self._fallback_mode = False

    def generate(self, prompt: str) -> LLMResponse:
        """
        텍스트를 생성합니다.

        Google API를 먼저 시도하고, 할당량 초과 시 OpenRouter로 폴백합니다.

        Args:
            prompt: 입력 프롬프트

        Returns:
            LLMResponse 객체

        Raises:
            ProviderError: 모든 프로바이더 실패 시
        """
        # 1차: Google API 시도 (폴백 모드가 아닌 경우)
        if self.is_google_available:
            for attempt in range(self.max_retries):
                try:
                    self.stats["google_calls"] += 1
                    response = self.google_provider.generate(prompt)
                    self.stats["google_success"] += 1
                    return response
                except QuotaExceededError as e:
                    print(f"[LLMManager] Google 할당량 초과: {e}")
                    self.enable_fallback_mode()
                    self.stats["fallback_count"] += 1
                    break  # OpenRouter로 폴백
                except RateLimitError as e:
                    print(f"[LLMManager] Google rate limit (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if e.retry_after:
                        self.google_provider.rate_limiter.wait_for_retry(e.retry_after)
                except ProviderError as e:
                    print(f"[LLMManager] Google 에러 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(2 ** attempt)

        # 2차: OpenRouter 폴백
        if self.is_openrouter_available:
            for attempt in range(self.max_retries):
                try:
                    self.stats["openrouter_calls"] += 1
                    response = self.openrouter_provider.generate(prompt)
                    self.stats["openrouter_success"] += 1
                    return response
                except RateLimitError as e:
                    print(f"[LLMManager] OpenRouter rate limit (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(2 ** attempt)
                except ProviderError as e:
                    print(f"[LLMManager] OpenRouter 에러 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(2 ** attempt)

        raise ProviderError("모든 LLM 프로바이더 호출 실패")

    def get_embedding(self, text: str) -> list[float]:
        """
        텍스트의 임베딩 벡터를 생성합니다.

        폴백 모드가 아니면 Google API를 먼저 시도하고, 실패 시 OpenRouter로 폴백합니다.
        폴백 모드가 활성화되면 Google API를 건너뛰고 바로 OpenRouter를 사용합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (1024차원)
        """
        # 1차: Google API (폴백 모드가 아닐 때만)
        if not self._fallback_mode and self.google_provider.is_available:
            try:
                embedding = self.google_provider.get_embedding(text)
                if embedding:
                    return embedding
            except Exception as e:
                print(f"[LLMManager] Google 임베딩 실패: {e}")
                # 임베딩도 할당량 초과 시 폴백 모드 활성화
                if "429" in str(e) or "quota" in str(e).lower():
                    self.enable_fallback_mode()

        # 2차: OpenRouter 폴백
        if self.is_openrouter_available:
            try:
                embedding = self.openrouter_provider.get_embedding(text)
                if embedding:
                    return embedding
            except Exception as e:
                print(f"[LLMManager] OpenRouter 임베딩 실패: {e}")

        return []

    def get_stats(self) -> dict[str, Any]:
        """프로바이더 호출 통계를 반환합니다."""
        return {
            **self.stats,
            "fallback_mode": self._fallback_mode,
            "google_available": self.is_google_available,
            "openrouter_available": self.is_openrouter_available
        }

    def reset_stats(self) -> None:
        """통계를 초기화합니다."""
        self.stats = {
            "google_calls": 0,
            "google_success": 0,
            "openrouter_calls": 0,
            "openrouter_success": 0,
            "fallback_count": 0
        }
