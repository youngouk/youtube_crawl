"""
Gemini 기반 텍스트 처리기.

YouTube 자막의 정제, 요약, 청크 컨텍스트 생성, 토픽 추출, 임베딩을 담당합니다.
Google API를 1차로 사용하고, 할당량 초과 시 OpenRouter로 자동 폴백합니다.

주요 기능:
- refine_transcript(): 자막 정제 (오타 교정, 의학 용어 표준화)
- summarize_video(): 영상 요약 생성
- generate_chunk_context_and_topics(): 청크 컨텍스트 + 토픽 추출 (통합)
- get_embedding(): 1024차원 임베딩 벡터 생성
"""
from config.settings import settings
from typing import Any, Optional
from src.processors.llm_provider import (
    LLMManager,
    RateLimiter,
    ProviderError,
)


# 의료 키워드가 없을 때 LLM이 출력할 수 있는 패턴들
# 이 패턴들은 모두 "NONE"으로 정규화됨
EMPTY_TOPIC_PATTERNS = [
    "해당 없음",
    "없음",
    "해당 청크에는",
    "의료 관련 키워드가",
    "키워드가 없",
    "추출할 수 없",
    "관련 키워드 없",
]


def _normalize_topics(topics: list[str]) -> list[str]:
    """
    토픽 리스트를 정규화합니다.

    의료 키워드가 없는 경우의 다양한 LLM 출력을 "NONE"으로 통일합니다.

    Args:
        topics: 원본 토픽 리스트

    Returns:
        정규화된 토픽 리스트. 유효한 키워드가 없으면 ["NONE"] 반환
    """
    if not topics:
        return ["NONE"]

    normalized = []
    for topic in topics:
        topic = topic.strip()

        # 빈 문자열 스킵
        if not topic:
            continue

        # 빈 토픽 패턴 체크
        is_empty_pattern = any(pattern in topic for pattern in EMPTY_TOPIC_PATTERNS)
        if is_empty_pattern:
            continue

        normalized.append(topic)

    # 유효한 토픽이 없으면 NONE 반환
    return normalized if normalized else ["NONE"]


class GeminiProcessor:
    """
    Gemini API를 사용하여 자막 정제, 요약, 임베딩을 생성합니다.

    Google API를 1차로 시도하고, 할당량 초과 시 OpenRouter로 폴백합니다.
    """

    llm_manager: LLMManager
    rate_limiter: RateLimiter  # 하위 호환성을 위해 유지

    def __init__(
        self,
        rpm: int = 5,
        google_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None
    ) -> None:
        """
        GeminiProcessor를 초기화합니다.

        Args:
            rpm: 분당 최대 요청 수 (기본값: 5, gemini-2.5-flash 무료 티어)
            google_api_key: Google API 키 (None이면 환경변수에서 로드)
            openrouter_api_key: OpenRouter API 키 (None이면 환경변수에서 로드)
        """
        # API 키 로드
        if google_api_key is None:
            google_api_key = settings.GOOGLE_API_KEY
        if openrouter_api_key is None:
            openrouter_api_key = getattr(settings, 'OPENROUTER_API_KEY', None)

        # Google API가 없어도 OpenRouter가 있으면 동작 가능
        if not google_api_key and not openrouter_api_key:
            raise ValueError("GOOGLE_API_KEY 또는 OPENROUTER_API_KEY 중 하나는 필요합니다.")

        # LLM 매니저 초기화 (폴백 로직 포함)
        self.llm_manager = LLMManager(
            google_api_key=google_api_key,
            openrouter_api_key=openrouter_api_key,
            google_rpm=rpm,
            max_retries=settings.MAX_RETRIES
        )

        # 하위 호환성을 위해 rate_limiter 속성 유지
        self.rate_limiter = self.llm_manager.google_provider.rate_limiter

    def refine_transcript(self, raw_text: str) -> str:
        """
        Gemini를 사용하여 자막을 정제합니다.

        Args:
            raw_text: 원본 자막 텍스트

        Returns:
            정제된 자막 텍스트
        """
        prompt = f"""
        역할: 소아과 의료 콘텐츠 전문 교정자

        작업:
        1. 발음 오인식 교정 (여리→열이, 해여제→해열제)
        2. 띄어쓰기/맞춤법 수정
        3. 의학 용어 표준화 (로타 바이러스→로타바이러스)
        4. 불완전한 문장 연결

        금지:
        - 의미 변경 또는 추가 금지
        - 원본에 없는 의학 정보 삽입 금지
        - 답변에는 정제된 텍스트만 출력하세요. 사족이나 설명은 넣지 마세요.

        텍스트:
        {raw_text}
        """
        return self._generate_patience(prompt)

    def summarize_video(self, full_text: str) -> str:
        """
        영상 콘텐츠의 요약을 생성합니다.

        Args:
            full_text: 전체 자막 텍스트

        Returns:
            3-5문장 요약
        """
        prompt = f"""
        다음은 소아과 관련 유튜브 영상의 자막입니다. 이 영상의 핵심 내용을 3-5문장으로 요약해주세요.
        이 요약은 나중에 이 영상의 특정 부분이 어떤 맥락인지 파악하는 데 사용됩니다.

        자막:
        {full_text}
        """
        return self._generate_patience(prompt)

    def generate_chunk_context(self, chunk_text: str, video_summary: str) -> str:
        """
        영상 요약을 기반으로 특정 청크의 컨텍스트를 생성합니다.

        Args:
            chunk_text: 청크 텍스트
            video_summary: 전체 영상 요약

        Returns:
            청크 컨텍스트 (1-2문장)
        """
        prompt = f"""
        전체 문서(영상) 요약:
        {video_summary}

        아래 청크가 전체 문서에서 어떤 맥락인지 1-2문장으로 설명하세요.
        청크 내용을 단순히 반복하지 말고, 이 내용이 전체 주제 내에서 어떤 역할을 하는지 서술하세요.

        청크:
        {chunk_text}
        """
        return self._generate_patience(prompt)

    def extract_topics(self, chunk_text: str) -> list[str]:
        """
        청크 텍스트에서 의료 관련 토픽 키워드를 추출합니다.

        Args:
            chunk_text: 분석할 청크 텍스트

        Returns:
            토픽 키워드 리스트 (최대 5개). 의료 키워드가 없으면 ["NONE"] 반환
        """
        prompt = f"""
        다음 소아과 관련 텍스트에서 핵심 의료 토픽 키워드를 추출하세요.

        규칙:
        1. 증상, 질병, 약물, 치료법, 신체 부위 등 의료 관련 키워드만 추출
        2. 한국어로 3-5개 키워드만 추출
        3. 쉼표로 구분하여 키워드만 출력 (설명 없이)
        4. 의료 관련 키워드가 없으면 정확히 "NONE"만 출력하세요

        예시 출력: 발열, 해열제, 응급실, 수분 섭취
        의료 키워드 없을 때 출력: NONE

        텍스트:
        {chunk_text}
        """
        result = self._generate_patience(prompt)
        if not result:
            return ["NONE"]

        # 쉼표로 분리하고 정리
        topics = [topic.strip() for topic in result.split(',') if topic.strip()]
        topics = topics[:5]  # 최대 5개

        # 정규화 적용 (빈 패턴 → NONE)
        return _normalize_topics(topics)

    def generate_chunk_context_and_topics(
        self, chunk_text: str, video_summary: str
    ) -> tuple[str, list[str]]:
        """
        청크의 컨텍스트와 토픽을 한 번의 API 호출로 생성합니다.

        API 호출 횟수를 줄이기 위해 generate_chunk_context()와 extract_topics()를
        하나의 호출로 통합한 함수입니다.

        Args:
            chunk_text: 청크 텍스트
            video_summary: 전체 영상 요약

        Returns:
            (context, topics) 튜플
            - context: 청크가 전체 문서에서 갖는 맥락 (1-2문장)
            - topics: 의료 관련 토픽 키워드 리스트 (최대 5개). 없으면 ["NONE"]
        """
        prompt = f"""
        전체 문서(영상) 요약:
        {video_summary}

        아래 청크에 대해 두 가지 작업을 수행하세요:

        [작업 1] 컨텍스트 설명
        이 청크가 전체 문서에서 어떤 맥락인지 1-2문장으로 설명하세요.
        청크 내용을 단순히 반복하지 말고, 전체 주제 내에서 어떤 역할을 하는지 서술하세요.

        [작업 2] 토픽 추출
        증상, 질병, 약물, 치료법, 신체 부위 등 의료 관련 핵심 키워드를 3-5개 추출하세요.
        의료 관련 키워드가 없으면 정확히 "NONE"만 출력하세요.

        청크:
        {chunk_text}

        응답 형식 (정확히 이 형식을 따르세요):
        CONTEXT: [컨텍스트 설명]
        TOPICS: [키워드1, 키워드2, 키워드3] 또는 TOPICS: NONE
        """
        result = self._generate_patience(prompt)

        if not result:
            return "", ["NONE"]

        # 응답 파싱
        context = ""
        topics: list[str] = []

        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("CONTEXT:"):
                context = line[8:].strip()
            elif line.startswith("TOPICS:"):
                topics_str = line[7:].strip()
                # 대괄호 제거
                topics_str = topics_str.strip('[]')
                # 쉼표로 분리
                topics = [t.strip() for t in topics_str.split(',') if t.strip()]
                topics = topics[:5]  # 최대 5개

        # 정규화 적용 (빈 패턴 → NONE)
        topics = _normalize_topics(topics)

        return context, topics

    def _generate_patience(self, prompt: str, retries: int | None = None) -> str:
        """
        LLM API 호출을 재시도와 함께 수행합니다.

        Google API를 1차로 시도하고, 할당량 초과 시 OpenRouter로 폴백합니다.

        Args:
            prompt: API에 전달할 프롬프트
            retries: 재시도 횟수 (None이면 settings.MAX_RETRIES 사용)

        Returns:
            생성된 텍스트 또는 빈 문자열 (실패 시)
        """
        try:
            response = self.llm_manager.generate(prompt)
            provider = response.provider.value
            print(f"[GeminiProcessor] {provider} 사용하여 응답 생성 완료")
            return response.text
        except ProviderError as e:
            print(f"[GeminiProcessor] 모든 프로바이더 실패: {e}")
            return ""

    def get_embedding(self, text: str) -> list[float]:
        """
        주어진 텍스트의 임베딩 벡터를 생성합니다.

        MRL(Matryoshka Representation Learning) 기법으로 3072차원을 1024차원으로 축소하여
        Pinecone 인덱스와 호환되면서도 최신 모델의 성능을 활용합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            1024차원 정규화된 임베딩 벡터
        """
        return self.llm_manager.get_embedding(text)

    def get_stats(self) -> dict[str, Any]:
        """프로바이더 호출 통계를 반환합니다."""
        return self.llm_manager.get_stats()

    def enable_fallback_mode(self) -> None:
        """폴백 모드를 수동으로 활성화합니다."""
        self.llm_manager.enable_fallback_mode()

    def disable_fallback_mode(self) -> None:
        """폴백 모드를 수동으로 비활성화합니다."""
        self.llm_manager.disable_fallback_mode()
