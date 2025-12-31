import google.generativeai as genai
from config.settings import settings
import time
from typing import Optional, List

class GeminiProcessor:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def refine_transcript(self, raw_text: str) -> str:
        """
        Refines the transcript using Gemini to correct typos and medical terms.
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
        """Generates a summary of the video content."""
        prompt = f"""
        다음은 소아과 관련 유튜브 영상의 자막입니다. 이 영상의 핵심 내용을 3-5문장으로 요약해주세요.
        이 요약은 나중에 이 영상의 특정 부분이 어떤 맥락인지 파악하는 데 사용됩니다.

        자막:
        {full_text}
        """
        return self._generate_patience(prompt)

    def generate_chunk_context(self, chunk_text: str, video_summary: str) -> str:
        """Generates context for a specific chunk based on the video summary."""
        prompt = f"""
        전체 문서(영상) 요약:
        {video_summary}

        아래 청크가 전체 문서에서 어떤 맥락인지 1-2문장으로 설명하세요.
        청크 내용을 단순히 반복하지 말고, 이 내용이 전체 주제 내에서 어떤 역할을 하는지 서술하세요.

        청크:
        {chunk_text}
        """
        return self._generate_patience(prompt)

    def _generate_patience(self, prompt: str, retries: int = 3) -> str:
        for i in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"[Gemini] Error generating content (Attempt {i+1}/{retries}): {e}")
                time.sleep(2 ** i)  # Exponential backoff
        return ""

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for the given text using gemini-embedding-001."""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",  # 768차원, 한국어 의료 콘텐츠에 최적화
                content=text,
                task_type="retrieval_document",
                title="Medical RAG Chunk"
            )
            return result['embedding']
        except Exception as e:
            print(f"[Gemini] Error generating embedding: {e}")
            return []
