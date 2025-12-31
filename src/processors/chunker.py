from typing import List, Dict

class Chunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of roughly `chunk_size` characters (or tokens).
        This is a simple character-based splitter for now. 
        For better results with huge texts, we might want a token-based splitter,
        but for Korean, character count is often a reasonable proxy or use spaces.
        """
        # Simple splitting by words/spaces to avoid cutting words in half
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        # We walk through words, adding to current chunk
        # This implementation is a "rolling window" style if we wanted strict overlap
        # But a simpler "stride" approach is often easier to adhere to overlap exactness.
        
        # Let's use a stepping approach
        # Note: chunk_size 300 tokens approx 1000-1200 characters for English,
        # For Korean, 300 tokens might be ~500-600 characters or words.
        # Let's treat chunk_size as "approximated word count" for simplicity
        
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1
            
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            
            # If this chunk reached the end of the text, we stop
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def split_transcript_with_timestamps(self, transcript_segments: List[Dict]) -> List[Dict]:
        """
        타임스탬프가 있는 트랜스크립트를 청크로 분할하고 시작/종료 시간을 매핑합니다.

        Args:
            transcript_segments: youtube-transcript-api 형식의 세그먼트 리스트
                               [{'text': str, 'start': float, 'duration': float}, ...]

        Returns:
            청크 리스트 [{'text': str, 'start_time': float, 'end_time': float}, ...]
        """
        if not transcript_segments:
            return []

        # 각 단어에 타임스탬프 할당
        word_timestamps = []
        for segment in transcript_segments:
            words = segment['text'].split()
            segment_start = segment['start']
            segment_duration = segment['duration']

            # 세그먼트 내 단어들에 균등하게 시간 분배
            if words:
                time_per_word = segment_duration / len(words)
                for i, word in enumerate(words):
                    word_timestamps.append({
                        'word': word,
                        'time': segment_start + (i * time_per_word)
                    })

        if not word_timestamps:
            return []

        # 기존 청킹 로직과 동일한 step 계산
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1

        chunks = []
        for i in range(0, len(word_timestamps), step):
            chunk_words = word_timestamps[i:i + self.chunk_size]
            if not chunk_words:
                break

            chunk_text = " ".join(w['word'] for w in chunk_words)
            start_time = chunk_words[0]['time']
            end_time = chunk_words[-1]['time']

            chunks.append({
                'text': chunk_text,
                'start_time': start_time,
                'end_time': end_time
            })

            if i + self.chunk_size >= len(word_timestamps):
                break

        return chunks


# Test
if __name__ == "__main__":
    text = "A B C D E F G H I J K L M N O P"
    chunker = Chunker(chunk_size=5, chunk_overlap=2)
    print(chunker.split_text(text))
    # Expected e.g.: "A B C D E", "D E F G H", ...
