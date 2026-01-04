# Medical RAG Pipeline 개선 구현 계획서

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 설계 문서와 실제 구현 간의 4가지 갭(임베딩 모델명, 메타데이터 확장, 타임스탬프 매핑, retry_count 추적)을 TDD 방식으로 해결

**Architecture:** 기존 파이프라인 구조를 유지하면서 각 컴포넌트를 개선. Chunker → GeminiProcessor → StateManager → Main 순서로 변경 적용

**Tech Stack:** Python 3.x, pytest, pytest-mock, google-generativeai, Pinecone

---

## Task 1: 임베딩 모델명 수정

**Files:**
- Modify: `src/processors/gemini_processor.py:75`
- Modify: `tests/test_gemini_processor.py`

### Step 1.1: 실패하는 테스트 작성

`tests/test_gemini_processor.py` 파일 끝에 추가:

```python
def test_gemini_embedding_uses_correct_model(mocker):
    """임베딩 생성 시 gemini-embedding-001 모델을 사용하는지 검증"""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mock_genai = mocker.patch('src.processors.gemini_processor.genai')
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2]}

    processor = GeminiProcessor()
    processor.get_embedding("테스트 텍스트")

    # 올바른 모델명으로 호출되었는지 검증
    call_args = mock_genai.embed_content.call_args
    assert call_args.kwargs['model'] == "models/gemini-embedding-001", \
        f"Expected 'models/gemini-embedding-001' but got '{call_args.kwargs['model']}'"
```

### Step 1.2: 테스트 실패 확인

실행: `pytest tests/test_gemini_processor.py::test_gemini_embedding_uses_correct_model -v`
예상: FAIL - "Expected 'models/gemini-embedding-001' but got 'models/emb-multilingual-001'"

### Step 1.3: 최소 구현으로 테스트 통과

`src/processors/gemini_processor.py:74-75` 수정:

```python
    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for the given text using gemini-embedding-001."""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",  # 수정: emb-multilingual-001 → gemini-embedding-001
                content=text,
                task_type="retrieval_document",
                title="Medical RAG Chunk"
            )
            return result['embedding']
```

### Step 1.4: 테스트 통과 확인

실행: `pytest tests/test_gemini_processor.py::test_gemini_embedding_uses_correct_model -v`
예상: PASS

### Step 1.5: 커밋

```bash
git add src/processors/gemini_processor.py tests/test_gemini_processor.py
git commit -m "수정: 임베딩 모델명을 gemini-embedding-001로 변경

emb-multilingual-001에서 gemini-embedding-001로 수정하여
설계 문서의 명세와 일치시킴"
```

---

## Task 2: Pinecone 메타데이터 확장

**Files:**
- Modify: `main.py:146-153`
- Create: `tests/test_main_metadata.py`

### Step 2.1: 실패하는 테스트 작성

`tests/test_main_metadata.py` 새 파일 생성:

```python
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

    # 타입 검증
    assert metadata['source_type'] == 'youtube'
    assert metadata['video_url'].startswith('https://www.youtube.com/watch?v=')
    assert isinstance(metadata['is_verified_professional'], bool)
```

### Step 2.2: 테스트 실패 확인

실행: `pytest tests/test_main_metadata.py::test_chunk_metadata_has_required_fields -v`
예상: FAIL - "cannot import name 'create_chunk_metadata' from 'main'"

### Step 2.3: 메타데이터 생성 함수 구현

`main.py`에 함수 추가 (import 섹션 아래):

```python
from datetime import datetime, timezone

def create_chunk_metadata(
    chunk_text: str,
    context: str,
    video_id: str,
    video_title: str,
    channel_name: str,
    chunk_index: int,
    published_at: str,
    is_verified_professional: bool,
    specialty: str
) -> dict:
    """
    Pinecone 저장용 청크 메타데이터를 생성합니다.

    Args:
        chunk_text: 청크 텍스트
        context: Contextual Retrieval로 생성된 컨텍스트
        video_id: YouTube 비디오 ID
        video_title: 비디오 제목
        channel_name: 채널 이름
        chunk_index: 청크 인덱스
        published_at: 비디오 게시일 (ISO 8601)
        is_verified_professional: 의료 전문가 인증 여부
        specialty: 전문 분야 (예: 소아과)

    Returns:
        메타데이터 딕셔너리
    """
    return {
        'text': chunk_text,
        'context': context,
        'video_id': video_id,
        'video_title': video_title,
        'channel_name': channel_name,
        'chunk_index': chunk_index,
        'source_type': 'youtube',
        'video_url': f'https://www.youtube.com/watch?v={video_id}',
        'published_at': published_at,
        'is_verified_professional': is_verified_professional,
        'specialty': specialty,
        'processed_at': datetime.now(timezone.utc).isoformat()
    }
```

### Step 2.4: 테스트 통과 확인

실행: `pytest tests/test_main_metadata.py::test_chunk_metadata_has_required_fields -v`
예상: PASS

### Step 2.5: main.py의 기존 코드에서 새 함수 사용하도록 수정

`main.py`에서 기존 메타데이터 생성 부분을 찾아 수정:

```python
# 기존 코드 (약 146-153 라인):
# "metadata": {
#     "text": chunk_text,
#     ...
# }

# 수정된 코드:
metadata = create_chunk_metadata(
    chunk_text=chunk_text,
    context=context,
    video_id=video_id,
    video_title=video_title,
    channel_name=channel['name'],
    chunk_index=idx,
    published_at=video.get('published_at', ''),
    is_verified_professional=channel.get('is_verified_professional', False),
    specialty=channel.get('specialty', '')
)
vectors.append({
    "id": f"{video_id}_chunk_{idx}",
    "values": embedding,
    "metadata": metadata
})
```

### Step 2.6: 전체 테스트 실행 및 커밋

실행: `pytest tests/test_main_metadata.py -v`
예상: PASS

```bash
git add main.py tests/test_main_metadata.py
git commit -m "기능: Pinecone 메타데이터 확장

source_type, video_url, published_at, is_verified_professional,
specialty, processed_at 필드 추가.
create_chunk_metadata() 함수로 메타데이터 생성 로직 분리"
```

---

## Task 3: 타임스탬프 매핑 기능 추가

**Files:**
- Modify: `src/processors/chunker.py`
- Modify: `tests/test_chunker.py`

### Step 3.1: 실패하는 테스트 작성

`tests/test_chunker.py` 파일 끝에 추가:

```python
def test_split_text_with_timestamps():
    """타임스탬프가 있는 트랜스크립트를 청크로 분할 시 시작/종료 시간 반환"""
    # 타임스탬프가 있는 트랜스크립트 형식 (youtube-transcript-api 출력)
    transcript_segments = [
        {'text': 'one two', 'start': 0.0, 'duration': 2.0},
        {'text': 'three four', 'start': 2.0, 'duration': 2.0},
        {'text': 'five six', 'start': 4.0, 'duration': 2.0},
        {'text': 'seven eight', 'start': 6.0, 'duration': 2.0},
        {'text': 'nine ten', 'start': 8.0, 'duration': 2.0},
    ]

    chunker = Chunker(chunk_size=4, chunk_overlap=1)
    chunks = chunker.split_transcript_with_timestamps(transcript_segments)

    # 반환 형식 검증: [{'text': str, 'start_time': float, 'end_time': float}, ...]
    assert len(chunks) >= 2
    assert 'text' in chunks[0]
    assert 'start_time' in chunks[0]
    assert 'end_time' in chunks[0]

    # 첫 번째 청크 검증
    assert chunks[0]['start_time'] == 0.0
    assert chunks[0]['end_time'] >= 4.0  # 최소 4초 이상
```

### Step 3.2: 테스트 실패 확인

실행: `pytest tests/test_chunker.py::test_split_text_with_timestamps -v`
예상: FAIL - "AttributeError: 'Chunker' object has no attribute 'split_transcript_with_timestamps'"

### Step 3.3: 타임스탬프 매핑 메서드 구현

`src/processors/chunker.py`에 메서드 추가:

```python
def split_transcript_with_timestamps(self, transcript_segments: List[dict]) -> List[dict]:
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
```

### Step 3.4: 테스트 통과 확인

실행: `pytest tests/test_chunker.py::test_split_text_with_timestamps -v`
예상: PASS

### Step 3.5: 추가 엣지 케이스 테스트

`tests/test_chunker.py`에 추가:

```python
def test_split_transcript_empty():
    """빈 트랜스크립트 처리"""
    chunker = Chunker()
    assert chunker.split_transcript_with_timestamps([]) == []

def test_split_transcript_single_segment():
    """단일 세그먼트 처리"""
    transcript = [{'text': 'hello world', 'start': 0.0, 'duration': 2.0}]
    chunker = Chunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.split_transcript_with_timestamps(transcript)

    assert len(chunks) == 1
    assert chunks[0]['text'] == 'hello world'
    assert chunks[0]['start_time'] == 0.0
```

### Step 3.6: 테스트 실행 및 커밋

실행: `pytest tests/test_chunker.py -v`
예상: 모든 테스트 PASS

```bash
git add src/processors/chunker.py tests/test_chunker.py
git commit -m "기능: 타임스탬프 매핑을 포함한 트랜스크립트 청킹 추가

split_transcript_with_timestamps() 메서드 추가로
각 청크의 시작/종료 시간을 추적할 수 있음"
```

---

## Task 4: retry_count 추적 기능 추가

**Files:**
- Modify: `src/storage/state_manager.py`
- Modify: `tests/test_state_manager.py`

### Step 4.1: 실패하는 테스트 작성

`tests/test_state_manager.py` 파일 끝에 추가:

```python
def test_retry_count_tracking(tmp_path, mocker):
    """실패 시 retry_count가 증가하는지 검증"""
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    manager = StateManager()

    # 첫 번째 시도 실패
    manager.update_video_status("vid1", "failed", error="Error 1")
    status = manager.get_video_status("vid1")
    assert status.get('retry_count', 0) == 1

    # 두 번째 시도 실패
    manager.update_video_status("vid1", "failed", error="Error 2")
    status = manager.get_video_status("vid1")
    assert status['retry_count'] == 2

    # 성공 시 retry_count 유지 (리셋하지 않음)
    manager.update_video_status("vid1", "completed")
    status = manager.get_video_status("vid1")
    assert status['retry_count'] == 2

def test_retry_count_not_incremented_on_success(tmp_path, mocker):
    """성공 상태에서는 retry_count가 증가하지 않음"""
    mocker.patch('src.storage.state_manager.settings.LOCAL_DATA_DIR', tmp_path)

    manager = StateManager()

    manager.update_video_status("vid1", "processing")
    status = manager.get_video_status("vid1")
    assert status.get('retry_count', 0) == 0

    manager.update_video_status("vid1", "completed")
    status = manager.get_video_status("vid1")
    assert status.get('retry_count', 0) == 0
```

### Step 4.2: 테스트 실패 확인

실행: `pytest tests/test_state_manager.py::test_retry_count_tracking -v`
예상: FAIL - AssertionError: assert 0 == 1 (retry_count가 없거나 증가하지 않음)

### Step 4.3: retry_count 로직 구현

`src/storage/state_manager.py`의 `update_video_status` 메서드 수정:

```python
def update_video_status(self, video_id: str, status: str, step: str = None, error: str = None):
    """
    비디오 처리 상태를 업데이트합니다.

    Args:
        video_id: YouTube 비디오 ID
        status: 상태 (pending, processing, completed, failed)
        step: 현재 처리 단계 (선택)
        error: 에러 메시지 (선택, failed 상태 시)
    """
    state = self._load_state()

    if video_id not in state:
        state[video_id] = {'retry_count': 0}

    current = state[video_id]
    current['status'] = status
    current['updated_at'] = datetime.now(timezone.utc).isoformat()

    if step:
        current['current_step'] = step

    if error:
        current['error_message'] = error

    # 실패 시 retry_count 증가
    if status == 'failed':
        current['retry_count'] = current.get('retry_count', 0) + 1

    self._save_state(state)
```

파일 상단에 import 추가 (없는 경우):

```python
from datetime import datetime, timezone
```

### Step 4.4: 테스트 통과 확인

실행: `pytest tests/test_state_manager.py::test_retry_count_tracking -v`
예상: PASS

실행: `pytest tests/test_state_manager.py::test_retry_count_not_incremented_on_success -v`
예상: PASS

### Step 4.5: 전체 StateManager 테스트 실행

실행: `pytest tests/test_state_manager.py -v`
예상: 모든 테스트 PASS

### Step 4.6: 커밋

```bash
git add src/storage/state_manager.py tests/test_state_manager.py
git commit -m "기능: 비디오 처리 실패 시 retry_count 추적 추가

failed 상태로 업데이트될 때마다 retry_count 증가.
이를 통해 재시도 횟수를 추적하고 최대 재시도 제한 구현 가능"
```

---

## 최종 검증

### 전체 테스트 실행

```bash
pytest tests/ -v
```

예상: 모든 테스트 PASS

### 린트 검사 (선택)

```bash
ruff check src/ tests/
```

### 최종 커밋 (태그)

```bash
git tag v0.2.0 -m "설계 문서 대비 구현 갭 해결"
```

---

## 구현 순서 요약

| 순서 | Task | 파일 | 예상 시간 |
|------|------|------|----------|
| 1 | 임베딩 모델명 수정 | gemini_processor.py | 5분 |
| 2 | 메타데이터 확장 | main.py | 10분 |
| 3 | 타임스탬프 매핑 | chunker.py | 15분 |
| 4 | retry_count 추적 | state_manager.py | 10분 |

총 예상 시간: 40분
