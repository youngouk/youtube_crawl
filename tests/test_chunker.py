import pytest
from src.processors.chunker import Chunker

def test_chunker_initialization():
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    assert chunker.chunk_size == 10
    assert chunker.chunk_overlap == 2

def test_split_text_basic():
    # Setup
    text = "one two three four five six seven eight nine ten" # 10 words
    chunker = Chunker(chunk_size=5, chunk_overlap=2)
    
    # Execution
    chunks = chunker.split_text(text)
    
    # Verification
    # Expected chunks (stride = 5-2=3):
    # 1. "one two three four five" (0-5)
    # 2. "four five six seven eight" (3-8)
    # 3. "seven eight nine ten"    (6-10) - remaining
    
    assert len(chunks) == 3
    assert chunks[0] == "one two three four five"
    assert chunks[1] == "four five six seven eight"
    assert chunks[2] == "seven eight nine ten"

def test_split_text_smaller_than_chunk_size():
    text = "small text"
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split_text(text)
    
    assert len(chunks) == 1
    assert chunks[0] == "small text"

def test_split_text_empty():
    chunker = Chunker()
    assert chunker.split_text("") == []

def test_split_text_exact_overlap_boundary():
    # 10 words, chunk 5, overlap 0 -> 2 chunks
    text = "one two three four five six seven eight nine ten"
    chunker = Chunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.split_text(text)

    assert len(chunks) == 2
    assert chunks[0] == "one two three four five"
    assert chunks[1] == "six seven eight nine ten"


def test_split_transcript_with_timestamps():
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
    assert chunks[0]['end_time'] >= 2.0  # 최소 2초 이상 (4단어 = 최소 2개 세그먼트)


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
