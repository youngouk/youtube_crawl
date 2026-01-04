# Medical RAG 파이프라인

유튜브 의료 채널에서 자막을 수집하고, AI로 정제하여 벡터 검색이 가능한 형태로 저장하는 파이프라인입니다.

## 주요 기능

- **자막 수집**: 전문의 인증 유튜브 채널에서 자막 자동 추출
- **AI 정제**: Gemini를 사용해 의료 용어 교정 및 문장 정리
- **벡터 검색**: Pinecone에 1024차원 임베딩으로 저장
- **클라우드 백업**: Cloudflare R2에 원본 데이터 보관

## 프로젝트 구조

```
├── main.py                 # 메인 실행 파일
├── config/                 # 설정
│   └── settings.py
├── src/
│   ├── collectors/         # 데이터 수집
│   │   ├── youtube_collector.py    # 영상 목록 수집
│   │   └── transcript_collector.py # 자막 추출
│   ├── processors/         # 데이터 처리
│   │   ├── gemini_processor.py     # AI 정제/요약/임베딩
│   │   ├── chunker.py              # 텍스트 청킹
│   │   └── llm_provider.py         # LLM 폴백 관리
│   ├── storage/            # 저장소
│   │   ├── r2_storage.py           # R2 클라우드 저장
│   │   └── state_manager.py        # 처리 상태 관리
│   └── vector_db/          # 벡터 DB
│       └── pinecone_manager.py     # Pinecone 연동
├── scripts/                # 운영 스크립트
│   ├── init_pinecone.py            # DB 초기화
│   ├── reembed_from_r2.py          # 재임베딩
│   └── test_search_quality.py      # 검색 품질 테스트
├── tests/                  # 테스트 (94개)
└── data/                   # 로컬 데이터
    ├── channels.json               # 채널 목록
    └── state.json                  # 처리 상태
```

## 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 변수 설정

`.env` 파일을 생성하고 아래 값을 설정하세요:

```bash
# 필수
YOUTUBE_API_KEY=your_youtube_api_key
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

# 선택 (Google 할당량 초과 시 폴백)
OPENROUTER_API_KEY=your_openrouter_api_key

# R2 스토리지 (선택)
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name
```

## 실행

### 1. Pinecone 인덱스 초기화 (최초 1회)

```bash
python scripts/init_pinecone.py
```

### 2. 파이프라인 실행

```bash
python main.py
```

### 3. 검색 품질 테스트

```bash
python scripts/test_search_quality.py
```

## 처리 흐름

```
1. 채널 목록 로드 (data/channels.json)
2. YouTube API로 영상 목록 수집
3. 각 영상에서 자막 추출
4. Gemini로 자막 정제 (오타 교정, 의료 용어 표준화)
5. 영상 전체 요약 생성
6. 300토큰 단위로 청킹 + 타임스탬프 매핑
7. 각 청크에 문맥(context) 및 토픽 추출
8. 1024차원 임베딩 생성
9. Pinecone에 벡터 저장
10. R2에 원본 데이터 백업
```

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 모듈 테스트
pytest tests/test_gemini_processor.py -v
```

## 기술 스택

| 구성요소 | 기술 |
|---------|------|
| 자막 추출 | youtube-transcript-api |
| AI 처리 | Gemini 2.5 Flash, gemini-embedding-001 |
| 벡터 DB | Pinecone (Serverless, 1024차원, dotproduct) |
| 클라우드 저장 | Cloudflare R2 |
| 폴백 LLM | OpenRouter (google/gemini-2.5-flash) |

## 라이선스

MIT License
