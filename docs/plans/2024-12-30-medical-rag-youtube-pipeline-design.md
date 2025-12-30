# 의료 RAG 챗봇 - 유튜브 데이터 수집 파이프라인 설계

> 작성일: 2024-12-30
> 상태: 설계 완료, 구현 대기

## 개요

RAG 챗봇의 데이터소스로 사용할 의료정보를 유튜브에서 수집하는 파이프라인 설계 문서입니다.
소아과/유아교육 전문 채널의 영상 자막을 수집하고, 검색 최적화된 형태로 저장합니다.

## 목표

- 전문의 인증 채널의 의료 콘텐츠 수집
- RAG 검색에 최적화된 청킹 및 임베딩
- 출처 추적 가능한 메타데이터 관리
- 확장 가능한 파이프라인 구조

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    데이터 수집 파이프라인                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [채널 목록]  →  [YouTube API]  →  [자막 추출]  →  [Gemini 정제]   │
│      │              │                 │               │         │
│      ▼              ▼                 ▼               ▼         │
│  수동 큐레이션    영상 메타데이터      원본 자막        정제된 텍스트   │
│  (전문의 채널)    (제목, 설명 등)     (VTT/SRT)       (교정 완료)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         저장소 구성                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Cloudflare R2]                   [Pinecone]                   │
│   ├── 원본 자막 (.json)             ├── 벡터 (dense, 768d)       │
│   ├── 정제된 텍스트 (.json)          ├── 키워드 (sparse/BM25)     │
│   └── 영상 메타데이터                 └── 메타데이터 필터           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 기술 스택

| 구성요소 | 선택 | 이유 |
|---------|------|------|
| **자막 추출** | youtube-transcript-api | 무료, 간편, 타임스탬프 포함 |
| **자막 정제** | Gemini 1.5 Flash | 저비용, 한국어 우수 |
| **청킹** | 고정 크기 + Contextual Retrieval | 비용 효율 + 품질 |
| **임베딩** | gemini-embedding-001 | 최신, MTEB 1위, 768d |
| **벡터 DB** | Pinecone (Serverless) | 하이브리드 검색, 관리형 |
| **스토리지** | Cloudflare R2 | S3 호환, egress 무료 |
| **상태 관리** | DynamoDB | 처리 상태 추적 |

---

## 3. 메타데이터 스키마

각 청크에 저장되는 메타데이터:

```python
{
    # === 출처 정보 (RAG 응답에 표시) ===
    "source_type": "youtube",           # youtube | blog | paper 등
    "channel_id": "UC...",              # 유튜브 채널 ID
    "channel_name": "삐뽀삐뽀 119 소아과",
    "video_id": "dQw4w9WgXcQ",
    "video_title": "아이 열 날 때 응급실 가야 할 때",
    "video_url": "https://youtube.com/watch?v=...",
    "published_at": "2024-03-15",

    # === 신뢰도 필터링 ===
    "is_verified_professional": True,   # 전문의 인증 여부
    "specialty": "pediatrics",          # pediatrics | nutrition | development
    "credentials": "소아청소년과 전문의",

    # === 청크 위치 ===
    "chunk_index": 3,
    "timestamp_start": "02:30",
    "timestamp_end": "04:15",

    # === 검색 최적화 ===
    "topics": ["발열", "응급실", "해열제"],
    "processed_at": "2024-12-30"
}
```

---

## 4. 데이터 수집 파이프라인

### 4.1 채널 큐레이션 (수동)

전문의 인증은 API로 자동화 불가 → `channels.json`에 수동 관리:

```json
{
  "channel_id": "UC...",
  "name": "삐뽀삐뽀 119 소아과",
  "is_verified_professional": true,
  "credentials": "하정훈 소아청소년과 전문의",
  "specialty": ["pediatrics"],
  "priority": "high"
}
```

### 4.2 영상 목록 수집

- YouTube Data API 사용
- `channels.list` → `playlistItems.list` (uploads playlist)
- API 할당량: 10,000 units/일 (무료), 영상 1개 조회 ≈ 1-3 units

### 4.3 자막 추출

- 라이브러리: `youtube-transcript-api`
- 우선순위: 수동 자막(ko) → 자동 생성 자막(ko) → 자동 생성(en)
- 출력: 타임스탬프 포함 텍스트 세그먼트

---

## 5. 자막 정제 (Gemini Flash)

### 프롬프트 설계

```
역할: 소아과 의료 콘텐츠 전문 교정자

작업:
1. 발음 오인식 교정 (여리→열이, 해여제→해열제)
2. 띄어쓰기/맞춤법 수정
3. 의학 용어 표준화 (로타 바이러스→로타바이러스)
4. 불완전한 문장 연결

금지:
- 의미 변경 또는 추가 금지
- 원본에 없는 의학 정보 삽입 금지
```

### 처리 단위

- 영상 1개 = 1 API 호출 (청크 분할 전 전체 정제)

---

## 6. 청킹 전략

### 설정

```python
chunking_config = {
    "strategy": "fixed_size",
    "chunk_size": 300,              # 300 토큰
    "chunk_overlap": 50,            # 50 토큰 오버랩
    "min_chunk_size": 100,          # 최소 크기
}
```

### 선택 이유

- 유튜브 자막: 이미 시간순 구조화 → 시맨틱 청커 불필요
- Contextual Retrieval로 문맥 보강 → 고정 크기로 충분
- 의료 정보 밀도 높음 → 작은 청크(300토큰)가 정밀 검색에 유리

---

## 7. Contextual Retrieval 구현

### 파이프라인

```
Step 1: 영상 전체 요약 생성 (1회)
  └─ Gemini Flash: "이 영상은 소아 발열 대처법에 대해 설명합니다..."

Step 2: 고정 크기 청킹 (300토큰, 50 오버랩)

Step 3: 각 청크별 문맥 생성 (청크 수만큼 호출)
  └─ Gemini Flash: "이 부분은 해열제 복용 시점을 설명합니다."

Step 4: 문맥 + 청크 결합 → 임베딩
  └─ final_text = f"{context}\n\n{chunk_text}"
  └─ gemini-embedding-001 (768 dim)
```

### 프롬프트

```
전체 문서: {영상 요약}
아래 청크가 전체 문서에서 어떤 맥락인지 1-2문장으로 설명하세요.
청크 내용을 반복하지 마세요.

청크: {chunk_text}
```

---

## 8. 임베딩 & 인덱싱

### 임베딩 모델

- **모델**: `gemini-embedding-001` (2025년 GA)
- **차원**: 768 (MRL 기법으로 truncate)
- **특징**: 100+ 언어 지원, MTEB 다국어 1위
- **비용**: $0.15 / 1M tokens

### Pinecone 인덱스

```python
index_config = {
    "name": "medical-rag-hybrid",
    "dimension": 768,
    "metric": "dotproduct",         # 하이브리드 검색용
    "spec": "Serverless (AWS us-east-1)"
}
```

### Upsert 구조

```python
{
    "id": "{video_id}_{chunk_index}",
    "values": [0.1, 0.2, ...],      # dense vector (768d)
    "sparse_values": {...},          # BM25 sparse
    "metadata": { ... }              # 섹션 3 참조
}
```

---

## 9. 스토리지 구조 (Cloudflare R2)

```
bucket/
├── raw/
│   ├── channels.json                    # 채널 목록
│   └── videos/{channel_id}/
│       └── list.json                    # 영상 목록
├── transcripts/
│   └── {video_id}/
│       ├── raw.json                     # 원본 자막 (타임스탬프)
│       └── refined.json                 # 정제된 자막
├── chunks/
│   └── {video_id}/
│       ├── chunks.json                  # 청크 목록
│       └── contextual.json              # 문맥 포함 청크
└── metadata/
    └── {video_id}.json                  # 영상 메타데이터
```

---

## 10. 실행 워크플로우

```
1. 초기화     → channels.json 로드
2. 영상 수집  → YouTube Data API → R2 저장
3. 자막 추출  → youtube-transcript-api → R2 저장
4. 자막 정제  → Gemini Flash → R2 저장
5. 청킹      → 300토큰 고정 크기 → R2 저장
6. 문맥 생성  → Contextual Retrieval (Gemini Flash)
7. 임베딩    → gemini-embedding-001
8. 인덱싱    → Pinecone upsert
```

### 실행 모드

- **초기 수집**: 전체 채널의 모든 영상 처리 (1회성)
- **증분 수집**: 새 영상만 감지하여 처리 (주기적, 예: 주 1회)
- **재처리**: 특정 영상/채널 재처리 (품질 개선 시)

---

## 11. 에러 핸들링

### 재시도 정책

- API 호출 실패 → 최대 3회 재시도 (exponential backoff)
- 자막 없음 → 스킵 후 로그 기록
- 할당량 초과 → 대기 후 재시도

### 상태 추적 (DynamoDB)

```python
{
    "video_id": "xxx",
    "status": "completed | failed | pending | processing",
    "current_step": "chunking",
    "retry_count": 0,
    "error_message": None,
    "updated_at": "2024-12-30T10:00:00Z"
}
```

### 실패 복구

- 실패한 영상 목록 별도 큐 관리
- 일일 배치로 실패 항목 재처리
- 3회 이상 실패 시 수동 검토 플래그

---

## 12. 비용 추정 (1,000개 영상)

| 항목 | 비용 |
|------|------|
| Gemini Flash (정제 + 문맥) | ~$3-4 |
| gemini-embedding-001 | ~$0.50 |
| Pinecone | 무료 티어 (100K 벡터) |
| Cloudflare R2 | ~$0.01 |
| YouTube API | 무료 (할당량 내) |
| **총합** | **~$5 이내** |

---

## 13. 확장성

### 다른 데이터 소스 추가 시

```python
class DataCollector(ABC):
    @abstractmethod
    def collect(self, source_config) -> RawData

    @abstractmethod
    def get_metadata(self) -> Metadata

class YouTubeCollector(DataCollector): ...
class BlogCollector(DataCollector): ...
class PaperCollector(DataCollector): ...
```

### 메타데이터 통합

- 공통 필드: `source_type`, `source_url`, `author`, `is_verified_professional`
- 소스별 추가 필드:
  - youtube: `video_id`, `timestamp_start`, `channel_id`
  - blog: `site_name`, `category`
  - paper: `doi`, `journal`, `peer_reviewed`

---

## 다음 단계

1. [ ] 프로젝트 초기 구조 생성
2. [ ] 채널 목록 (channels.json) 큐레이션
3. [ ] YouTube 수집기 구현
4. [ ] Gemini 정제 파이프라인 구현
5. [ ] 청킹 + Contextual Retrieval 구현
6. [ ] Pinecone 인덱싱 구현
7. [ ] 증분 수집 스케줄러 구현
8. [ ] 테스트 및 품질 검증
