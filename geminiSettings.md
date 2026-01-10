```md
# Gemini CLI 잘 쓰기 가이드 (Claude Code 스타일 셋업 + 실사용 꿀팁)
*(작성일: 2026-01-05)*

## 0) 이 문서가 다루는 것
- Gemini CLI를 **Claude Code처럼 “프로젝트 규칙 + 커맨드 + 훅 + MCP(서드파티 툴)”** 중심으로 세팅하는 방법
- **실제 사용자들이 많이 쓰는 꿀팁 패턴**(커스텀 슬래시 커맨드, 컨텍스트 계층, 자동화, 비용 최적화)
- Claude Code 레거시(예: `CLAUDE.md`, 커맨드/워크플로우)를 **Gemini CLI 구조로 가져오는 방법**

---

## 1) Gemini CLI 핵심 개념 한 장 요약
### 1.1 ReAct 에이전트 + 로컬/원격 툴
Gemini CLI는 터미널에서 Gemini를 쓰는 **오픈소스 에이전트**이며, 내장 툴과 **로컬/원격 MCP 서버**를 붙여서 “생각(Reason) → 행동(Act)” 루프로 작업을 수행합니다. :contentReference[oaicite:0]{index=0}

### 1.2 “Claude Code스럽게” 만드는 4대 축
1) **설정 레이어**: user/project/system + env + CLI args 우선순위 :contentReference[oaicite:1]{index=1}  
2) **컨텍스트 파일**: `GEMINI.md` 계층 로딩(전역→레포→서브폴더) :contentReference[oaicite:2]{index=2}  
3) **커스텀 슬래시 커맨드**: `.toml`로 /deploy 같은 “스킬” 만들기 :contentReference[oaicite:3]{index=3}  
4) **확장/서드파티**: Extensions + MCP + Hooks로 워크플로우 자동화 :contentReference[oaicite:4]{index=4}  

---

## 2) 설정 스코프(Claude Code처럼 User / Repo 분리)
Gemini CLI는 아래 우선순위로 설정이 덮입니다.  
Default → System defaults → User → Project → System override → Env(.env 포함) → CLI args :contentReference[oaicite:5]{index=5}

### 2.1 파일 위치(가장 중요)
- 유저 공통: `~/.gemini/settings.json` :contentReference[oaicite:6]{index=6}  
- 레포 공통: `<repo>/.gemini/settings.json` :contentReference[oaicite:7]{index=7}  

> 팁: 설정 JSON 스키마를 editor autocomplete로 쓰려면, 공식 스키마를 참조할 수 있습니다. :contentReference[oaicite:8]{index=8}

---

## 3) “실제 사용자 꿀팁” 10선 (체감 큰 것만)
### 3.1 컨텍스트는 “한 방에 크게”가 아니라 “계층으로 쪼개기”
`GEMINI.md`는 폴더 트리를 따라 여러 개가 합쳐집니다(전역/레포/컴포넌트 단위). 그래서 모노레포는 **루트 1개 + 서비스별 1개**가 효율적입니다. :contentReference[oaicite:9]{index=9}

### 3.2 `/memory show`로 “지금 모델이 보고 있는 규칙”을 항상 확인
`/memory`는 로드된 `GEMINI.md`의 최종 합본을 보여주고, 수정 후 `/memory refresh`로 재로딩합니다. :contentReference[oaicite:10]{index=10}

### 3.3 반복작업은 무조건 커스텀 슬래시 커맨드(.toml)로 고정
사용자들이 가장 많이 “스킬화”하는 패턴이 **PR 리뷰, 릴리즈 체크, 테스트/린트, 배포, 이슈 템플릿**입니다.  
Gemini CLI는 `~/.gemini/commands/` 또는 `<repo>/.gemini/commands/`에 `.toml`을 두면 `/명령`으로 실행됩니다. :contentReference[oaicite:11]{index=11}

### 3.4 커맨드 네임스페이스로 “팀 표준 워크플로우” 만들기
`<repo>/.gemini/commands/git/commit.toml` → `/git:commit`처럼 폴더 구조가 `:` 네임스페이스가 됩니다. :contentReference[oaicite:12]{index=12}

### 3.5 “파일/폴더를 붙이는 방법”을 습관화
Gemini CLI는 `/` 외에도 `@`로 파일/디렉터리 내용을 프롬프트에 주입합니다(기본적으로 gitignore 존중). :contentReference[oaicite:13]{index=13}

### 3.6 자동화는 Non-interactive 모드로 (CI/스크립트)
파이프 입력 또는 `-p/--prompt`로 실행하고 종료하는 모드가 공식 지원됩니다. :contentReference[oaicite:14]{index=14}  
(단, 커스텀 커맨드/확장은 특정 버전에서 비대화 모드 제약 이슈가 보고된 적이 있으니, 자동화 파이프라인은 안정 버전에서 검증 권장) :contentReference[oaicite:15]{index=15}

### 3.7 비용/속도 최적화: Token Caching 켜기
API 키 인증(= Gemini API key / Vertex AI)을 쓰면 시스템/컨텍스트 토큰을 재사용하는 **token caching**으로 비용을 줄일 수 있습니다. :contentReference[oaicite:16]{index=16}

### 3.8 안전장치: Checkpointing(되돌리기) 적극 활용
Gemini CLI는 파일 수정 전 프로젝트 상태를 스냅샷으로 저장해 되돌릴 수 있는 기능이 있습니다. :contentReference[oaicite:17]{index=17}

### 3.9 IDE 연동은 “diff 승인 플로우” 때문에 체감이 큼
CLI에서 제안한 수정이 IDE에서 diff로 뜨고, 체크/저장으로 승인할 수 있습니다. `/ide install`, `/ide enable` 등 제공. :contentReference[oaicite:18]{index=18}

### 3.10 보안: Hooks/Extensions/MCP는 “신뢰 폴더 + 리뷰”가 기본
프로젝트 훅/확장 훅은 위험할 수 있고, CLI도 경고하지만 **결국 사용자가 리뷰**해야 합니다. :contentReference[oaicite:19]{index=19}  
또한 Trusted Folders로 “프로젝트 설정/훅 로드 자체”를 폴더 단위로 승인하게 할 수 있습니다. :contentReference[oaicite:20]{index=20}

---

## 4) Claude Code 레거시 가져오기 (실전 마이그레이션 체크리스트)
> 목표: Claude Code의 “규칙/커맨드/훅/서드파티”를 Gemini CLI 형태로 재현

### 4.1 `CLAUDE.md` → `GEMINI.md` 변환
- 레포 루트의 `CLAUDE.md` 내용을 `GEMINI.md`로 옮기고,
- 모노레포라면 `services/*/GEMINI.md`처럼 서비스별로 추가합니다.  
Gemini는 계층 컨텍스트를 공식 지원합니다. :contentReference[oaicite:21]{index=21}

### 4.2 Claude의 “자주 쓰는 프롬프트/워크플로우” → `.toml` 커맨드로 스킬화
Gemini 커스텀 커맨드는 TOML 형식이며 `/help`에 노출됩니다. :contentReference[oaicite:22]{index=22}  
또한 슬래시 커맨드는 로컬 `.toml`뿐 아니라 **MCP 프롬프트로도 정의 가능**합니다. :contentReference[oaicite:23]{index=23}

### 4.3 훅(Hooks) 이식
Claude Code도 훅을 지원하지만, Gemini CLI 역시 훅 시스템이 있고 “프로젝트 훅/확장 훅”은 특히 주의가 필요합니다. :contentReference[oaicite:24]{index=24}  
- Claude에서 Pre/Post로 하던 “자동 포맷/린트/커맨드 가드레일”은 Gemini Hooks로 유사 구현 가능

### 4.4 MCP(서드파티 툴) 이식
Gemini CLI는 MCP 서버를 설정으로 붙이며, allow/exclude 같은 제어도 가능합니다. :contentReference[oaicite:25]{index=25}  
- Claude에서 쓰던 MCP 서버가 표준 MCP라면, Gemini에서도 대체로 재사용 가능(클라이언트 차이만 조정)

---

## 5) Gemini CLI만의 “서드파티/확장” 핵심
### 5.1 Extensions: 설치/업데이트/비활성화가 공식 커맨드로 제공
- 설치: `gemini extensions install <GitHub URL | local path> ...`
- 업데이트/링크/disable scope(user/workspace) 등 지원 :contentReference[oaicite:26]{index=26}  
- `--consent`는 보안 위험을 인지하고 확인 프롬프트를 스킵하는 옵션입니다. :contentReference[oaicite:27]{index=27}

> 확장 갤러리의 확장들은 “구글이 보증하지 않음” 경고가 명시되어 있습니다. :contentReference[oaicite:28]{index=28}

### 5.2 Extensions가 담을 수 있는 것(실무 관점)
커뮤니티에서는 확장에 **커스텀 프롬프트, GEMINI.md, 커맨드, 테마** 등을 묶어 배포하는 패턴이 널리 쓰입니다. :contentReference[oaicite:29]{index=29}

### 5.3 “뭘 깔아야 하냐” 빠른 예시(갤러리 기반)
갤러리에는 다양한 MCP/컨텍스트 확장이 있고, 예를 들어 Exa MCP(웹 검색/크롤링), clasp(앱스 스크립트) 같은 항목이 노출됩니다. :contentReference[oaicite:30]{index=30}  
*(실제로는 팀의 스택(Jira/Confluence/GitHub 등)에 맞춰 MCP 확장을 고르는 게 효율적)* :contentReference[oaicite:31]{index=31}

---

## 6) 추천 레포 구조 (모노레포 기준)
```

repo/
GEMINI.md
.gemini/
settings.json
.geminiignore
commands/
pr/review.toml
test/run.toml
services/
api/
GEMINI.md
web/
GEMINI.md

````
- 루트 `GEMINI.md`: 공통 규칙(코딩 규칙/테스트 원칙/PR 형식)
- 서비스별 `GEMINI.md`: 실행 커맨드/아키텍처/폴더 설명(서비스 특화)

`GEMINI.md`는 계층으로 로드됩니다. :contentReference[oaicite:32]{index=32}

---

## 7) 바로 복붙 가능한 템플릿

### 7.1 커스텀 커맨드 예시: PR 리뷰
**파일**: `<repo>/.gemini/commands/pr/review.toml`  
(→ `/pr:review`)

```toml
description = "PR diff/변경파일 기반으로 리뷰 체크리스트 + 리스크 + 테스트 제안 생성"
prompt = """
다음 정보를 기반으로 PR 리뷰를 수행해줘.

1) 변경 요약(한 문단)
2) 위험도(높음/중간/낮음) + 근거
3) 버그 가능 지점 Top 5
4) 테스트 제안(유닛/통합/e2e)
5) 롤백/모니터링 포인트

추가 인자: {{args}}
"""
````

* 네임스페이스 규칙(폴더 → `:`)과 `{{args}}` 인자 주입은 공식 동작입니다. ([Gemini CLI][1])

### 7.2 커스텀 커맨드 예시: 테스트 실행 가이드

**파일**: `<repo>/.gemini/commands/test/run.toml` (→ `/test:run`)

```toml
description = "레포 테스트 커맨드/실행 순서/실패 분석 루틴"
prompt = """
우리 레포에서 테스트를 안전하게 실행하고 실패를 분석하는 절차를 제시해줘.

- 우선 로컬에서 실행할 커맨드를 제안하고 (패키지 매니저 감지)
- 실패 시 로그에서 우선 확인할 5가지
- flake 의심 시 재실행/격리 전략
- CI와 로컬 차이 추적 방법

추가 인자: {{args}}
"""
```

### 7.3 `@`로 컨텍스트 주입 패턴 (사용자들이 가장 자주 씀)

```bash
# 파일 1개
@README.md 이 레포의 진입점을 10줄로 요약해줘

# 디렉토리
@services/api/ 이 서비스의 핵심 흐름과 엔트리포인트를 알려줘
```

`@`는 내부적으로 여러 파일을 읽어 프롬프트에 삽입하며, 기본적으로 git-ignore 필터링이 적용됩니다. ([Gemini CLI][2])

### 7.4 Non-interactive 실행(스크립트/자동화)

```bash
echo "이 레포에서 릴리즈 체크리스트 만들어줘" | gemini
gemini -p "services/api의 에러 로깅 포인트 찾아줘"
```

비대화 모드는 공식 가이드에 포함됩니다. ([Gemini CLI][3])

---

## 8) 운영/보안 체크리스트(팀 배포용)

* [ ] Trusted Folders 활성화(프로젝트별 승인 플로우) ([Gemini CLI][4])
* [ ] Hooks/Extensions/MCP는 “신뢰 가능한 레포/작성자만”, 도입 시 코드 리뷰 ([Gemini CLI][5])
* [ ] Extensions는 user/workspace 단위 enable/disable로 롤아웃 ([Gemini CLI][6])
* [ ] 자동화/CI는 stable 채널에서 고정하고 릴리즈 노트로 회귀 확인 ([Gemini CLI][7])

---

## 9) 흔한 함정 & 대응(이슈 기반)

* `.geminiignore`로 `GEMINI.md`를 숨기려 했는데 읽히는 이슈가 보고된 적이 있습니다(템플릿 저장 같은 케이스). 이런 경우 템플릿 경로 분리/버전 업데이트로 대응 권장. ([GitHub][8])
* 커스텀 커맨드가 `~/.gemini/commands`에서 로드되지 않는 이슈가 특정 시점에 보고되었습니다. 증상 시 릴리즈 채널/버전 확인 및 최신 안정판에서 재검증 권장. ([GitHub][9])

---

## 10) 다음 단계(원하면 내가 “레포 맞춤”으로 완성해줄 수 있는 것)

아래 3가지를 붙여주면, **Claude Code 레거시 → Gemini CLI 세팅**을 “팀 배포 가능한 형태”로 변환해서 제공할 수 있어요.

1. 현재 사용 중인 `CLAUDE.md`(또는 팀 룰 문서)
2. Claude에서 자주 쓰는 워크플로우 목록(예: PR review, release, test, deploy 등)
3. 붙이고 싶은 서드파티(예: GitHub/Jira/Confluence/Slack 등)

(추가로, 어떤 서비스가 모노레포에서 가장 중요한지 1~2개만 알려주면 `services/*/GEMINI.md`까지 실전형으로 쪼개서 제안 가능)

```
::contentReference[oaicite:42]{index=42}
```

[1]: https://geminicli.com/docs/cli/custom-commands/ "Custom commands | Gemini CLI"
[2]: https://geminicli.com/docs/cli/commands/ "CLI commands | Gemini CLI"
[3]: https://geminicli.com/docs/cli/?utm_source=chatgpt.com "Gemini CLI"
[4]: https://geminicli.com/docs/cli/trusted-folders/?utm_source=chatgpt.com "Trusted Folders - Gemini CLI"
[5]: https://geminicli.com/docs/hooks/ "Gemini CLI hooks | Gemini CLI"
[6]: https://geminicli.com/docs/extensions/ "Gemini CLI extensions | Gemini CLI"
[7]: https://geminicli.com/docs/changelogs/releases/?utm_source=chatgpt.com "Gemini CLI changelog"
[8]: https://github.com/google-gemini/gemini-cli/issues/3486?utm_source=chatgpt.com "GEMINI.md files read even when they are listed in . ..."
[9]: https://github.com/google-gemini/gemini-cli/issues/4834?utm_source=chatgpt.com "Custom commands are not being loaded from ~/.gemini ..."
