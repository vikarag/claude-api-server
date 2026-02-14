# Claude API Server - 아키텍처 및 기술 보고서

## 1. 이것은 무엇인가?

Claude API Server는 **Claude Max 구독에 포함된 Claude Code CLI(`claude -p`)를 OpenAI 호환 API, 커스텀 REST API, MCP(Model Context Protocol) 서버로 변환하는 게이트웨이**다. v2에서 OpenAI 호환 포맷(`/v1/chat/completions`)과 SSE 스트리밍을 추가했다.

핵심 아이디어는 간단하다:

| 방식 | 비용 구조 | 호출 방법 |
|------|----------|----------|
| `anthropic.messages.create()` (Python SDK) | **토큰당 과금** — input $3/M, output $15/M (Sonnet) | API Key |
| `claude -p "prompt"` (이 서버) | **월 정액제** — Max 구독에 포함, 추가 비용 없음 | OAuth (이미 인증됨) |

즉, fire-litigation-tool이나 n8n 같은 외부 시스템이 AI를 호출할 때 Anthropic API를 직접 호출하는 대신 이 서버를 거치면, **동일한 AI 기능을 추가 비용 없이** 사용할 수 있다.

## 2. 아키텍처

```
외부 클라이언트
├── OpenAI SDK / LangChain / Open WebUI / Continue.dev
│   └── POST /v1/chat/completions  (OpenAI 호환 포맷)
│   └── GET  /v1/models
├── curl / Python / n8n HTTP Request
│   └── POST /api/chat, /api/code, /api/analyze  (커스텀 포맷)
│   └── POST /api/tools/file-read, /api/tools/bash
└── MCP 클라이언트 (Claude Code, AI 에이전트)
    └── GET /mcp  (SSE)
        │
        ▼
┌─────────────────────────────────────────┐
│  FastAPI Server (port 8020)             │
│                                         │
│  Bearer Token 인증                      │
│  ├── OpenAI 호환 API (/v1/*)   [v2 신규]│
│  │   ├── 모델 별칭 해석                 │
│  │   │   (gpt-4o → opus → claude-opus)  │
│  │   ├── SSE 스트리밍 지원              │
│  │   └── 도구 호출(Tool Calling) 지원   │
│  ├── 커스텀 REST API (/api/*)           │
│  ├── MCP Protocol (/mcp)                │
│  └── Health Check (/health)             │
│                                         │
│  ClaudeService                          │
│  ├── asyncio.Semaphore(2)               │
│  ├── chat()  → subprocess json          │
│  ├── chat_stream() → subprocess stream  │
│  └── code() / analyze()                 │
└────────────┬────────────────────────────┘
             │ async subprocess
             ▼
┌─────────────────────────────────────────┐
│  Claude Code CLI                        │
│  claude -p "prompt"                     │
│  --output-format json|stream-json       │
│  --model <opus|sonnet|haiku>            │
│  (Max 구독 — 정액제)                    │
└─────────────────────────────────────────┘
```

**핵심 구성 요소**:
- **ClaudeService** — Claude CLI를 async subprocess로 호출하는 싱글톤 서비스. `chat()`(일반), `chat_stream()`(스트리밍), `code()`, `analyze()` 메서드 제공. Semaphore(2)로 동시 요청 2개 제한
- **OpenAI 호환 API** — `/v1/chat/completions`(스트리밍, 도구 호출 포함), `/v1/models`. 모델 별칭 자동 해석 (gpt-4o → opus 등). `tools` 파라미터로 도구 정의를 전달하면 OpenAI 호환 tool_calls 포맷으로 응답
- **커스텀 REST API** — 7개 엔드포인트 (chat, code, analyze, file-read, bash, models, health)
- **MCP Endpoint** — `fastapi-mcp` 라이브러리가 REST API를 자동으로 MCP 도구로 변환
- **Bearer Auth** — `.env`의 토큰으로 모든 API 엔드포인트 보호

## 3. 사용 가능한 모델

| 단축키 | 모델 ID | 특성 |
|--------|---------|------|
| `haiku` | claude-haiku-4-5-20251001 | 빠름, 간단한 작업용 |
| `sonnet` (기본값) | claude-sonnet-4-5-20250929 | 균형잡힌 성능 |
| `opus` | claude-opus-4-6 | 최고 품질, 복잡한 추론 |

### OpenAI 모델 별칭 (v2 신규)

| OpenAI 모델명 | → 내부 키 | → Claude 모델 ID |
|---------------|----------|------------------|
| `gpt-4o`, `gpt-4`, `gpt-4-turbo` | opus | claude-opus-4-6 |
| `gpt-4o-mini` | sonnet | claude-sonnet-4-5-20250929 |
| `gpt-3.5-turbo` | haiku | claude-haiku-4-5-20251001 |

OpenAI SDK나 호환 도구에서 `gpt-4o`를 보내면 자동으로 `claude-opus-4-6`으로 변환된다. `resolve_model()` 메서드가 이를 처리한다.

## 4. 엔드포인트별 상세 (총 11개)

### 4.1 `POST /api/chat` — 범용 AI 채팅

가장 기본적인 엔드포인트. 프롬프트를 보내면 AI 응답을 받는다.

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "대한민국 민법 제750조를 설명해줘",
    "model": "sonnet",
    "system_prompt": "당신은 한국 법률 전문가입니다.",
    "max_budget_usd": 0.5
  }'
```

### 4.2 `POST /api/code` — 코드 생성/분석 (도구 접근 포함)

Claude Code의 내장 도구(Read, Glob, Grep, Bash)를 사용할 수 있어, 서버의 실제 파일을 읽고 명령을 실행하며 코드를 분석할 수 있다.

```bash
curl -X POST http://localhost:8020/api/code \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "nursing-home-panel 프로젝트의 라우터 구조를 분석해줘",
    "model": "sonnet",
    "allowed_tools": ["Read", "Glob", "Grep"]
  }'
```

### 4.3 `POST /api/analyze` — 문서/데이터 분석

서버의 파일을 컨텍스트로 첨부하여 분석을 요청한다. 파일 내용이 프롬프트에 자동 삽입된다.

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "이 두 파일의 차이점을 분석해줘",
    "context_files": [
      "/home/gslee/nursing-home-panel/app/routers/api.py",
      "/home/gslee/fire-litigation-tool/app/routers/api.py"
    ],
    "model": "sonnet"
  }'
```

### 4.4 `POST /api/tools/file-read` — 서버 파일 읽기

허용된 경로(`/home/gslee/` 하위)의 파일만 읽을 수 있다.

```bash
curl -X POST http://localhost:8020/api/tools/file-read \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/gslee/nursing-home-panel/requirements.txt"}'
```

### 4.5 `POST /api/tools/bash` — 화이트리스트 셸 명령

`ls`, `cat`, `grep`, `find`, `df`, `du`, `free`, `date`, `uptime`, `wc`, `head`, `tail`, `echo`, `file`, `stat`, `python3 -c`, `python3 -m json.tool`만 허용된다. `rm`, `mv`, `curl` 등 위험한 명령은 차단.

```bash
curl -X POST http://localhost:8020/api/tools/bash \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"command": "df -h"}'
```

### 4.6 `GET /api/models` — 모델 목록

### 4.7 `GET /health` — 헬스 체크 (인증 불필요)

### 4.8 `GET /mcp` — MCP Protocol (SSE)

MCP 클라이언트(Claude Code, 다른 AI 에이전트 등)가 이 서버를 도구 서버로 연결할 수 있다.

```bash
claude mcp add claude-gateway http://localhost:8020/mcp
```

### 4.9 `POST /v1/chat/completions` — OpenAI 호환 채팅 (v2 신규)

OpenAI ChatCompletion 포맷과 완전 호환. 스트리밍(`stream: true`)과 도구 호출(`tools`) 지원.

**추가 파라미터**:
- `tools` — OpenAI 도구 정의 배열 (`[{"type": "function", "function": {"name": "...", ...}}]`)
- `tool_choice` — 도구 선택 제어 (인식만 됨, 내부적으로는 Claude가 판단)

```bash
# 일반 응답
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "한국어로 답변하세요."},
      {"role": "user", "content": "FastAPI의 장점을 알려줘"}
    ]
  }'
```

```bash
# 스트리밍 응답 (SSE)
curl -N -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "haiku", "messages": [{"role": "user", "content": "안녕!"}], "stream": true}'
```

```bash
# 도구 호출 (Tool Calling)
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [{"role": "user", "content": "배터리 상태를 확인해줘"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "mcp",
        "description": "MCP 서버의 도구를 호출합니다",
        "parameters": {
          "type": "object",
          "properties": {
            "server": {"type": "string", "description": "MCP 서버명"},
            "tool": {"type": "string", "description": "도구명"}
          },
          "required": ["server", "tool"]
        }
      }
    }]
  }'
```

응답 (도구 호출 시):
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "claude-sonnet-4-5-20250929",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_a1b2c3d4e5f6",
        "type": "function",
        "function": {
          "name": "mcp",
          "arguments": "{\"server\": \"aster\", \"tool\": \"get_battery\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 120, "completion_tokens": 35, "total_tokens": 155}
}
```

**도구 호출 아키텍처**:
```
도구 정의 (tools 파라미터)
  │  시스템 프롬프트에 도구 설명 주입
  ▼
Claude CLI (내부적으로 항상 non-streaming)
  │  Claude 응답에서 JSON 코드 블록 파싱
  │  ├── {"tool_calls": [{"name": "...", "arguments": {...}}]}
  ▼
도구명 정규화
  │  ├── mcp__aster__get_battery → mcp(server=aster, tool=get_battery)
  │  ├── Read → read, Bash → exec 등
  ▼
OpenAI tool_calls 포맷으로 변환
  │  ├── finish_reason: "tool_calls"
  │  └── message.tool_calls: [{id, type, function: {name, arguments}}]
```

**MCP 도구명 정규화**: Claude Code가 `mcp__server__tool` 형식으로 도구를 호출할 수 있다. 이 서버는 이를 자동으로 `mcp` 도구 + `{server, tool}` 인자로 변환한다.

| 원본 (Claude 내부) | 변환 결과 |
|---------------------|----------|
| `mcp__aster__get_battery` | `mcp` + `{"server": "aster", "tool": "get_battery"}` |
| `Read` | `read` |
| `Bash` | `exec` |
| `WebFetch` | `web_fetch` |

**스트리밍 아키텍처** (일반 텍스트 응답):
```
Claude CLI (--output-format stream-json --verbose)
  │  newline-delimited JSON
  │  ├── {"type":"system",...}       ← 무시
  │  ├── {"type":"assistant",...}    ← 텍스트 추출 → delta 계산
  │  └── {"type":"result",...}       ← usage 추출 → 종료
  ▼
SSE 변환
  │  ├── data: {"choices":[{"delta":{"content":"..."}}]}
  │  ├── data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
  │  └── data: [DONE]
```

### 4.10 `GET /v1/models` — OpenAI 호환 모델 목록 (v2 신규)

OpenAI SDK의 `client.models.list()`와 호환.

```bash
curl http://localhost:8020/v1/models -H "Authorization: Bearer <token>"
```

## 5. 이걸로 할 수 있는 것들

### A. 기존 프로젝트의 Anthropic API 비용 제거

| 프로젝트 | 현재 | 전환 후 |
|---------|------|---------|
| fire-litigation-tool | `anthropic.messages.create()` — 토큰당 과금 | `POST /api/chat` — 무료 |
| nursing-home-panel (모바일 AI 챗) | Claude Haiku API — 토큰당 과금 | `POST /api/chat` + model=haiku — 무료 |

**v2 이점**: OpenAI 호환 포맷 지원으로 기존 OpenAI SDK 코드를 `base_url`만 바꾸면 바로 전환 가능.

```python
# 기존 OpenAI 코드
client = OpenAI(api_key="sk-...")

# → 이 서버로 전환 (코드 1줄만 수정)
client = OpenAI(base_url="http://localhost:8020/v1", api_key="<bearer-token>")
```

### B. n8n / 자동화 워크플로우에서 AI 호출

n8n에서 OpenAI credentials 노드에 Base URL `http://localhost:8020/v1`을 설정하면 OpenAI 노드를 그대로 사용할 수 있다. 또는 HTTP Request 노드로 직접 호출도 가능:
- 이메일 자동 분류 및 응답 초안
- 문서 요약 자동화
- 데이터 추출/변환 파이프라인
- 알림 내용 자동 생성

### C. 문서 분석 및 요약

`/api/analyze`에 서버의 파일 경로를 전달하여:
- 법률 문서 분석 (fire-litigation-tool 연동)
- 코드 리뷰 자동화
- 로그 파일 분석
- 설정 파일 비교
- CSV/JSON 데이터 해석

### D. 코드 생성 및 리팩토링

`/api/code`는 Claude Code의 도구 접근 기능을 그대로 활용:
- 코드 리뷰 요청 ("이 파일의 보안 취약점을 찾아줘")
- 코드 생성 ("FastAPI 라우터를 만들어줘")
- 코드베이스 탐색 ("이 프로젝트에서 인증이 어떻게 구현되어 있는지 설명해줘")
- 버그 분석 ("이 에러 로그를 분석하고 원인을 찾아줘")

### E. AI 기반 모니터링 & 알림

cron이나 스크립트에서 주기적으로 호출:
```bash
# 서버 상태를 AI에게 분석 시키기
curl -X POST http://localhost:8020/api/chat \
  -d '{"prompt": "다음 서버 상태를 분석하고 문제가 있으면 알려줘: '"$(df -h | tr '\n' ' ')"'", "model": "haiku"}'
```

### F. 다국어 번역

```bash
curl -X POST http://localhost:8020/api/chat \
  -d '{"prompt": "Translate to English: 요양원 입소자 일일 기록", "model": "haiku"}'
```

### G. MCP 클라이언트 연동

이 서버를 MCP 도구 서버로 등록하면, 다른 Claude Code 인스턴스나 MCP 호환 클라이언트가 이 서버의 모든 기능을 도구로 사용할 수 있다. 예를 들어:
- 다른 머신의 Claude Code에서 이 서버의 AI를 호출
- MCP 호환 에이전트 프레임워크와 연동
- 여러 AI 도구를 체인으로 연결

### H. 커스텀 system_prompt로 전문 AI 구축

같은 엔드포인트를 다양한 system_prompt로 호출하여 전문 AI를 즉석 생성:
- **법률 AI**: `system_prompt: "당신은 대한민국 화재 소송 전문 변호사입니다"`
- **의료 AI**: `system_prompt: "당신은 요양원 간호 기록 전문가입니다"`
- **코딩 AI**: `system_prompt: "당신은 FastAPI+SQLAlchemy 전문가입니다"`
- **번역 AI**: `system_prompt: "당신은 한영 의료 번역 전문가입니다"`

### I. 서버 파일 조회 API

`/api/tools/file-read`로 원격에서 서버 파일을 읽을 수 있다:
- 설정 파일 확인
- 로그 파일 조회
- 데이터 파일 미리보기

### J. 경량 서버 상태 조회

`/api/tools/bash`로 안전한 명령어만 실행:
- `df -h` — 디스크 사용량
- `free -m` — 메모리 사용량
- `uptime` — 서버 가동 시간
- `find /path -name "*.log" -mtime -1` — 최근 로그 파일 찾기

## 6. 보안 구조

| 계층 | 보호 방법 |
|------|----------|
| 인증 | Bearer Token (`.env`에 설정) |
| 파일 접근 | 허용 경로 화이트리스트 (`/home/gslee/` 하위만) |
| 명령 실행 | 명령어 화이트리스트 (18개 안전한 접두사만) |
| 동시성 | Semaphore(2) — 최대 2개 동시 처리 |
| 타임아웃 | 요청당 120초, bash 명령 30초 |
| 네트워크 | `127.0.0.1`만 바인딩 (Caddy 통해서만 외부 노출) |

## 7. Anthropic API 대비 비교

| 항목 | Anthropic API | 이 서버 |
|------|--------------|---------|
| 비용 | $3-$15/M tokens | Max 구독에 포함 (추가 $0) |
| Opus 4.6 | API 미제공 | 사용 가능 |
| 내장 도구 | tool_use 직접 구현 | Read, Bash, Glob, Grep 내장 |
| 도구 호출 | tool_use 네이티브 지원 | OpenAI 호환 tool_calls 포맷 지원 (시스템 프롬프트 주입 방식) |
| Rate Limit | 티어별 제한 | Max 20x 상향 |
| 설정 | API Key 발급 필요 | 이미 인증됨 |
| 지연 시간 | ~1-3초 | ~4-6초 (subprocess 오버헤드) |

## 8. 제약사항

1. **시작 지연** — 각 요청마다 Claude CLI 프로세스 시작에 ~2-5초 소요
2. **Stateless** — 대화 이력 없음 (`--no-session-persistence`). 매 요청이 독립적
3. **동시성 제한** — Semaphore(2)로 최대 2개 동시 처리. 3번째부터 대기
4. **Max 구독 한도** — 무제한은 아님 (20x 상향). 대량 사용 시 모니터링 필요
5. **프롬프트 크기** — 최대 100,000자 (API 제한과 별개로 서버 측 제한)

## 9. 서버 관리

```bash
# 시작
cd /home/gslee/claude-api-server && source venv/bin/activate
nohup uvicorn app.main:app --host 127.0.0.1 --port 8020 &

# 중지
pkill -f "uvicorn.*8020"

# 로그
tail -f /tmp/claude-api-server.log

# 상태
curl http://localhost:8020/health
```

---

**파일 위치**: `/home/gslee/claude-api-server/`
**포트**: 8020
**핵심 파일**: 6개 (main.py, config.py, auth.py, api.py, claude_service.py, .env)
**테스트**: `test_server.py` — 6개 카테고리, 20+ 자동 테스트 (`python3 test_server.py --skip-ai`로 빠른 검증)
**영문 매뉴얼**: `MANUAL.md` — 초보자용 영문 사용 가이드
**버전**: v2.0 (OpenAI 호환 + 스트리밍)
