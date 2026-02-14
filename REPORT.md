# Claude API Server — 종합 보고서

## 이것은 무엇인가

Claude API Server는 Claude Code CLI(`claude -p`)를 HTTP API 서버로 감싸는 게이트웨이다. Claude Max 구독(월 정액제)으로 이미 인증된 CLI를 활용하므로, Anthropic API의 토큰당 과금 없이 어떤 외부 시스템에서든 Claude AI를 호출할 수 있다.

**한 줄 요약**: Anthropic API 대신 Max 구독 CLI를 쓰는 무료 AI API 서버 — OpenAI 호환 포맷(`/v1/chat/completions`) + 스트리밍 지원.

### 비용 비교

| 방식 | 1,000회 호출 (평균 1K 입력 + 500 출력 토큰) | 월간 예상 |
|------|-------------------------------------------|----------|
| Anthropic API (Sonnet) | ~$10.50 | 하루 100회 × 30일 = ~$315 |
| Anthropic API (Opus) | ~$37.50 | 하루 100회 × 30일 = ~$1,125 |
| **이 서버 (Max 구독)** | **$0 추가** | **$0 추가** (Max 구독료에 포함) |

### 동작 원리

```
외부 시스템 (n8n, LangChain, Open WebUI, Continue.dev, curl, Python 등)
    │
    │  OpenAI 호환: POST /v1/chat/completions  {"model":"gpt-4o","messages":[...]}
    │  커스텀 API:  POST /api/chat              {"prompt":"..."}
    │  MCP 프로토콜: GET /mcp
    ▼
FastAPI 서버 (port 8020)
    │
    │  모델 별칭 해석: gpt-4o → opus → claude-opus-4-6
    │  async subprocess
    ▼
claude -p "..." --output-format json|stream-json --model <model>
    │
    │  Max 구독 사용 (정액제)
    ▼
JSON 응답 반환 (일반) 또는 SSE 스트리밍 (stream: true)
```

각 API 요청마다 `claude -p` 프로세스를 비동기로 실행하고, JSON 형식의 응답을 파싱해서 반환한다. `--no-session-persistence`로 모든 요청이 독립적(stateless)이다.

---

## 서버 정보

| 항목 | 값 |
|------|-----|
| 위치 | `/home/gslee/claude-api-server/` |
| 포트 | 8020 |
| 프레임워크 | FastAPI + uvicorn |
| 인증 | Bearer Token |
| 프로토콜 | REST API + OpenAI 호환 API + MCP (SSE) |
| 동시 요청 | 최대 2개 (Semaphore) |
| 타임아웃 | 120초 (AI 요청), 30초 (bash 명령) |

### 파일 구조

```
claude-api-server/
├── app/
│   ├── main.py              ← FastAPI 앱, 라우터 등록, MCP 마운트
│   ├── config.py            ← 설정 (pydantic-settings, .env 로딩)
│   ├── auth.py              ← Bearer Token 인증
│   ├── routers/
│   │   └── api.py           ← REST API 7개 + OpenAI 호환 2개 엔드포인트
│   └── services/
│       └── claude_service.py ← Claude CLI 관리 (일반 + 스트리밍)
├── .env                     ← 토큰, 설정값
├── REPORT.md                ← 종합 보고서 (이 파일)
├── ARCHITECTURE.md          ← 기술 아키텍처
├── OPENAI_ENDPOINTS.md      ← OpenAI 호환 API 상세 문서
├── test_server.py           ← 자동 테스트 (6개 카테고리, 20+ 테스트)
├── MANUAL.md                ← 영문 사용 매뉴얼 (초보자용)
├── requirements.txt         ← 의존성
└── venv/                    ← Python 가상환경
```

---

## 서버 시작/중지

```bash
# 시작
cd /home/gslee/claude-api-server && source venv/bin/activate
nohup uvicorn app.main:app --host 127.0.0.1 --port 8020 > /tmp/claude-api-server.log 2>&1 &

# 상태 확인
curl http://localhost:8020/health

# 로그 확인
tail -f /tmp/claude-api-server.log

# 중지
pkill -f "uvicorn.*8020"
```

---

## API 엔드포인트

모든 API 호출에는 다음 헤더가 필요하다 (`/health` 제외):

```
Authorization: Bearer <API_SECRET_TOKEN>
Content-Type: application/json
```

토큰은 `.env` 파일의 `API_SECRET_TOKEN` 값이다.

---

### 1. `GET /health` — 헬스 체크

인증 불필요. 서버 상태 확인용.

```bash
curl http://localhost:8020/health
```

```json
{"status": "ok", "default_model": "sonnet", "max_concurrent": 2}
```

---

### 2. `POST /api/chat` — AI 채팅

가장 범용적인 엔드포인트. 프롬프트를 보내면 AI 응답을 받는다.

**파라미터**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `prompt` | string | O | AI에게 보낼 메시지 (최대 100,000자) |
| `model` | string | X | `haiku`, `sonnet`(기본값), `opus` |
| `system_prompt` | string | X | AI의 역할/성격 지정 |
| `max_budget_usd` | float | X | 요청당 최대 비용 제한 |

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "파이썬에서 비동기 프로그래밍이 뭔지 3줄로 설명해줘",
    "model": "haiku"
  }'
```

```json
{
  "result": "비동기 프로그래밍은 ...",
  "model": "claude-haiku-4-5-20251001",
  "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
  "duration_ms": 4523,
  "is_error": false
}
```

---

### 3. `POST /api/code` — 코드 생성/분석

Claude Code의 내장 도구(Read, Glob, Grep, Bash)에 접근할 수 있어, 서버의 실제 파일시스템을 탐색하고 명령을 실행하면서 코드를 분석하거나 생성한다.

**파라미터**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `prompt` | string | O | 코드 관련 요청 |
| `model` | string | X | 모델 선택 |
| `allowed_tools` | string[] | X | 허용 도구 (기본: Read, Glob, Grep, Bash) |
| `max_budget_usd` | float | X | 비용 제한 |

```bash
curl -X POST http://localhost:8020/api/code \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "/home/gslee/nursing-home-panel/app/routers/ 디렉토리의 라우터 파일 목록과 각 파일의 엔드포인트 개수를 알려줘",
    "model": "sonnet"
  }'
```

---

### 4. `POST /api/analyze` — 문서/데이터 분석

서버의 파일을 컨텍스트로 첨부하여 분석을 요청한다. 파일 내용이 프롬프트에 자동 삽입된다.

**파라미터**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `prompt` | string | O | 분석 요청 |
| `context_files` | string[] | X | 서버의 파일 경로 목록 (허용 경로만) |
| `model` | string | X | 모델 선택 |
| `max_budget_usd` | float | X | 비용 제한 |

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "이 SQL 마이그레이션 스크립트를 분석하고 잠재적 문제를 찾아줘",
    "context_files": ["/home/gslee/nursing-home-panel/scripts/migration_care_v2.sql"],
    "model": "sonnet"
  }'
```

---

### 5. `POST /api/tools/file-read` — 서버 파일 읽기

허용된 경로(`/home/gslee/` 하위)의 파일만 읽을 수 있다.

```bash
curl -X POST http://localhost:8020/api/tools/file-read \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/gslee/nursing-home-panel/requirements.txt"}'
```

```json
{"path": "...", "content": "fastapi>=0.115...", "size": 342}
```

---

### 6. `POST /api/tools/bash` — 셸 명령 실행 (화이트리스트)

안전한 명령어만 실행 가능. 30초 타임아웃.

**허용되는 명령어**:
`ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `date`, `uptime`, `df`, `du`, `free`, `whoami`, `pwd`, `echo`, `file`, `stat`, `python3 -c`, `python3 -m json.tool`

```bash
curl -X POST http://localhost:8020/api/tools/bash \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command": "df -h"}'
```

```json
{"stdout": "Filesystem   Size  Used  Avail ...", "stderr": "", "returncode": 0}
```

차단 예시: `rm`, `mv`, `cp`, `curl`, `wget`, `apt`, `pip`, `kill` 등은 403 Forbidden.

---

### 7. `GET /api/models` — 사용 가능 모델 목록

```bash
curl http://localhost:8020/api/models -H "Authorization: Bearer $TOKEN"
```

```json
{
  "models": [
    {"key": "opus",   "model_id": "claude-opus-4-6",           "default": false},
    {"key": "sonnet", "model_id": "claude-sonnet-4-5-20250929", "default": true},
    {"key": "haiku",  "model_id": "claude-haiku-4-5-20251001",  "default": false}
  ]
}
```

---

### 8. `POST /v1/chat/completions` — OpenAI 호환 채팅 (v2 신규)

OpenAI SDK, LangChain, n8n, Open WebUI 등 OpenAI 포맷을 사용하는 모든 도구와 호환된다. 스트리밍도 지원한다.

**파라미터**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `model` | string | X | 모델명 (아래 별칭표 참고, 기본값: `sonnet`) |
| `messages` | array | O | `[{"role": "user", "content": "..."}]` 형식 |
| `stream` | bool | X | `true`면 SSE 스트리밍 응답 |
| `temperature` | float | X | (인식만 됨, Claude CLI에 전달되지 않음) |
| `max_tokens` | int | X | (인식만 됨, Claude CLI에 전달되지 않음) |
| `tools` | list[object] | X | OpenAI 도구 정의 배열 (`[{"type":"function","function":{...}}]`) |
| `tool_choice` | any | X | 도구 선택 제어 (인식만 됨, Claude가 자체 판단) |

**모델 별칭**:

| OpenAI 모델명 | 매핑되는 Claude 모델 |
|---------------|---------------------|
| `gpt-4o`, `gpt-4`, `gpt-4-turbo` | claude-opus-4-6 |
| `gpt-4o-mini` | claude-sonnet-4-5-20250929 |
| `gpt-3.5-turbo` | claude-haiku-4-5-20251001 |

물론 `opus`, `sonnet`, `haiku` 또는 전체 Claude 모델 ID도 사용 가능하다.

**일반 응답**:
```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "한국어로 답변하세요."},
      {"role": "user", "content": "비동기 프로그래밍이 뭔지 3줄로 설명해줘"}
    ]
  }'
```

```json
{
  "id": "chatcmpl-2f68fde3578c",
  "object": "chat.completion",
  "created": 1770606602,
  "model": "claude-sonnet-4-5-20250929",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 15, "completion_tokens": 42, "total_tokens": 57}
}
```

**스트리밍 응답** (`stream: true`):
```bash
curl -N -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "haiku", "messages": [{"role": "user", "content": "안녕!"}], "stream": true}'
```

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"안녕"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"하세요!"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

**도구 호출 (Tool Calling)**:

```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
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
            "server": {"type": "string"},
            "tool": {"type": "string"}
          },
          "required": ["server", "tool"]
        }
      }
    }]
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1770606602,
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

**Multi-turn 도구 대화** — 도구 결과를 `tool` role로 전달:

```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [
      {"role": "user", "content": "배터리 상태를 확인해줘"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_a1b2c3d4e5f6", "type": "function", "function": {"name": "mcp", "arguments": "{\"server\":\"aster\",\"tool\":\"get_battery\"}"}}]},
      {"role": "tool", "tool_call_id": "call_a1b2c3d4e5f6", "name": "mcp", "content": "{\"level\": 85, \"status\": \"charging\"}"}
    ],
    "tools": [...]
  }'
```

응답: Claude가 도구 결과를 요약하여 `"현재 배터리는 85%이며 충전 중입니다."` 같은 자연어로 응답한다.

---

### 9. `GET /v1/models` — OpenAI 호환 모델 목록 (v2 신규)

OpenAI SDK의 `client.models.list()` 호환.

```bash
curl http://localhost:8020/v1/models -H "Authorization: Bearer $TOKEN"
```

```json
{
  "object": "list",
  "data": [
    {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
    {"id": "gpt-4o", "object": "model", "owned_by": "anthropic", "parent": "claude-opus-4-6"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "anthropic", "parent": "claude-sonnet-4-5-20250929"},
    {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "anthropic", "parent": "claude-haiku-4-5-20251001"}
  ]
}
```

---

### 10. `GET /mcp` — MCP Protocol 엔드포인트

MCP(Model Context Protocol) 클라이언트가 SSE로 연결하는 엔드포인트. 위의 REST API들이 자동으로 MCP 도구로 변환되어 노출된다.

```bash
# Claude Code에서 MCP 서버로 등록
claude mcp add claude-gateway http://localhost:8020/mcp

# 등록 확인
claude mcp list
```

---

## 활용 예시 모음

### A. 텍스트 요약

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "다음 텍스트를 3줄로 요약해줘:\n\n장기요양보험제도는 고령이나 노인성 질병 등의 사유로 일상생활을 혼자서 수행하기 어려운 노인 등에게 제공하는 신체활동 또는 가사활동 지원 등의 장기요양급여에 관한 사항을 규정함으로써...",
    "model": "haiku"
  }'
```

### B. 번역

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to English:\n입소자 김OO님의 오늘 일일기록: 식사량 양호, 배변 정상, 수면 6시간. 오전 물리치료 참여.",
    "model": "haiku"
  }'
```

### C. 전문 역할 AI — 법률 분석

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "전기화재로 인한 건물 손해배상 소송에서 원고의 입증책임 범위와 관련 판례를 설명해줘",
    "system_prompt": "당신은 대한민국 화재 소송 전문 변호사입니다. 관련 법률과 판례를 근거로 답변하세요.",
    "model": "opus"
  }'
```

### D. 전문 역할 AI — 의료/간호

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "욕창 Braden Scale 점수가 12점인 입소자에 대한 간호 중재 계획을 수립해줘",
    "system_prompt": "당신은 노인요양시설 전문 간호사입니다. 장기요양기관 평가 기준에 맞게 답변하세요.",
    "model": "sonnet"
  }'
```

### E. 코드 리뷰

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "이 파일의 보안 취약점과 개선 사항을 분석해줘",
    "context_files": ["/home/gslee/fire-litigation-tool/app/routers/api.py"],
    "model": "opus"
  }'
```

### F. 데이터 추출/변환

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "다음 텍스트에서 이름, 날짜, 금액을 JSON으로 추출해줘:\n\n2026년 1월 15일 김철수님에게 요양급여 350,000원이 청구되었습니다. 보호자 이영희님이 확인하였습니다.",
    "model": "haiku"
  }'
```

### G. SQL 쿼리 생성

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "MariaDB에서 r_daily_care 테이블과 r_residents 테이블을 조인해서, 최근 7일간 기록이 없는 입소자 목록을 조회하는 쿼리를 작성해줘. r_daily_care에는 resident_id, care_date, meal_amount, sleep_hours 컬럼이 있고, r_residents에는 resident_id, name, status 컬럼이 있어.",
    "model": "sonnet"
  }'
```

### H. 서버 상태 모니터링 (cron 연동)

```bash
#!/bin/bash
# /home/gslee/scripts/daily_check.sh
STATUS=$(df -h && echo "---" && free -m && echo "---" && uptime)

curl -s -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"서버 상태를 분석하고 문제가 있으면 경고해줘:\n\n$STATUS\",
    \"model\": \"haiku\"
  }"
```

### I. 파일 비교 분석

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "이 두 설정 파일의 차이점을 분석하고 잠재적 문제를 알려줘",
    "context_files": [
      "/home/gslee/nursing-home-panel/.env",
      "/home/gslee/fire-litigation-tool/.env"
    ],
    "model": "haiku"
  }'
```

### J. Python 스크립트에서 호출

```python
import requests

API = "http://localhost:8020"
TOKEN = "<API_SECRET_TOKEN>"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# 간단한 채팅
resp = requests.post(f"{API}/api/chat", headers=HEADERS, json={
    "prompt": "1+1은?",
    "model": "haiku"
})
print(resp.json()["result"])

# 파일 분석
resp = requests.post(f"{API}/api/analyze", headers=HEADERS, json={
    "prompt": "이 코드를 리뷰해줘",
    "context_files": ["/home/gslee/nursing-home-panel/app/main.py"],
    "model": "sonnet"
})
print(resp.json()["result"])
```

### K. n8n 워크플로우 연동

**방법 1: OpenAI 호환 모드 (권장)** — n8n의 OpenAI 노드 또는 "AI Agent" 노드에서:
- **Credentials**: OpenAI API type, Base URL `http://localhost:8020/v1`, API Key에 Bearer 토큰 입력
- 별도 코드 없이 바로 연동 가능

**방법 2: HTTP Request 노드**:
- **URL**: `http://localhost:8020/v1/chat/completions`
- **Headers**: `Authorization: Bearer <token>`
- **Body**: `{"model": "haiku", "messages": [{"role": "user", "content": "{{$json.email_body}}를 요약해줘"}]}`

활용 예:
- 수신 이메일 자동 분류 → AI 요약 → Slack 알림
- 입소자 기록 텍스트 → AI 분석 → 위험 신호 감지
- 문서 업로드 → AI 요약 → 데이터베이스 저장

### K-2. OpenAI SDK로 호출 (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>"
)

# 일반 호출
resp = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "안녕하세요!"}]
)
print(resp.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="haiku",
    messages=[{"role": "user", "content": "1부터 5까지 세어줘"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### L. MCP 클라이언트로 다른 Claude Code에서 연결

```bash
# 다른 Claude Code 인스턴스에서
claude mcp add my-ai-gateway http://localhost:8020/mcp

# 이제 Claude Code가 이 서버를 도구로 사용 가능
# "my-ai-gateway의 chat 도구를 써서 ..."
```

### M. AI 에이전트 — 도구 호출 (Tool Calling)

OpenAI 호환 Tool Calling을 활용하면 AI 에이전트가 외부 도구를 자율적으로 호출할 수 있다.

**n8n AI Agent에서 MCP 도구 활용**:

n8n의 AI Agent 노드에 OpenAI credentials (Base URL: `http://localhost:8020/v1`)를 설정하고, n8n의 도구 노드(HTTP Request, Code 등)를 연결하면:
1. 사용자가 "서버 디스크 사용량 확인해줘" 요청
2. AI Agent가 `tool_calls`로 도구 호출 결정
3. n8n이 도구를 실행하고 결과를 `tool` role로 전달
4. AI가 결과를 자연어로 요약하여 응답

**OpenAI SDK로 도구 호출 (Python)**:

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>"
)

# 도구 정의
tools = [{
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

# 1단계: AI에게 요청 → 도구 호출 반환
messages = [{"role": "user", "content": "배터리 상태를 확인해줘"}]
resp = client.chat.completions.create(model="sonnet", messages=messages, tools=tools)

if resp.choices[0].finish_reason == "tool_calls":
    # 2단계: 도구 실행 (실제로는 MCP 호출 등)
    tc = resp.choices[0].message.tool_calls[0]
    tool_result = json.dumps({"level": 85, "status": "charging"})

    # 3단계: 도구 결과를 전달하여 최종 응답 받기
    messages.append(resp.choices[0].message.model_dump())
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.function.name,
        "content": tool_result,
    })
    final = client.chat.completions.create(model="sonnet", messages=messages, tools=tools)
    print(final.choices[0].message.content)
    # → "현재 배터리는 85%이며 충전 중입니다."
```

---

## 보안 구조

| 계층 | 보호 방법 | 상세 |
|------|----------|------|
| 네트워크 | localhost만 바인딩 | `127.0.0.1:8020` — 외부 직접 접근 불가 |
| 인증 | Bearer Token | `.env`의 `API_SECRET_TOKEN` — 없거나 틀리면 401 |
| 파일 접근 | 경로 화이트리스트 | `/home/gslee/` 하위만 허용, 그 외 403 |
| 명령 실행 | 명령어 화이트리스트 | 18개 안전한 접두사만 허용, 그 외 403 |
| 동시성 | Semaphore(2) | 최대 2개 동시 처리 — 서버 과부하 방지 |
| 타임아웃 | 요청별 제한 | AI 120초, bash 30초 — 무한 대기 방지 |

외부에 노출하려면 Caddy 리버스 프록시(예: `mcp.mmh.li → 127.0.0.1:8020`)를 추가한다.

---

## 제약사항

| 항목 | 설명 |
|------|------|
| 시작 지연 | 매 요청마다 CLI 프로세스 시작에 ~3-6초 소요 |
| Stateless | 대화 이력 없음. 매 요청이 독립적 |
| 동시성 | 최대 2개 동시 처리. 3번째 요청부터 대기 |
| Max 구독 한도 | 무제한 아님 (일반 대비 20x 상향). 대량 사용 시 주의 |
| 프롬프트 크기 | 최대 100,000자 |
| 출력 형식 | 텍스트만 (이미지 생성 불가) |

---

## 설정 변경 (.env)

```env
API_SECRET_TOKEN=<토큰>          # Bearer 인증 토큰
CLAUDE_CLI_PATH=/home/gslee/.local/bin/claude  # CLI 경로
DEFAULT_MODEL=sonnet              # 기본 모델 (haiku/sonnet/opus)
MAX_CONCURRENT=2                  # 동시 요청 수
REQUEST_TIMEOUT=120               # AI 요청 타임아웃 (초)
ALLOWED_PATHS=/home/gslee         # 파일 접근 허용 경로 (쉼표 구분)
```

변경 후 서버 재시작 필요: `pkill -f "uvicorn.*8020"` → 다시 시작.

---

## 테스트

`test_server.py`로 서버의 모든 기능을 자동 검증할 수 있다. 표준 라이브러리만 사용하므로 추가 설치 없이 실행 가능.

```bash
# 전체 테스트 (AI 호출 포함, ~30초)
python3 test_server.py

# AI 호출 제외 빠른 테스트 (~2초)
python3 test_server.py --skip-ai

# 특정 카테고리만
python3 test_server.py --only security

# 응답 본문 포함 디버그 출력
python3 test_server.py --verbose
```

**6개 테스트 카테고리**:

| 카테고리 | 검증 내용 |
|---------|----------|
| `connectivity` | 서버 연결, 헬스 체크 |
| `auth` | 인증 토큰 검증, 미인증 요청 차단 |
| `api` | 커스텀 API 엔드포인트 (chat, code, analyze) |
| `security` | 경로 탐색 차단, 명령어 화이트리스트, 프롬프트 크기 제한 |
| `openai` | OpenAI 호환 API, 모델 별칭, 스트리밍 |
| `mcp` | MCP 엔드포인트 접근 |

원격 서버 테스트:
```bash
python3 test_server.py --url http://remote-host:8020 --token <토큰>
```

---

## 영문 매뉴얼

영문 사용자를 위한 초보자 친화적 매뉴얼은 `MANUAL.md`를 참고.

For English documentation, see `MANUAL.md`.
