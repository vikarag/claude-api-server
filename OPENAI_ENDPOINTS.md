# OpenAI 호환 API 엔드포인트

## Overview

The server now includes OpenAI-compatible API endpoints that allow you to use Claude models with OpenAI client libraries.

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "sonnet",  // or "opus", "haiku", "gpt-4o", "gpt-3.5-turbo", etc.
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,  // optional
  "max_tokens": 1000,  // optional
  "stream": false,     // set to true for SSE streaming
  "tools": [],         // optional — OpenAI tool definitions array
  "tool_choice": "auto" // optional — tool selection control
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1707523200,
  "model": "claude-sonnet-4-5-20250929",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

## Tool Calling

`/v1/chat/completions` 엔드포인트는 OpenAI 호환 도구 호출(Tool Calling)을 지원한다. `tools` 파라미터로 도구를 정의하면, Claude가 도구 호출이 필요하다고 판단할 때 `tool_calls`를 포함한 응답을 반환한다.

### Request (with tools)

```json
{
  "model": "sonnet",
  "messages": [{"role": "user", "content": "배터리 상태를 확인해줘"}],
  "tools": [
    {
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
    }
  ]
}
```

### Response (tool_calls)

도구 호출 시 `finish_reason`이 `"tool_calls"`로 반환된다:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1770606602,
  "model": "claude-sonnet-4-5-20250929",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_a1b2c3d4e5f6",
            "type": "function",
            "function": {
              "name": "mcp",
              "arguments": "{\"server\": \"aster\", \"tool\": \"get_battery\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {"prompt_tokens": 120, "completion_tokens": 35, "total_tokens": 155}
}
```

### Multi-turn Tool Conversation

도구 결과를 `tool` role 메시지로 전달하면 대화를 이어갈 수 있다:

```json
{
  "model": "sonnet",
  "messages": [
    {"role": "user", "content": "배터리 상태를 확인해줘"},
    {
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
    {
      "role": "tool",
      "tool_call_id": "call_a1b2c3d4e5f6",
      "name": "mcp",
      "content": "{\"level\": 85, \"status\": \"charging\"}"
    }
  ],
  "tools": [...]
}
```

Claude가 도구 결과를 자연스럽게 요약하여 사용자에게 응답한다:
```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "현재 배터리는 85%이며 충전 중입니다."},
    "finish_reason": "stop"
  }]
}
```

### MCP 도구명 정규화

Claude Code는 내부적으로 `mcp__server__tool` 형식의 도구명을 사용할 수 있다. 이 서버는 이를 자동으로 정규화한다:

| 원본 (Claude 내부) | 변환된 도구명 | 변환된 인자 |
|---------------------|-------------|------------|
| `mcp__aster__get_battery` | `mcp` | `{"server": "aster", "tool": "get_battery"}` |
| `mcp__aster__send_sms` | `mcp` | `{"server": "aster", "tool": "send_sms", "arguments": {...}}` |
| `Read` | `read` | 원본 유지 |
| `Bash` | `exec` | 원본 유지 |
| `WebFetch` | `web_fetch` | 원본 유지 |
| `WebSearch` | `web_search` | 원본 유지 |

### Tool Calling과 Streaming

도구 호출 시 `stream: true`를 설정하면 도구 호출도 SSE 청크로 스트리밍된다. 단, 내부적으로는 Claude 응답을 완전히 받은 후 파싱하여 청크로 분할 전송한다 (도구 호출 파싱의 정확성을 위해).

```
data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xxx","type":"function","function":{"name":"mcp","arguments":"{...}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{...}}

data: [DONE]
```

---

**Response (streaming):**
Server-sent events (SSE) format with `data: {...}` chunks, ending with `data: [DONE]`.

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1770606602,"model":"claude-sonnet-4-5-20250929","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1770606602,"model":"claude-sonnet-4-5-20250929","choices":[{"index":0,"delta":{"content":"! How can I help?"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1770606602,"model":"claude-sonnet-4-5-20250929","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":8,"total_tokens":23}}

data: [DONE]
```

### GET /v1/models

List available models in OpenAI format.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-opus-4-6",
      "object": "model",
      "created": 1700000000,
      "owned_by": "anthropic"
    },
    {
      "id": "gpt-4o",
      "object": "model",
      "created": 1700000000,
      "owned_by": "anthropic",
      "parent": "claude-opus-4-6"
    }
  ]
}
```

## Model Aliases

The following OpenAI model names are mapped to Claude models:

| OpenAI Model | Claude Model |
|--------------|--------------|
| `gpt-4o`, `gpt-4`, `gpt-4-turbo` | `claude-opus-4-6` |
| `gpt-4o-mini` | `claude-sonnet-4-5-20250929` |
| `gpt-3.5-turbo` | `claude-haiku-4-5-20251001` |

You can also use:
- Short names: `opus`, `sonnet`, `haiku`
- Full Claude model IDs: `claude-opus-4-6`, etc.

## Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>"
)

# Non-streaming
response = client.chat.completions.create(
    model="sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### cURL

```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer <API_SECRET_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Authentication

All endpoints require the same authentication as other API endpoints:
- Header: `Authorization: Bearer <your-api-key>`

## Integration Examples

### n8n

n8n의 OpenAI credentials에서:
- **API Key**: `.env`의 `API_SECRET_TOKEN` 값
- **Base URL**: `http://localhost:8020/v1`

이후 OpenAI Chat Model 노드나 AI Agent 노드에서 그대로 사용 가능.

### LangChain (Python)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>",
    model="sonnet",
)
response = llm.invoke("안녕하세요!")
print(response.content)
```

### Open WebUI

Settings → Connections에서:
- **API Base URL**: `http://localhost:8020/v1`
- **API Key**: `.env`의 `API_SECRET_TOKEN` 값

### Continue.dev (VS Code AI)

`~/.continue/config.yaml`에:
```yaml
models:
  - title: Claude (via Gateway)
    provider: openai
    model: sonnet
    apiBase: http://localhost:8020/v1
    apiKey: <API_SECRET_TOKEN>
```

### n8n AI Agent (Tool Calling 활용)

n8n의 AI Agent 노드에서 OpenAI credentials를 설정하면, 에이전트가 자동으로 도구를 호출한다:

1. **Credentials** → OpenAI API, Base URL: `http://localhost:8020/v1`, API Key: Bearer 토큰
2. **AI Agent 노드** → Chat Model에 위 credentials 연결
3. **Tools 노드** 연결 → HTTP Request, Code 등 n8n 도구를 AI Agent에 연결
4. 에이전트가 필요에 따라 `tool_calls`로 도구를 호출하고, n8n이 결과를 전달

### Python OpenAI SDK (Tool Calling)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>"
)

# 도구 정의
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "현재 날씨를 조회합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "도시명"}
            },
            "required": ["location"]
        }
    }
}]

# 도구 호출 요청
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "서울 날씨 알려줘"}],
    tools=tools
)

# 도구 호출 결과 확인
choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    for tc in choice.message.tool_calls:
        print(f"도구: {tc.function.name}, 인자: {tc.function.arguments}")
        # → 실제 도구 실행 후 결과를 tool role 메시지로 다시 전달
```

### LangChain Agent (Tool Calling)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def get_server_status() -> str:
    """서버 상태를 조회합니다"""
    import subprocess
    return subprocess.check_output(["uptime"]).decode()

llm = ChatOpenAI(
    base_url="http://localhost:8020/v1",
    api_key="<API_SECRET_TOKEN>",
    model="sonnet",
)

# LangChain이 자동으로 tools 파라미터를 활용
agent = create_tool_calling_agent(llm, [get_server_status], prompt=...)
executor = AgentExecutor(agent=agent, tools=[get_server_status])
result = executor.invoke({"input": "서버 상태를 확인해줘"})
```

## Implementation Details

- **Router**: Separate `openai_router` in `app/routers/api.py`
- **Schemas**: OpenAI-compatible Pydantic models (`OpenAIChatRequest`, `OpenAIToolDef`, `OpenAIFunctionDef` 등)
- **Message conversion**: System prompts extracted, user/assistant/tool messages combined. `tool` role 메시지는 `[Tool Result for ...]` 형식으로 변환
- **Streaming**: Full SSE support with proper chunk formatting
- **Tool calling**: `tools` 파라미터가 있으면 도구 설명을 시스템 프롬프트에 주입 (`_build_tools_prompt()`). 내부적으로 non-streaming으로 전체 응답을 받은 후 JSON 코드 블록을 파싱 (`_parse_tool_calls()`)하여 OpenAI `tool_calls` 포맷으로 변환. 도구명은 `_normalize_tool_call()`로 정규화 (MCP `mcp__server__tool` → `mcp` + `{server, tool}`, Claude 내장 도구 → OpenClaw 매핑)
- **Error handling**: HTTP 400 for invalid requests, 502 for Claude API errors
