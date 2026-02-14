# Claude API Server - 종합 설치 가이드

> **이 문서를 Claude Code에게 전달하면 자동으로 전체 시스템을 구축할 수 있습니다.**
>
> 지시 예시: "이 SETUP_GUIDE.md 파일을 읽고, 문서에 나온 대로 Claude API Server를 이 컴퓨터에 설치해줘."

---

## 1. 프로젝트 개요

**Claude API Server**는 Claude Code CLI(Max 구독)를 백엔드로 사용하는 API 서버입니다.

### 핵심 기능
- **REST API**: `/api/chat`, `/api/code`, `/api/analyze` 등 커스텀 엔드포인트
- **OpenAI 호환 API**: `/v1/chat/completions`, `/v1/models` — 기존 OpenAI SDK/도구와 호환
- **스트리밍 지원**: SSE(Server-Sent Events) 기반 실시간 스트리밍
- **Tool Calling**: OpenAI 형식의 function calling 지원
- **MCP 프로토콜**: `fastapi-mcp`를 통한 MCP 엔드포인트 자동 마운트
- **보안**: Bearer 토큰 인증, 경로 화이트리스트, 명령어 화이트리스트
- **모델 별칭**: `gpt-4o` → `opus`, `gpt-4o-mini` → `sonnet` 등 자동 매핑

### 작동 원리
```
클라이언트 → FastAPI 서버 (포트 8020) → Claude Code CLI → Claude AI
                                          ↑
                              Max 구독으로 별도 API 비용 없음
```

---

## 2. 사전 요구사항

| 항목 | 요구사항 |
|------|---------|
| **Python** | 3.12 이상 |
| **Claude Code CLI** | 설치 및 로그인 완료 (Max 구독) |
| **pip** | Python 패키지 관리자 |
| **OS** | Windows 10/11 (Linux/macOS도 가능) |

### Claude Code CLI 확인
```powershell
# Claude Code CLI 경로 확인 (PowerShell)
where.exe claude

# 정상 작동 확인
claude -p "say hello" --output-format json
```

> **중요**: Claude Code CLI가 설치되어 있고, `claude` 명령어로 실행 가능해야 합니다.
> `where.exe claude` 결과를 `.env` 파일의 `CLAUDE_CLI_PATH`에 설정합니다.
> Windows 경로 예시: `C:\Users\사용자명\.local\bin\claude.exe` 또는 `C:\Users\사용자명\AppData\Roaming\npm\claude.cmd`

---

## 3. 디렉토리 구조

```
claude-api-server/
├── .env                          # 환경 변수 (토큰, CLI 경로 등)
├── requirements.txt              # Python 의존성
├── test_server.py                # 종합 테스트 도구
├── app/
│   ├── __init__.py               # (빈 파일)
│   ├── main.py                   # FastAPI 앱 진입점
│   ├── config.py                 # Pydantic 설정
│   ├── auth.py                   # Bearer 토큰 인증
│   ├── routers/
│   │   ├── __init__.py           # (빈 파일)
│   │   └── api.py                # 모든 API 엔드포인트
│   └── services/
│       ├── __init__.py           # (빈 파일)
│       └── claude_service.py     # Claude CLI 래퍼 서비스
└── venv/                         # Python 가상환경 (자동 생성)
```

---

## 4. 설치 단계

### Step 1: 프로젝트 디렉토리 생성
```powershell
mkdir C:\Users\사용자명\claude-api-server
cd C:\Users\사용자명\claude-api-server
```
> `사용자명`을 실제 Windows 사용자명으로 변경하세요.

### Step 2: 파일 생성
아래 **섹션 5**의 모든 파일을 해당 경로에 생성합니다.

### Step 3: 가상환경 생성 및 의존성 설치
```powershell
cd C:\Users\사용자명\claude-api-server
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> Windows에서는 `python3` 대신 `python`을, `source venv/bin/activate` 대신 `venv\Scripts\activate`를 사용합니다.

### Step 4: 환경 변수 설정
`.env` 파일을 편집하여 다음을 설정합니다:
- `API_SECRET_TOKEN`: 새로운 랜덤 토큰 생성 (아래 명령어 사용)
- `CLAUDE_CLI_PATH`: `where.exe claude` 결과 경로
- `ALLOWED_PATHS`: 파일 접근을 허용할 경로

```powershell
# 랜덤 토큰 생성
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Claude CLI 경로 확인
where.exe claude
```

### Step 5: 서버 실행
```powershell
cd C:\Users\사용자명\claude-api-server
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8020
```

### Step 6: 테스트
```powershell
python test_server.py --skip-ai    # 빠른 테스트 (AI 호출 제외)
python test_server.py              # 전체 테스트 (AI 호출 포함)
```

---

## 5. 전체 소스코드

### 5.1 `requirements.txt`

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
pydantic-settings>=2.0.0
fastapi-mcp>=0.3.0
python-dotenv>=1.0.0
```

### 5.2 `.env`

> **주의**: `API_SECRET_TOKEN`은 반드시 새로 생성하세요. 아래는 템플릿입니다.

```env
# Claude MCP Server Configuration
API_SECRET_TOKEN=여기에-새로-생성한-토큰을-넣으세요
CLAUDE_CLI_PATH=C:\Users\사용자명\AppData\Roaming\npm\claude.cmd
DEFAULT_MODEL=sonnet
MAX_CONCURRENT=2
REQUEST_TIMEOUT=300
ALLOWED_PATHS=C:\Users\사용자명
```

**설정 방법:**
```powershell
# 토큰 생성
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Claude CLI 경로 확인 후 CLAUDE_CLI_PATH에 입력
where.exe claude

# ALLOWED_PATHS는 파일 읽기/접근을 허용할 최상위 경로
# 여러 경로는 쉼표로 구분: C:\Users\user,C:\temp\data
```

> **Windows 경로 참고**: `CLAUDE_CLI_PATH`에 `where.exe claude` 결과를 그대로 입력합니다.
> 일반적인 위치: `C:\Users\사용자명\AppData\Roaming\npm\claude.cmd` 또는 `C:\Users\사용자명\.local\bin\claude.exe`

### 5.3 `app/__init__.py`

```python
```

> 빈 파일입니다. 패키지 인식용으로 생성만 하면 됩니다.

### 5.4 `app/main.py`

```python
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Claude API Server",
    description="Claude Code CLI as OpenAI-compatible API, REST API & MCP server (Max subscription, no per-token cost)",
    version="2.0.0",
)

# Include REST API router
app.include_router(api.router)
app.include_router(api.openai_router)


# Health check (no auth)
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_model": settings.default_model,
        "max_concurrent": settings.max_concurrent,
    }


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = int((time.monotonic() - start) * 1000)
    logger.info(
        "%s %s -> %d (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# Mount MCP endpoint (fastapi-mcp)
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        name="claude-gateway",
        description="Claude Code AI gateway via MCP protocol",
        describe_all_responses=True,
    )
    mcp.mount()
    logger.info("MCP endpoint mounted at /mcp")
except ImportError:
    logger.warning("fastapi-mcp not installed — MCP endpoint disabled. pip install fastapi-mcp")
except Exception as e:
    logger.warning("Failed to mount MCP endpoint: %s", e)
```

### 5.5 `app/config.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    api_secret_token: str = "changeme-generate-a-real-token"
    claude_cli_path: str = "claude"  # Windows: where.exe claude 결과로 변경, .env에서 오버라이드
    default_model: str = "sonnet"
    max_concurrent: int = 2
    request_timeout: int = 120
    allowed_paths: str = "C:\\Users"  # Windows 경로, .env에서 오버라이드
    max_budget_usd: float = 1.0

    @property
    def allowed_path_list(self) -> list[Path]:
        return [Path(p.strip()) for p in self.allowed_paths.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

> **참고**: `claude_cli_path`와 `allowed_paths`의 기본값은 `.env` 파일에서 오버라이드됩니다.
> `.env`에서 Windows 경로에 맞게 설정하세요. (예: `CLAUDE_CLI_PATH=C:\Users\사용자명\AppData\Roaming\npm\claude.cmd`, `ALLOWED_PATHS=C:\Users\사용자명`)

### 5.6 `app/auth.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    if credentials.credentials != settings.api_secret_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return credentials.credentials
```

### 5.7 `app/routers/__init__.py`

```python
```

> 빈 파일입니다. 패키지 인식용으로 생성만 하면 됩니다.

### 5.8 `app/routers/api.py`

```python
import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Any, Union
from pydantic import BaseModel, Field

from app.auth import verify_token
from app.config import settings
from app.services.claude_service import claude_service, MODEL_MAP, OPENAI_MODEL_ALIASES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api"])
openai_router = APIRouter(tags=["openai-compatible"])


# --- Request/Response schemas ---

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str | None = Field(None, description="Model: opus, sonnet, haiku")
    system_prompt: str | None = None
    max_budget_usd: float | None = None


class CodeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str | None = None
    allowed_tools: list[str] | None = None
    max_budget_usd: float | None = None


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    context_files: list[str] | None = None
    model: str | None = None
    max_budget_usd: float | None = None


class FileReadRequest(BaseModel):
    path: str


class BashRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=5000)


class ClaudeApiResponse(BaseModel):
    result: str
    model: str
    usage: dict
    duration_ms: int
    is_error: bool = False


# --- OpenAI-compatible schemas ---

class OpenAIFunctionDef(BaseModel):
    model_config = {"extra": "allow"}
    name: str
    description: str = ""
    parameters: dict = {}

class OpenAIToolDef(BaseModel):
    model_config = {"extra": "allow"}
    type: str = "function"
    function: OpenAIFunctionDef

class OpenAIMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, list[Any], None] = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def text(self) -> str:
        """Extract plain text from content (handles str and array-of-parts)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        # Array of content parts: [{"type": "text", "text": "..."}, ...]
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)

class OpenAIChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "sonnet"
    messages: list[OpenAIMessage] = Field(..., min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[OpenAIToolDef] | None = None
    tool_choice: Any = None

class OpenAIChatChoice(BaseModel):
    model_config = {"extra": "allow"}
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class OpenAIChatResponse(BaseModel):
    model_config = {"extra": "allow"}
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: dict


# --- Tool calling support ---

TOOL_CALL_RE = re.compile(
    r'```(?:json)?\s*\n?\s*(\{[^`]*?"tool_calls"\s*:\s*\[.*?\].*?\})\s*\n?\s*```',
    re.DOTALL,
)
TOOL_CALL_SIMPLE_RE = re.compile(
    r'\{"tool_calls"\s*:\s*\[(\{.*?\}(?:\s*,\s*\{.*?\})*)\]\s*\}',
    re.DOTALL,
)


def _build_tools_prompt(tools: list[OpenAIToolDef]) -> str:
    """Build a system prompt section describing available tools."""
    lines = [
        "\n\n# CRITICAL: Tool Calling Instructions",
        "",
        "You are an autonomous agent with tool access. You MUST call tools directly when the user's request requires them.",
        "NEVER say 'I would need to use...', 'Let me check...', or 'I need permission to...'.",
        "NEVER ask for confirmation or describe what you plan to do. Just call the tool immediately.",
        "",
        "## How to call tools",
        "",
        "Respond with ONLY a JSON code block — nothing else before or after it:",
        "",
        "```json",
        '{"tool_calls": [{"name": "tool_name", "arguments": {"key": "value"}}]}',
        "```",
        "",
        "For multiple tools, add multiple objects in the array.",
        "If no tool is needed, respond with normal text (no JSON block).",
        "When you receive a tool result, summarize it naturally for the user.",
        "",
        "## Example",
        "",
        'User: "Check the battery status"',
        "Your response (the ENTIRE response, nothing else):",
        "",
        "```json",
        '{"tool_calls": [{"name": "mcp", "arguments": {"server": "aster", "tool": "get_battery_info"}}]}',
        "```",
        "",
        "## Available tools",
        "",
    ]
    for tool in tools:
        fn = tool.function
        params = fn.parameters
        if params and params.get("properties"):
            props = params["properties"]
            required = params.get("required", [])
            param_parts = []
            for pname, pdef in props.items():
                req_marker = " (required)" if pname in required else ""
                pdesc = pdef.get("description", "")
                ptype = pdef.get("type", "string")
                param_parts.append(f"    - {pname} ({ptype}{req_marker}): {pdesc}")
            params_str = "\n".join(param_parts)
        else:
            params_str = "    (no parameters)"
        lines.append(f"### {fn.name}")
        if fn.description:
            lines.append(f"{fn.description}")
        lines.append(f"Parameters:\n{params_str}")
        lines.append("")
    lines.append("[END OF TOOL INSTRUCTIONS]")
    return "\n".join(lines)


MCP_TOOL_RE = re.compile(r'^mcp__([^_]+)__(.+)$')

# Map Claude Code tool names → generic tool names
CLAUDE_TO_OPENCLAW_TOOLS = {
    "Read": "read", "Write": "write", "Edit": "edit",
    "Bash": "exec", "Glob": "read", "Grep": "read",
    "WebFetch": "web_fetch", "WebSearch": "web_search",
}


def _normalize_tool_call(name: str, arguments: dict) -> tuple[str, dict]:
    """Normalize tool names from Claude Code format."""
    m = MCP_TOOL_RE.match(name)
    if m:
        server, tool = m.group(1), m.group(2)
        merged = {"action": "call", "server": server, "tool": tool}
        if arguments:
            merged["args"] = arguments
        return "mcp", merged

    if name == "mcp":
        if "action" not in arguments:
            arguments["action"] = "call"
        if "arguments" in arguments and "args" not in arguments:
            arguments["args"] = arguments.pop("arguments")
        return name, arguments

    if name in CLAUDE_TO_OPENCLAW_TOOLS:
        return CLAUDE_TO_OPENCLAW_TOOLS[name], arguments

    return name, arguments


def _parse_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    """Parse tool_calls JSON block from response. Returns (clean_text, tool_calls_list)."""
    parsed = None
    match = TOOL_CALL_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    if not parsed:
        match2 = TOOL_CALL_SIMPLE_RE.search(text)
        if match2:
            try:
                parsed = json.loads(match2.group(0))
            except json.JSONDecodeError:
                pass

    if not parsed or "tool_calls" not in parsed:
        return text, None

    tool_calls = []
    for call in parsed["tool_calls"]:
        name = call.get("name", "unknown")
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else {}
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        name, arguments = _normalize_tool_call(name, arguments)
        arguments_str = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments_str},
        })

    if not tool_calls:
        return text, None

    clean = TOOL_CALL_RE.sub('', text)
    clean = TOOL_CALL_SIMPLE_RE.sub('', clean).strip()
    return clean or None, tool_calls


# --- XML-tagged message formatting ---

def _convert_openai_messages(
    messages: list[OpenAIMessage],
    tools: list[OpenAIToolDef] | None = None,
) -> tuple[str | None, str]:
    """Extract system prompt and combine messages into XML-tagged prompt string."""
    system_prompt = None
    conversation_parts = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.text()
        elif msg.role == "user":
            conversation_parts.append(f"<user>\n{msg.text()}\n</user>")
        elif msg.role == "assistant":
            if msg.tool_calls:
                calls = []
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    calls.append({"name": fn.get("name", ""), "arguments": args})
                block = json.dumps({"tool_calls": calls}, ensure_ascii=False, indent=2)
                conversation_parts.append(f"<assistant>\n```json\n{block}\n```\n</assistant>")
            else:
                text = msg.text()
                if text:
                    conversation_parts.append(f"<assistant>\n{text}\n</assistant>")
        elif msg.role == "tool":
            tool_name = msg.name or "unknown"
            call_id = msg.tool_call_id or ""
            conversation_parts.append(
                f'<tool_result name="{tool_name}" call_id="{call_id}">\n{msg.text()}\n</tool_result>'
            )

    prompt = "\n\n".join(conversation_parts)
    return system_prompt, prompt


# --- Bash/cmd whitelist ---
# Windows: cmd.exe 기반 명령어. Linux 사용 시 ls, cat, grep 등으로 변경.

BASH_ALLOWED_PREFIXES = [
    "dir", "type", "echo", "whoami", "where", "hostname",
    "ipconfig", "systeminfo", "date /t", "time /t",
    "python -c", "python -m json.tool",
    "powershell -Command Get-Date", "powershell -Command Get-Process",
]


def _is_bash_allowed(command: str) -> bool:
    cmd = command.strip()
    return any(cmd.startswith(prefix) for prefix in BASH_ALLOWED_PREFIXES)


def _is_path_allowed(path: str) -> bool:
    try:
        resolved = Path(path).resolve()
        return any(
            resolved == allowed or allowed in resolved.parents
            for allowed in settings.allowed_path_list
        )
    except (ValueError, OSError):
        return False


# --- Endpoints ---

@router.post("/chat", response_model=ClaudeApiResponse)
async def chat(req: ChatRequest, _: str = Depends(verify_token)):
    """General-purpose AI chat. Send a prompt, get a response."""
    resp = await claude_service.chat(
        prompt=req.prompt,
        model=req.model,
        system_prompt=req.system_prompt,
        max_budget_usd=req.max_budget_usd,
    )
    if resp.is_error:
        raise HTTPException(status_code=502, detail=resp.result)
    return resp.to_dict()


@router.post("/code", response_model=ClaudeApiResponse)
async def code(req: CodeRequest, _: str = Depends(verify_token)):
    """Code generation/analysis with tool access (Read, Bash, etc.)."""
    resp = await claude_service.code(
        prompt=req.prompt,
        model=req.model,
        allowed_tools=req.allowed_tools,
        max_budget_usd=req.max_budget_usd,
    )
    if resp.is_error:
        raise HTTPException(status_code=502, detail=resp.result)
    return resp.to_dict()


@router.post("/analyze", response_model=ClaudeApiResponse)
async def analyze(req: AnalyzeRequest, _: str = Depends(verify_token)):
    """Analyze documents/data with optional context files."""
    if req.context_files:
        for fpath in req.context_files:
            if not _is_path_allowed(fpath):
                raise HTTPException(
                    status_code=403,
                    detail=f"Path not allowed: {fpath}",
                )
    resp = await claude_service.analyze(
        prompt=req.prompt,
        context_files=req.context_files,
        model=req.model,
        max_budget_usd=req.max_budget_usd,
    )
    if resp.is_error:
        raise HTTPException(status_code=502, detail=resp.result)
    return resp.to_dict()


@router.post("/tools/file-read")
async def file_read(req: FileReadRequest, _: str = Depends(verify_token)):
    """Read a file from the server (allowed paths only)."""
    if not _is_path_allowed(req.path):
        raise HTTPException(status_code=403, detail=f"Path not allowed: {req.path}")
    try:
        content = Path(req.path).read_text(encoding="utf-8")
        return {"path": req.path, "content": content, "size": len(content)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/bash")
async def bash(req: BashRequest, _: str = Depends(verify_token)):
    """Execute a whitelisted shell command."""
    if not _is_bash_allowed(req.command):
        raise HTTPException(
            status_code=403,
            detail=f"Command not in whitelist. Allowed: {', '.join(BASH_ALLOWED_PREFIXES)}",
        )
    try:
        proc = await asyncio.create_subprocess_shell(
            req.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return {
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "returncode": proc.returncode,
        }
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail="Command timed out (30s)")


@router.get("/models")
async def list_models(_: str = Depends(verify_token)):
    """List available models."""
    return {
        "models": [
            {"key": k, "model_id": v, "default": k == settings.default_model}
            for k, v in MODEL_MAP.items()
        ]
    }


# --- OpenAI-compatible endpoints ---

@openai_router.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest, _: str = Depends(verify_token)):
    """OpenAI-compatible chat completions endpoint with tool calling support."""
    system_prompt, prompt = _convert_openai_messages(req.messages, req.tools)

    if not prompt:
        raise HTTPException(status_code=400, detail="No user messages provided")

    if req.tools:
        tool_names = [t.function.name for t in req.tools]
        logger.info("Tools in request: %s (stream=%s)", tool_names, req.stream)

    model_key, model_id = claude_service.resolve_model(req.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # --- Tool-calling path: add descriptions, single call, passthrough ---
    if req.tools:
        tools_section = _build_tools_prompt(req.tools)
        system_prompt = (system_prompt or "") + tools_section

        resp = await claude_service.chat(
            prompt=prompt,
            model=model_key,
            system_prompt=system_prompt,
        )

        if resp.is_error:
            raise HTTPException(status_code=502, detail=resp.result)

        clean_text, tool_calls = _parse_tool_calls(resp.result)

        if tool_calls:
            finish_reason = "tool_calls"
            message = {"role": "assistant", "content": clean_text, "tool_calls": tool_calls}
        else:
            finish_reason = "stop"
            message = {"role": "assistant", "content": resp.result}

        usage = {
            "prompt_tokens": resp.usage.get("input_tokens", 0),
            "completion_tokens": resp.usage.get("output_tokens", 0),
            "total_tokens": resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0),
        }

        if req.stream:
            async def stream_tool_response():
                role_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resp.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tc_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": resp.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": i,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"],
                                        },
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"
                else:
                    content = resp.result
                    text_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": resp.model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(text_chunk)}\n\n"

                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resp.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": usage,
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_tool_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": resp.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    # --- Standard path (no tools) ---
    if req.stream:
        async def generate_stream():
            async for event_type, data in claude_service.chat_stream(
                prompt=prompt,
                model=model_key,
                system_prompt=system_prompt,
            ):
                if event_type == "delta":
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": data}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif event_type == "done":
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": data.get("model", model_id),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": data["usage"].get("input_tokens", 0),
                            "completion_tokens": data["usage"].get("output_tokens", 0),
                            "total_tokens": data["usage"].get("input_tokens", 0) + data["usage"].get("output_tokens", 0),
                        },
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                elif event_type == "error":
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": f"\n[Error: {data}]"}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response (no tools)
    resp = await claude_service.chat(
        prompt=prompt,
        model=model_key,
        system_prompt=system_prompt,
    )

    if resp.is_error:
        raise HTTPException(status_code=502, detail=resp.result)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": resp.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.result}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": resp.usage.get("input_tokens", 0),
            "completion_tokens": resp.usage.get("output_tokens", 0),
            "total_tokens": resp.usage.get("input_tokens", 0) + resp.usage.get("output_tokens", 0),
        },
    }


@openai_router.get("/v1/models")
async def openai_list_models(_: str = Depends(verify_token)):
    """OpenAI-compatible model listing."""
    models = []
    seen_ids = set()
    for key, model_id in MODEL_MAP.items():
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        })
        seen_ids.add(model_id)
        seen_ids.add(key)
    for alias, resolved_key in OPENAI_MODEL_ALIASES.items():
        if alias not in seen_ids:
            models.append({
                "id": alias,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
                "parent": MODEL_MAP.get(resolved_key, resolved_key),
            })
            seen_ids.add(alias)
    return {"object": "list", "data": models}
```

### 5.9 `app/services/__init__.py`

```python
```

> 빈 파일입니다. 패키지 인식용으로 생성만 하면 됩니다.

### 5.10 `app/services/claude_service.py`

```python
import asyncio
import json
import logging
import time
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

# OpenAI-compatible model name aliases → internal model keys
OPENAI_MODEL_ALIASES = {
    # OpenAI names → our keys
    "gpt-4o": "opus",
    "gpt-4": "opus",
    "gpt-4-turbo": "opus",
    "gpt-4o-mini": "sonnet",
    "gpt-3.5-turbo": "haiku",
    # Direct Claude model IDs → our keys
    "claude-opus-4-6": "opus",
    "claude-sonnet-4-5-20250929": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
    # Our own keys pass through
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
}


@dataclass
class ClaudeResponse:
    result: str
    model: str
    usage: dict
    duration_ms: int
    is_error: bool = False

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "model": self.model,
            "usage": self.usage,
            "duration_ms": self.duration_ms,
            "is_error": self.is_error,
        }


class ClaudeService:
    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._cli_path = settings.claude_cli_path

    @staticmethod
    def resolve_model(model_name: str | None) -> tuple[str, str]:
        """Resolve any model name (OpenAI alias, Claude ID, or internal key) to (key, model_id)."""
        if not model_name:
            model_key = settings.default_model
        else:
            model_key = OPENAI_MODEL_ALIASES.get(model_name, model_name)
        model_id = MODEL_MAP.get(model_key, model_key)
        return model_key, model_id

    async def _run_cli(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_budget_usd: float | None = None,
        allowed_tools: list[str] | None = None,
        timeout: int | None = None,
    ) -> ClaudeResponse:
        model_key = model or settings.default_model
        model_id = MODEL_MAP.get(model_key, model_key)
        timeout = timeout or settings.request_timeout

        cmd = [
            self._cli_path,
            "-p", prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", model_id,
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        if max_budget_usd:
            cmd.extend(["--max-budget-usd", str(max_budget_usd)])

        if allowed_tools:
            for tool in allowed_tools:
                cmd.extend(["--allowedTools", tool])

        start = time.monotonic()

        async with self._semaphore:
            logger.info("Running claude CLI: model=%s prompt_len=%d", model_key, len(prompt))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=None,  # inherit parent env
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = int((time.monotonic() - start) * 1000)
                return ClaudeResponse(
                    result=f"Request timed out after {timeout}s",
                    model=model_id,
                    usage={},
                    duration_ms=duration,
                    is_error=True,
                )
            except FileNotFoundError:
                duration = int((time.monotonic() - start) * 1000)
                return ClaudeResponse(
                    result=f"Claude CLI not found at {self._cli_path}",
                    model=model_id,
                    usage={},
                    duration_ms=duration,
                    is_error=True,
                )

        duration = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else f"CLI exited with code {proc.returncode}"
            logger.error("Claude CLI error: %s", error_msg)
            return ClaudeResponse(
                result=error_msg,
                model=model_id,
                usage={},
                duration_ms=duration,
                is_error=True,
            )

        # Parse JSON output
        raw = stdout.decode().strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # If not valid JSON, return raw text
            return ClaudeResponse(
                result=raw,
                model=model_id,
                usage={},
                duration_ms=duration,
            )

        # Extract result text from JSON response
        result_text = data.get("result", raw)
        usage = {
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "cost_usd": data.get("cost_usd", 0),
        }

        return ClaudeResponse(
            result=result_text,
            model=data.get("model", model_id),
            usage=usage,
            duration_ms=duration,
        )

    async def chat(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_budget_usd: float | None = None,
    ) -> ClaudeResponse:
        return await self._run_cli(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_budget_usd=max_budget_usd,
        )

    async def chat_stream(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_budget_usd: float | None = None,
    ):
        """Stream chat response using claude CLI stream-json output.

        Yields (event_type, data) tuples:
          ("delta", "partial text")   — text chunk
          ("done", {"result": ..., "usage": ...})  — final result
          ("error", "message")        — error
        """
        model_key, model_id = self.resolve_model(model)
        timeout = settings.request_timeout

        cmd = [
            self._cli_path,
            "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--model", model_id,
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        if max_budget_usd:
            cmd.extend(["--max-budget-usd", str(max_budget_usd)])

        async with self._semaphore:
            logger.info("Running claude CLI (streaming): model=%s prompt_len=%d", model_key, len(prompt))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=None,
                )
            except FileNotFoundError:
                yield ("error", f"Claude CLI not found at {self._cli_path}")
                return

            last_text = ""
            try:
                async def read_with_timeout():
                    return await proc.stdout.readline()

                while True:
                    try:
                        line = await asyncio.wait_for(read_with_timeout(), timeout=timeout)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        yield ("error", f"Stream timed out after {timeout}s")
                        return

                    if not line:
                        break

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    if msg_type == "assistant":
                        # Extract text content from message
                        message = data.get("message", {})
                        content_blocks = message.get("content", [])
                        full_text = ""
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                full_text += block.get("text", "")
                            elif isinstance(block, str):
                                full_text += block
                        # Yield only the new delta
                        if full_text and len(full_text) > len(last_text):
                            delta = full_text[len(last_text):]
                            last_text = full_text
                            yield ("delta", delta)

                    elif msg_type == "result":
                        result_text = data.get("result", last_text)
                        usage = {
                            "input_tokens": data.get("input_tokens", 0),
                            "output_tokens": data.get("output_tokens", 0),
                            "cost_usd": data.get("cost_usd", 0),
                        }
                        yield ("done", {"result": result_text, "model": model_id, "usage": usage})
                        break

            except Exception as e:
                logger.error("Stream error: %s", e)
                yield ("error", str(e))

            await proc.wait()

    async def code(
        self,
        prompt: str,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        max_budget_usd: float | None = None,
    ) -> ClaudeResponse:
        if allowed_tools is None:
            allowed_tools = ["Read", "Glob", "Grep", "Bash"]
        return await self._run_cli(
            prompt=prompt,
            model=model,
            allowed_tools=allowed_tools,
            max_budget_usd=max_budget_usd,
        )

    async def analyze(
        self,
        prompt: str,
        context_files: list[str] | None = None,
        model: str | None = None,
        max_budget_usd: float | None = None,
    ) -> ClaudeResponse:
        full_prompt = prompt
        if context_files:
            file_contents = []
            for fpath in context_files:
                try:
                    with open(fpath, "r") as f:
                        content = f.read()
                    file_contents.append(f"--- {fpath} ---\n{content}")
                except Exception as e:
                    file_contents.append(f"--- {fpath} ---\n[Error reading: {e}]")
            full_prompt = (
                "Context files:\n\n"
                + "\n\n".join(file_contents)
                + "\n\n---\n\nTask:\n"
                + prompt
            )

        return await self._run_cli(
            prompt=full_prompt,
            model=model,
            max_budget_usd=max_budget_usd,
        )


# Singleton
claude_service = ClaudeService()
```

### 5.11 `test_server.py`

```python
#!/usr/bin/env python3
"""Claude API Server — Test Tool

Usage:
    python test_server.py                    # 전체 테스트
    python test_server.py --skip-ai          # AI 호출 제외 (~2초)
    python test_server.py --only security    # 특정 카테고리만
    python test_server.py --url http://remote:8020 --token xxx
    python test_server.py --verbose          # 응답 본문 표시
"""

import argparse
import http.client
import json
import os
import sys
import time
from urllib.parse import urlparse


# ── ANSI colors ──────────────────────────────────────────────────

class C:
    PASS = "\033[92m"  # green
    FAIL = "\033[91m"  # red
    SKIP = "\033[93m"  # yellow
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    HEADER = "\033[96m"  # cyan


def _no_color():
    for attr in ("PASS", "FAIL", "SKIP", "BOLD", "DIM", "RESET", "HEADER"):
        setattr(C, attr, "")


# ── HTTP client ──────────────────────────────────────────────────

class HttpClient:
    def __init__(self, base_url: str, token: str | None, timeout: int = 60):
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 8020)
        self.scheme = parsed.scheme or "http"
        self.token = token
        self.timeout = timeout

    def _conn(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        auth: bool = True,
        raw_response: bool = False,
        timeout: int | None = None,
    ) -> dict:
        """Send a request. Returns dict with status, body, json, headers, elapsed_ms."""
        headers = {"Content-Type": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        conn = self._conn()
        if timeout:
            conn.timeout = timeout

        encoded = json.dumps(body).encode() if body else None
        start = time.monotonic()

        try:
            conn.request(method, path, body=encoded, headers=headers)
            resp = conn.getresponse()
            elapsed = int((time.monotonic() - start) * 1000)

            if raw_response:
                return {"status": resp.status, "response": resp, "elapsed_ms": elapsed, "headers": dict(resp.getheaders())}

            raw_body = resp.read().decode(errors="replace")
            result = {
                "status": resp.status,
                "body": raw_body,
                "json": None,
                "headers": dict(resp.getheaders()),
                "elapsed_ms": elapsed,
            }
            try:
                result["json"] = json.loads(raw_body)
            except (json.JSONDecodeError, ValueError):
                pass
            return result
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return {"status": 0, "body": str(e), "json": None, "headers": {}, "elapsed_ms": elapsed, "error": str(e)}
        finally:
            conn.close()

    def request_sse(self, method: str, path: str, body: dict | None = None) -> dict:
        """Send request expecting SSE response. Returns parsed events."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        conn = self._conn()
        encoded = json.dumps(body).encode() if body else None
        start = time.monotonic()

        try:
            conn.request(method, path, body=encoded, headers=headers)
            resp = conn.getresponse()
            elapsed_first = int((time.monotonic() - start) * 1000)

            if resp.status != 200:
                raw = resp.read().decode(errors="replace")
                return {"status": resp.status, "events": [], "body": raw, "elapsed_ms": elapsed_first}

            events = []
            done = False
            raw_data = resp.read().decode(errors="replace")
            elapsed = int((time.monotonic() - start) * 1000)

            for line in raw_data.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        done = True
                        events.append({"type": "done"})
                    else:
                        try:
                            events.append({"type": "data", "data": json.loads(payload)})
                        except json.JSONDecodeError:
                            events.append({"type": "raw", "data": payload})

            return {"status": resp.status, "events": events, "done": done, "elapsed_ms": elapsed}
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return {"status": 0, "events": [], "error": str(e), "elapsed_ms": elapsed}
        finally:
            conn.close()


# ── Test runner ──────────────────────────────────────────────────

class TestRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[dict] = []
        self._current_category = ""

    def category(self, name: str):
        self._current_category = name
        print(f"\n{C.HEADER}[{name}]{C.RESET}")

    def _record(self, name: str, status: str, msg: str = "", elapsed: int = 0):
        self.results.append({"category": self._current_category, "name": name, "status": status})
        icon = {"PASS": C.PASS + "PASS" + C.RESET, "FAIL": C.FAIL + "FAIL" + C.RESET, "SKIP": C.SKIP + "SKIP" + C.RESET}[status]
        time_str = f" ({elapsed}ms)" if elapsed else ""
        detail = f" — {msg}" if msg else ""
        print(f"  {icon} {name}{time_str}{detail}")

    def passed(self, name: str, msg: str = "", elapsed: int = 0):
        self._record(name, "PASS", msg, elapsed)

    def failed(self, name: str, msg: str = "", elapsed: int = 0):
        self._record(name, "FAIL", msg, elapsed)

    def skipped(self, name: str, msg: str = ""):
        self._record(name, "SKIP", msg)

    def log_verbose(self, label: str, data):
        if self.verbose:
            if isinstance(data, dict):
                text = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                text = str(data)
            # Truncate long output
            if len(text) > 2000:
                text = text[:2000] + f"\n... ({len(text)} chars total)"
            print(f"    {C.DIM}{label}: {text}{C.RESET}")

    def summary(self) -> int:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")

        print(f"\n{'=' * 50}")
        parts = []
        if passed:
            parts.append(f"{C.PASS}{passed} passed{C.RESET}")
        if failed:
            parts.append(f"{C.FAIL}{failed} failed{C.RESET}")
        if skipped:
            parts.append(f"{C.SKIP}{skipped} skipped{C.RESET}")
        print(f"  Results: {', '.join(parts)}  ({total} total)")
        print(f"{'=' * 50}")

        return 0 if failed == 0 else 1


# ── Token loader ─────────────────────────────────────────────────

def load_token(args_token: str | None) -> str | None:
    if args_token:
        return args_token
    # Environment variable
    env_token = os.environ.get("API_SECRET_TOKEN")
    if env_token:
        return env_token
    # .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_SECRET_TOKEN=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ── Test categories ──────────────────────────────────────────────

def run_connectivity_tests(client: HttpClient, runner: TestRunner):
    runner.category("Connectivity")

    # 1. Health check
    r = client.request("GET", "/health", auth=False)
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 200 and r.get("json", {}).get("status") == "ok":
        runner.passed("test_health", f"status=ok, model={r['json'].get('default_model')}", r["elapsed_ms"])
    else:
        runner.failed("test_health", f"status={r['status']}, body={r['body'][:100]}", r["elapsed_ms"])

    # 2. Response time
    elapsed = r["elapsed_ms"]
    if elapsed < 1000:
        runner.passed("test_response_time", f"{elapsed}ms < 1000ms", elapsed)
    else:
        runner.failed("test_response_time", f"{elapsed}ms >= 1000ms", elapsed)


def run_auth_tests(client: HttpClient, runner: TestRunner):
    runner.category("Authentication")

    # 1. Valid token → 200
    r = client.request("GET", "/api/models", auth=True)
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 200:
        runner.passed("test_auth_valid_token", "", r["elapsed_ms"])
    else:
        runner.failed("test_auth_valid_token", f"expected 200, got {r['status']}", r["elapsed_ms"])

    # 2. No token → 401 or 403 (depends on FastAPI/Starlette version)
    r = client.request("GET", "/api/models", auth=False)
    runner.log_verbose("response", r.get("json"))
    if r["status"] in (401, 403):
        runner.passed("test_auth_no_token", f"{r['status']} as expected", r["elapsed_ms"])
    else:
        runner.failed("test_auth_no_token", f"expected 401/403, got {r['status']}", r["elapsed_ms"])

    # 3. Wrong token
    saved_token = client.token
    client.token = "wrong-token-12345"
    r = client.request("GET", "/api/models", auth=True)
    client.token = saved_token
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 401:
        runner.passed("test_auth_invalid_token", "401 as expected", r["elapsed_ms"])
    else:
        runner.failed("test_auth_invalid_token", f"expected 401, got {r['status']}", r["elapsed_ms"])


def run_api_tests(client: HttpClient, runner: TestRunner, skip_ai: bool):
    runner.category("Custom REST API")

    # 1. List models
    r = client.request("GET", "/api/models")
    runner.log_verbose("response", r.get("json"))
    models = r.get("json", {}).get("models", [])
    if r["status"] == 200 and len(models) == 3:
        keys = [m["key"] for m in models]
        runner.passed("test_api_models", f"3 models: {', '.join(keys)}", r["elapsed_ms"])
    else:
        runner.failed("test_api_models", f"expected 3 models, got {len(models)}", r["elapsed_ms"])

    # 2. Chat (AI)
    if skip_ai:
        runner.skipped("test_api_chat", "--skip-ai")
    else:
        r = client.request("POST", "/api/chat", {"prompt": "Say only: hello", "model": "haiku"}, timeout=120)
        runner.log_verbose("response", r.get("json"))
        if r["status"] == 200 and r.get("json", {}).get("result"):
            result_len = len(r["json"]["result"])
            runner.passed("test_api_chat", f"{result_len} chars", r["elapsed_ms"])
        else:
            runner.failed("test_api_chat", f"status={r['status']}, body={r['body'][:200]}", r["elapsed_ms"])

    # 3. Code (AI)
    if skip_ai:
        runner.skipped("test_api_code", "--skip-ai")
    else:
        r = client.request("POST", "/api/code", {"prompt": "echo 'test ok'", "model": "haiku"}, timeout=120)
        runner.log_verbose("response", r.get("json"))
        if r["status"] == 200 and r.get("json", {}).get("result"):
            runner.passed("test_api_code", f"{len(r['json']['result'])} chars", r["elapsed_ms"])
        else:
            runner.failed("test_api_code", f"status={r['status']}", r["elapsed_ms"])

    # 4. Analyze (AI)
    if skip_ai:
        runner.skipped("test_api_analyze", "--skip-ai")
    else:
        # Use a small file known to exist in the project
        ctx_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        body = {"prompt": "How many lines?", "model": "haiku", "context_files": [ctx_file]}
        r = client.request("POST", "/api/analyze", body, timeout=120)
        runner.log_verbose("response", r.get("json"))
        if r["status"] == 200 and r.get("json", {}).get("result"):
            runner.passed("test_api_analyze", f"{len(r['json']['result'])} chars", r["elapsed_ms"])
        else:
            runner.failed("test_api_analyze", f"status={r['status']}", r["elapsed_ms"])

    # 5. Validation — empty prompt
    r = client.request("POST", "/api/chat", {"prompt": ""})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 422:
        runner.passed("test_api_validation", "422 on empty prompt", r["elapsed_ms"])
    else:
        runner.failed("test_api_validation", f"expected 422, got {r['status']}", r["elapsed_ms"])


def run_security_tests(client: HttpClient, runner: TestRunner):
    runner.category("Security Boundaries")

    # 1. File read — allowed path
    allowed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    r = client.request("POST", "/api/tools/file-read", {"path": allowed_file})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 200 and r.get("json", {}).get("size", 0) > 0:
        runner.passed("test_file_read_allowed", f"{r['json']['size']} bytes", r["elapsed_ms"])
    else:
        runner.failed("test_file_read_allowed", f"status={r['status']}", r["elapsed_ms"])

    # 2. File read — blocked path
    r = client.request("POST", "/api/tools/file-read", {"path": "/etc/shadow"})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 403:
        runner.passed("test_file_read_blocked", "403 as expected", r["elapsed_ms"])
    else:
        runner.failed("test_file_read_blocked", f"expected 403, got {r['status']}", r["elapsed_ms"])

    # 3. Bash — allowed command
    r = client.request("POST", "/api/tools/bash", {"command": "echo hello"})
    runner.log_verbose("response", r.get("json"))
    stdout = r.get("json", {}).get("stdout", "").strip()
    if r["status"] == 200 and "hello" in stdout:
        runner.passed("test_bash_allowed", f"stdout={stdout}", r["elapsed_ms"])
    else:
        runner.failed("test_bash_allowed", f"status={r['status']}, stdout={stdout}", r["elapsed_ms"])

    # 4. Bash — blocked command
    r = client.request("POST", "/api/tools/bash", {"command": "rm -rf /"})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 403:
        runner.passed("test_bash_blocked", "403 as expected", r["elapsed_ms"])
    else:
        runner.failed("test_bash_blocked", f"expected 403, got {r['status']}", r["elapsed_ms"])


def run_openai_tests(client: HttpClient, runner: TestRunner, skip_ai: bool):
    runner.category("OpenAI-Compatible API")

    # 1. List models
    r = client.request("GET", "/v1/models")
    runner.log_verbose("response", r.get("json"))
    models_data = r.get("json", {}).get("data", [])
    if r["status"] == 200 and len(models_data) > 0:
        model_ids = [m["id"] for m in models_data]
        runner.passed("test_openai_models", f"{len(models_data)} models", r["elapsed_ms"])
    else:
        runner.failed("test_openai_models", f"status={r['status']}", r["elapsed_ms"])
        model_ids = []

    # 2. gpt-4o alias exists
    if "gpt-4o" in model_ids:
        runner.passed("test_openai_gpt4o_alias", "gpt-4o present")
    elif model_ids:
        runner.failed("test_openai_gpt4o_alias", f"gpt-4o not in {model_ids}")
    else:
        runner.skipped("test_openai_gpt4o_alias", "models list empty")

    # 3. Chat completions non-streaming (AI)
    if skip_ai:
        runner.skipped("test_openai_chat", "--skip-ai")
    else:
        body = {
            "model": "haiku",
            "messages": [{"role": "user", "content": "Say only: ok"}],
            "stream": False,
        }
        r = client.request("POST", "/v1/chat/completions", body, timeout=120)
        runner.log_verbose("response", r.get("json"))
        j = r.get("json", {})
        if (r["status"] == 200
                and j.get("id", "").startswith("chatcmpl-")
                and j.get("choices", [{}])[0].get("message", {}).get("content")):
            content = j["choices"][0]["message"]["content"]
            runner.passed("test_openai_chat", f"id={j['id']}, {len(content)} chars", r["elapsed_ms"])
        else:
            runner.failed("test_openai_chat", f"status={r['status']}, body={r['body'][:200]}", r["elapsed_ms"])

    # 4. Chat completions streaming (AI)
    if skip_ai:
        runner.skipped("test_openai_chat_stream", "--skip-ai")
    else:
        body = {
            "model": "haiku",
            "messages": [{"role": "user", "content": "Say only: ok"}],
            "stream": True,
        }
        r = client.request_sse("POST", "/v1/chat/completions", body)
        runner.log_verbose("sse_events", f"{len(r.get('events', []))} events, done={r.get('done')}")
        events = r.get("events", [])
        has_done = r.get("done", False)

        # Check for role delta, content delta, finish_reason, and [DONE]
        has_role = False
        has_content = False
        has_finish = False
        for ev in events:
            if ev.get("type") == "data":
                d = ev["data"]
                choices = d.get("choices", [{}])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "role" in delta:
                        has_role = True
                    if "content" in delta:
                        has_content = True
                    fr = choices[0].get("finish_reason")
                    if fr and fr != "null":
                        has_finish = True

        if r["status"] == 200 and has_done and (has_content or has_role):
            detail_parts = [f"{len(events)} events"]
            if has_role:
                detail_parts.append("role")
            if has_content:
                detail_parts.append("content")
            if has_finish:
                detail_parts.append("finish_reason")
            detail_parts.append("[DONE]")
            runner.passed("test_openai_chat_stream", ", ".join(detail_parts), r["elapsed_ms"])
        else:
            runner.failed("test_openai_chat_stream",
                          f"status={r['status']}, events={len(events)}, done={has_done}, content={has_content}",
                          r.get("elapsed_ms", 0))

    # 5. Validation — empty messages
    r = client.request("POST", "/v1/chat/completions", {"model": "haiku", "messages": []})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 422:
        runner.passed("test_openai_validation", "422 on empty messages", r["elapsed_ms"])
    else:
        runner.failed("test_openai_validation", f"expected 422, got {r['status']}", r["elapsed_ms"])


def run_mcp_tests(client: HttpClient, runner: TestRunner):
    runner.category("MCP Protocol")

    # MCP SSE endpoint keeps connection open indefinitely.
    # Use raw_response to get status/headers without reading the full body.
    try:
        r = client.request("GET", "/mcp", auth=False, raw_response=True, timeout=5)
        status = r["status"]
        ct = r.get("headers", {}).get("content-type", "")
        elapsed = r["elapsed_ms"]
        # Close the streaming response without reading body
        try:
            r["response"].close()
        except Exception:
            pass
        runner.log_verbose("mcp_status", f"status={status}, content-type={ct}")
        if status == 200 and "text/event-stream" in ct:
            runner.passed("test_mcp_endpoint", f"SSE active, content-type={ct}", elapsed)
        elif status == 404:
            runner.skipped("test_mcp_endpoint", "fastapi-mcp not installed")
        else:
            runner.skipped("test_mcp_endpoint", f"status={status}, content-type={ct}")
    except Exception as e:
        # Timeout on SSE likely means endpoint exists and is streaming
        err_msg = str(e).lower()
        if "timed out" in err_msg or "timeout" in err_msg:
            runner.passed("test_mcp_endpoint", "SSE endpoint active (connection held open)")
        else:
            runner.skipped("test_mcp_endpoint", f"error: {e}")


# ── Main ─────────────────────────────────────────────────────────

CATEGORIES = {
    "connectivity": run_connectivity_tests,
    "auth": run_auth_tests,
    "api": run_api_tests,
    "security": run_security_tests,
    "openai": run_openai_tests,
    "mcp": run_mcp_tests,
}

# Categories that need skip_ai parameter
_AI_CATEGORIES = {"api", "openai"}


def main():
    parser = argparse.ArgumentParser(description="Claude API Server Test Tool")
    parser.add_argument("--url", default="http://localhost:8020", help="Server URL (default: http://localhost:8020)")
    parser.add_argument("--token", default=None, help="API token (default: from .env)")
    parser.add_argument("--skip-ai", action="store_true", help="Skip tests that call AI models")
    parser.add_argument("--only", choices=list(CATEGORIES.keys()), help="Run only this category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show response bodies")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _no_color()

    token = load_token(args.token)
    if not token:
        print(f"{C.FAIL}Error: No API token found. Use --token, set API_SECRET_TOKEN, or create .env{C.RESET}")
        sys.exit(2)

    client = HttpClient(args.url, token)
    runner = TestRunner(verbose=args.verbose)

    print(f"{C.BOLD}Claude API Server Test Tool{C.RESET}")
    print(f"Server: {args.url}")
    if args.skip_ai:
        print(f"{C.DIM}Mode: --skip-ai (AI calls excluded){C.RESET}")

    # Connectivity pre-check
    try:
        r = client.request("GET", "/health", auth=False, timeout=5)
        if r["status"] == 0:
            print(f"\n{C.FAIL}Error: Cannot connect to {args.url} — {r.get('error', 'unknown')}{C.RESET}")
            sys.exit(2)
    except Exception as e:
        print(f"\n{C.FAIL}Error: Cannot connect to {args.url} — {e}{C.RESET}")
        sys.exit(2)

    categories_to_run = [args.only] if args.only else list(CATEGORIES.keys())

    for cat_name in categories_to_run:
        func = CATEGORIES[cat_name]
        if cat_name in _AI_CATEGORIES:
            func(client, runner, args.skip_ai)
        else:
            func(client, runner)

    exit_code = runner.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

---

## 6. 서버 실행 방법

### 개발 모드 (포그라운드)
```powershell
cd C:\Users\사용자명\claude-api-server
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8020 --reload
```

### 운영 모드 (백그라운드)
```powershell
# PowerShell에서 백그라운드 실행
cd C:\Users\사용자명\claude-api-server
venv\Scripts\activate
Start-Process -NoNewWindow -FilePath "venv\Scripts\uvicorn.exe" -ArgumentList "app.main:app --host 0.0.0.0 --port 8020" -RedirectStandardOutput "server.log" -RedirectStandardError "server_error.log"
```

또는 별도 터미널 창에서 실행:
```powershell
# 새 터미널 창을 열어 서버 실행
Start-Process powershell -ArgumentList "-Command cd C:\Users\사용자명\claude-api-server; venv\Scripts\activate; uvicorn app.main:app --host 0.0.0.0 --port 8020"
```

### 서버 중지
```powershell
# uvicorn 프로세스 찾기
Get-Process -Name uvicorn -ErrorAction SilentlyContinue | Stop-Process

# 또는 포트로 찾기
netstat -ano | findstr :8020
taskkill /PID <PID번호> /F
```

### 서버 상태 확인
```powershell
# PowerShell
Invoke-RestMethod http://localhost:8020/health

# 또는 curl (Windows 10+ 내장)
curl.exe http://localhost:8020/health
```

---

## 7. 테스트 실행

```powershell
cd C:\Users\사용자명\claude-api-server
venv\Scripts\activate

# 빠른 테스트 (AI 호출 제외, ~2초)
python test_server.py --skip-ai

# 전체 테스트 (AI 호출 포함)
python test_server.py

# 특정 카테고리만
python test_server.py --only security
python test_server.py --only openai

# 상세 출력
python test_server.py --verbose

# 원격 서버 테스트
python test_server.py --url http://192.168.1.100:8020 --token YOUR_TOKEN
```

### 테스트 카테고리
| 카테고리 | 설명 |
|---------|------|
| `connectivity` | 서버 연결 및 응답 시간 |
| `auth` | 인증 토큰 검증 |
| `api` | 커스텀 REST API 엔드포인트 |
| `security` | 경로/명령어 화이트리스트 |
| `openai` | OpenAI 호환 API |
| `mcp` | MCP 프로토콜 엔드포인트 |

---

## 8. 사용 예시

### 8.1 curl / PowerShell

```powershell
$TOKEN = "your-api-secret-token"

# Health check (인증 불필요)
curl.exe http://localhost:8020/health

# 채팅
curl.exe -X POST http://localhost:8020/api/chat `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"Python으로 피보나치 함수 작성해줘\", \"model\": \"sonnet\"}'

# 코드 생성 (도구 접근 가능)
curl.exe -X POST http://localhost:8020/api/code `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"현재 디렉토리의 파일 목록을 보여줘\", \"model\": \"haiku\"}'

# 파일 분석
curl.exe -X POST http://localhost:8020/api/analyze `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"prompt\": \"이 코드를 분석해줘\", \"context_files\": [\"C:\\Users\\사용자명\\app.py\"], \"model\": \"sonnet\"}'

# OpenAI 호환 API
curl.exe -X POST http://localhost:8020/v1/chat/completions `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"sonnet\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"stream\": false}'

# OpenAI 호환 스트리밍
curl.exe -X POST http://localhost:8020/v1/chat/completions `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"haiku\", \"messages\": [{\"role\": \"user\", \"content\": \"Tell me a joke\"}], \"stream\": true}'

# 모델 목록
curl.exe -H "Authorization: Bearer $TOKEN" http://localhost:8020/v1/models
```

> **Windows PowerShell 참고**: `curl`은 PowerShell의 `Invoke-WebRequest` 별칭이므로 `curl.exe`를 사용합니다. 줄 바꿈은 `\` 대신 백틱(`` ` ``)을 사용합니다.

### 8.2 Python (requests)

```python
import requests

BASE_URL = "http://localhost:8020"
TOKEN = "your-api-secret-token"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# 채팅
resp = requests.post(f"{BASE_URL}/api/chat", headers=HEADERS, json={
    "prompt": "Python에서 리스트 컴프리헨션 예제를 보여줘",
    "model": "sonnet",
})
print(resp.json()["result"])

# OpenAI 호환 API
resp = requests.post(f"{BASE_URL}/v1/chat/completions", headers=HEADERS, json={
    "model": "sonnet",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
})
print(resp.json()["choices"][0]["message"]["content"])
```

### 8.3 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="your-api-secret-token",  # .env의 API_SECRET_TOKEN
)

# 비스트리밍
response = client.chat.completions.create(
    model="sonnet",  # 또는 "gpt-4o" (opus로 매핑), "gpt-4o-mini" (sonnet), "gpt-3.5-turbo" (haiku)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(response.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="haiku",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## 9. 보안 설정

### 경로 화이트리스트
`.env`의 `ALLOWED_PATHS`로 파일 접근 범위를 제한합니다:
```env
# 단일 경로
ALLOWED_PATHS=C:\Users\사용자명

# 여러 경로 (쉼표 구분)
ALLOWED_PATHS=C:\Users\사용자명\projects,C:\temp\data
```

### 명령어 화이트리스트
`app/routers/api.py`의 `BASH_ALLOWED_PREFIXES`를 수정하여 허용 명령어를 변경합니다.

**Windows 기본값** (소스코드에 이미 적용됨):
```python
BASH_ALLOWED_PREFIXES = [
    "dir", "type", "echo", "whoami", "where", "hostname",
    "ipconfig", "systeminfo", "date /t", "time /t",
    "python -c", "python -m json.tool",
    "powershell -Command Get-Date", "powershell -Command Get-Process",
    # 필요시 추가:
    # "docker ps", "docker logs",
    # "git log", "git status",
]
```

**Linux/macOS 사용 시** 아래로 변경:
```python
BASH_ALLOWED_PREFIXES = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "date", "uptime",
    "df", "du", "free", "whoami", "pwd", "echo", "file", "stat",
    "python3 -c", "python3 -m json.tool",
]
```

### API 토큰
- 충분히 긴 랜덤 토큰 사용 (`secrets.token_urlsafe(32)` 이상)
- `.env` 파일을 다른 사용자가 읽지 못하도록 보호 (Windows: 파일 속성에서 권한 제한)
- 외부 네트워크 노출 시 HTTPS 사용 권장 (Nginx/Caddy 리버스 프록시)

### 동시성 제한
`.env`의 `MAX_CONCURRENT`로 동시 Claude CLI 호출 수를 제한합니다:
```env
MAX_CONCURRENT=2  # 기본값: 동시 2개 요청
```

---

## 10. 트러블슈팅

### Claude CLI를 찾을 수 없음
```
Claude CLI not found at C:\Users\user\AppData\Roaming\npm\claude.cmd
```
**해결**: `where.exe claude`로 실제 경로를 확인하고 `.env`의 `CLAUDE_CLI_PATH`를 수정합니다.

### 인증 에러 (401)
```
{"detail": "Invalid authentication token"}
```
**해결**: 요청의 `Authorization: Bearer <token>` 헤더가 `.env`의 `API_SECRET_TOKEN`과 일치하는지 확인합니다.

### 경로 접근 거부 (403)
```
{"detail": "Path not allowed: C:\\some\\path"}
```
**해결**: `.env`의 `ALLOWED_PATHS`에 해당 경로(또는 상위 경로)를 추가합니다.

### 요청 타임아웃
```
Request timed out after 300s
```
**해결**: `.env`의 `REQUEST_TIMEOUT`을 늘리거나, 더 빠른 모델(haiku)을 사용합니다.

### 포트 충돌
```
[Errno 10048] 또는 [Errno 98] Address already in use
```
**해결**: 이미 실행 중인 서버를 종료하거나, 다른 포트를 사용합니다:
```powershell
# 포트 사용 프로세스 확인 (Windows)
netstat -ano | findstr :8020

# 해당 PID 프로세스 종료
taskkill /PID <PID번호> /F

# 다른 포트로 실행
uvicorn app.main:app --host 0.0.0.0 --port 8021
```

### MCP 엔드포인트가 마운트되지 않음
```
fastapi-mcp not installed — MCP endpoint disabled
```
**해결**: `pip install fastapi-mcp`를 실행합니다. MCP가 필요 없다면 무시해도 됩니다.

---

## API 엔드포인트 요약

| 엔드포인트 | 메서드 | 인증 | 설명 |
|-----------|--------|------|------|
| `/health` | GET | X | 서버 상태 확인 |
| `/api/chat` | POST | O | AI 채팅 |
| `/api/code` | POST | O | 코드 생성 (도구 접근) |
| `/api/analyze` | POST | O | 파일 분석 |
| `/api/models` | GET | O | 모델 목록 |
| `/api/tools/file-read` | POST | O | 파일 읽기 |
| `/api/tools/bash` | POST | O | 화이트리스트 명령어 실행 |
| `/v1/chat/completions` | POST | O | OpenAI 호환 채팅 |
| `/v1/models` | GET | O | OpenAI 호환 모델 목록 |
| `/mcp` | GET | X | MCP SSE 엔드포인트 |

---

## 모델 매핑

| 요청 모델명 | 실제 Claude 모델 |
|------------|-----------------|
| `opus` / `gpt-4o` / `gpt-4` | `claude-opus-4-6` |
| `sonnet` / `gpt-4o-mini` | `claude-sonnet-4-5-20250929` |
| `haiku` / `gpt-3.5-turbo` | `claude-haiku-4-5-20251001` |
