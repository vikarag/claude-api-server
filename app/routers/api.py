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


# --- Bash whitelist ---

BASH_ALLOWED_PREFIXES = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "date", "uptime",
    "df", "du", "free", "whoami", "pwd", "echo", "file", "stat",
    "python3 -c", "python3 -m json.tool",
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
