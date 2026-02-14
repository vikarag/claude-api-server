import asyncio
import json
import logging
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth import verify_token
from app.config import settings
from app.schemas import (
    ChatRequest, CodeRequest, AnalyzeRequest, FileReadRequest, BashRequest,
    ClaudeApiResponse, OpenAIChatRequest,
)
from app.utils import (
    build_tools_prompt, parse_tool_calls, convert_openai_messages,
    is_bash_allowed, is_path_allowed, BASH_ALLOWED_PREFIXES,
)
from app.services.claude_service import claude_service, MODEL_MAP, OPENAI_MODEL_ALIASES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api"])
openai_router = APIRouter(tags=["openai-compatible"])


# --- Cached model list builders ---

@lru_cache(maxsize=1)
def _build_openai_models_list() -> dict:
    """Build the OpenAI-compatible models list (cached, static data)."""
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


# --- Custom REST API endpoints ---

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
            if not is_path_allowed(fpath):
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
    if not is_path_allowed(req.path):
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
    if not is_bash_allowed(req.command):
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
    system_prompt, prompt = convert_openai_messages(req.messages, req.tools)

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
        tools_section = build_tools_prompt(req.tools)
        system_prompt = (system_prompt or "") + tools_section

        resp = await claude_service.chat(
            prompt=prompt,
            model=model_key,
            system_prompt=system_prompt,
        )

        if resp.is_error:
            raise HTTPException(status_code=502, detail=resp.result)

        clean_text, tool_calls = parse_tool_calls(resp.result)

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
    return _build_openai_models_list()
