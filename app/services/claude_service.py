import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

from app.config import settings

# Remove env vars that prevent nested Claude Code sessions
_CLEAN_ENV = {k: v for k, v in os.environ.items()
              if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}

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
                    env=_CLEAN_ENV,
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
                    env=_CLEAN_ENV,
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
