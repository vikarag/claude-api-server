"""Request/Response Pydantic schemas for Claude API Server."""

from typing import Any, Union

from pydantic import BaseModel, Field


# --- Custom REST API schemas ---

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
