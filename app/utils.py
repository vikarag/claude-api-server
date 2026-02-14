"""Utility functions for Claude API Server: tool calling, message conversion, security."""

import json
import re
import uuid
from pathlib import Path

from app.config import settings
from app.schemas import OpenAIMessage, OpenAIToolDef


# --- Tool calling support ---

TOOL_CALL_RE = re.compile(
    r'```(?:json)?\s*\n?\s*(\{[^`]*?"tool_calls"\s*:\s*\[.*?\].*?\})\s*\n?\s*```',
    re.DOTALL,
)
TOOL_CALL_SIMPLE_RE = re.compile(
    r'\{"tool_calls"\s*:\s*\[(\{.*?\}(?:\s*,\s*\{.*?\})*)\]\s*\}',
    re.DOTALL,
)

MCP_TOOL_RE = re.compile(r'^mcp__([^_]+)__(.+)$')

# Map Claude Code tool names -> generic tool names
CLAUDE_TO_OPENCLAW_TOOLS = {
    "Read": "read", "Write": "write", "Edit": "edit",
    "Bash": "exec", "Glob": "read", "Grep": "read",
    "WebFetch": "web_fetch", "WebSearch": "web_search",
}


def build_tools_prompt(tools: list[OpenAIToolDef]) -> str:
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


def normalize_tool_call(name: str, arguments: dict) -> tuple[str, dict]:
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


def parse_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
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
        name, arguments = normalize_tool_call(name, arguments)
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

def convert_openai_messages(
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


def is_bash_allowed(command: str) -> bool:
    cmd = command.strip()
    return any(cmd.startswith(prefix) for prefix in BASH_ALLOWED_PREFIXES)


def is_path_allowed(path: str) -> bool:
    try:
        resolved = Path(path).resolve()
        return any(
            resolved == allowed or allowed in resolved.parents
            for allowed in settings.allowed_path_list
        )
    except (ValueError, OSError):
        return False
