# Claude API Server

A production-ready API server that turns your **Claude Max subscription** into a full-featured AI backend. Get OpenAI-compatible endpoints, a custom REST API, and MCP protocol support — all at zero additional per-token cost.

## Why This Server?

| Method | Cost |
|--------|------|
| Anthropic API (direct) | ~$3–$15 per million tokens |
| OpenAI API (direct) | ~$2.50–$15 per million tokens |
| **This server** | **$0 extra** — included in Claude Max subscription |

Any tool that speaks the OpenAI API format — LangChain, n8n, Open WebUI, Continue.dev, Cursor — works out of the box. Just change the base URL.

## How It Compares

Most Claude-to-API proxy projects are minimal wrappers that pipe text in and out of the CLI. This server goes further:

| Feature | Claude API Server | Typical Proxy |
|---------|------------------|---------------|
| OpenAI-compatible API | Yes | Yes |
| Custom REST API (chat, code, analyze) | Yes | No |
| MCP Protocol | Yes | No |
| Tool Calling (function calling) | Yes | No |
| Security layers (auth, whitelists, limits) | Yes | Minimal |
| File & bash tool endpoints | Yes | No |
| Test suite included | Yes | No |
| Production deployment files | Yes | No |

### Triple API Surface

Unlike most proxies that only expose an OpenAI-compatible endpoint, this server provides **three ways to connect**:

1. **OpenAI-compatible** (`/v1/chat/completions`, `/v1/models`) — drop-in replacement for any OpenAI client
2. **Custom REST API** (`/api/chat`, `/api/code`, `/api/analyze`, `/api/tools/*`) — purpose-built endpoints with features like tool access and file analysis
3. **MCP Protocol** — auto-exposed via `fastapi-mcp` for agent-to-agent communication

### Security-First Design

Production deployments need guardrails. This server includes:

- **Bearer token authentication** — every request requires a valid API key
- **File path whitelisting** — restrict which directories Claude can access
- **Bash command whitelisting** — control which shell commands are allowed
- **Concurrency limiting** — prevent resource exhaustion with semaphore-based rate limiting

### Tool-Enabled Endpoints

The `/api/code` endpoint gives Claude access to `Read`, `Glob`, `Grep`, and `Bash` tools — it can actually explore files and run commands to solve coding tasks, not just generate text. Direct tool endpoints (`/api/tools/file-read`, `/api/tools/bash`) let you build file-aware and shell-aware workflows.

### Stateless by Design

Each request runs as an independent Claude CLI invocation. This is intentional:

- **No server-side session state** to manage, leak, or corrupt
- **Horizontal scaling** — any instance can handle any request
- **Multi-turn conversations** are handled by the client sending the full `messages` array (the standard OpenAI pattern used by Open WebUI, LangChain, etc.)

## Features

- **OpenAI-compatible API** — `POST /v1/chat/completions`, `GET /v1/models`
- **Custom REST API** — `/api/chat`, `/api/code`, `/api/analyze`, `/api/tools/file-read`, `/api/tools/bash`
- **SSE streaming** — real-time token-by-token responses
- **Tool calling** — OpenAI-format function calling support
- **MCP protocol** — auto-exposed via `fastapi-mcp`
- **Model aliases** — `gpt-4o` maps to Opus, `gpt-4o-mini` to Sonnet, `gpt-3.5-turbo` to Haiku
- **Security** — Bearer token auth, file path whitelist, command whitelist, concurrency limits
- **Production-ready** — systemd service file, `.env` configuration, comprehensive test suite

## Quick Start

```bash
# Clone and set up
git clone https://github.com/vikarag/claude-api-server.git
cd claude-api-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set API_SECRET_TOKEN and CLAUDE_CLI_PATH

# Run
uvicorn app.main:app --host 127.0.0.1 --port 8020
```

## Usage

```bash
# Health check
curl http://localhost:8020/health

# Chat (custom API)
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "model": "sonnet"}'

# OpenAI-compatible
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
# Works with the OpenAI Python SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="YOUR_TOKEN")
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Available Models

| Short Name | Model ID | Best For |
|------------|----------|----------|
| `haiku` | claude-haiku-4-5-20251001 | Fast, simple tasks |
| `sonnet` | claude-sonnet-4-5-20250929 | Balanced (default) |
| `opus` | claude-opus-4-6 | Complex reasoning |

OpenAI aliases (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`) are automatically mapped to the corresponding Claude models, so existing OpenAI client code works without modification.

## Prerequisites

- Python 3.12+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated with a Max subscription

## Documentation

See **[MANUAL.md](MANUAL.md)** for the full beginner-friendly guide covering all endpoints, security, testing, troubleshooting, and integration examples.

## Testing

```bash
python3 test_server.py --skip-ai    # Quick test (~2s, no AI calls)
python3 test_server.py              # Full test (includes AI calls)
```

## License

MIT
