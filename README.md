# Claude API Server

An OpenAI-compatible API server that wraps the **Claude Code CLI** as a REST API and MCP server. Use your Claude Max subscription to serve AI requests at no additional per-token cost.

## Why?

| Method | Cost |
|--------|------|
| Anthropic API (direct) | ~$3–$15 per million tokens |
| **This server** | **$0 extra** — included in Claude Max subscription |

Any tool that speaks the OpenAI API format (LangChain, n8n, Open WebUI, Continue.dev) works out of the box — just change the base URL.

## Features

- **OpenAI-compatible API** — `POST /v1/chat/completions`, `GET /v1/models`
- **Custom REST API** — `/api/chat`, `/api/code`, `/api/analyze`, `/api/tools/file-read`, `/api/tools/bash`
- **SSE streaming** — real-time token-by-token responses
- **Tool calling** — OpenAI-format function calling support
- **MCP protocol** — auto-exposed via `fastapi-mcp`
- **Model aliases** — `gpt-4o` maps to Opus, `gpt-4o-mini` to Sonnet, `gpt-3.5-turbo` to Haiku
- **Security** — Bearer token auth, file path whitelist, command whitelist, concurrency limits

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

OpenAI aliases (`gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`) are automatically mapped to the corresponding Claude models.

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
