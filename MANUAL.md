# Claude API Server — Beginner's Manual

A plain-English guide for developers who want to understand and use this system.
No prior API or FastAPI experience required.

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Key Concepts Explained](#2-key-concepts-explained)
3. [Architecture Overview](#3-architecture-overview)
4. [Setup & Running](#4-setup--running)
5. [Authentication](#5-authentication)
6. [Available Models](#6-available-models)
7. [API Reference — Custom REST](#7-api-reference--custom-rest)
8. [API Reference — OpenAI-Compatible](#8-api-reference--openai-compatible)
9. [Security Boundaries](#9-security-boundaries)
10. [Testing](#10-testing)
11. [Common Use Cases](#11-common-use-cases)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)

---

## 1. What Is This?

Claude API Server is a **gateway server that lets you talk to Claude AI through HTTP
requests**. Think of it as a translator sitting between your code (or any tool that
can make web requests) and the Claude AI.

**Analogy:** Imagine Claude AI is a person who only speaks one language — the Claude
CLI command-line tool. Your code speaks a different language — HTTP requests. This
server sits in the middle and translates between the two. Your code sends an HTTP
request, the server converts it into a CLI command, gets Claude's answer, and sends
it back to your code in a format it understands.

### Why Does It Exist?

Normally, to use Claude AI in your programs, you would call the Anthropic API
directly. That API charges you **per token** (roughly per word) — the more you use
it, the more you pay. Costs can add up quickly:

| Method | Cost |
|--------|------|
| Anthropic API (direct) | ~$3 per million input tokens, ~$15 per million output tokens (Sonnet) |
| **This server** | **$0 additional** — included in the Claude Max subscription (flat monthly fee) |

This server uses the **Claude Code CLI** — a command-line tool that comes with a
Claude Max subscription. Since the Max subscription is a flat fee, every request
through this server costs nothing extra. If you make 10 requests or 10,000 requests,
the cost is the same.

### What Can It Replace?

Any place in your code where you currently call `anthropic.messages.create()` or use
the OpenAI SDK can be redirected to this server instead. You get the same AI
capabilities at no per-request cost.

---

## 2. Key Concepts Explained

If you already know what APIs and HTTP requests are, feel free to skip ahead. This
section explains the building blocks you will encounter throughout this manual.

### API (Application Programming Interface)

An API is a way for two programs to talk to each other. When you use Claude through
the chat window on claude.ai, *you* are the one typing and reading. An API lets your
*code* do the same thing — send a question, get an answer — without any human
clicking around in a browser.

### HTTP Requests

HTTP is the protocol (set of rules) that web browsers use to load websites. Your code
can use the same protocol to send data to a server and get data back.

There are different **methods** (types of requests):

- **GET** — "Give me some information." Like loading a web page. Example: fetching the
  list of available models.
- **POST** — "Here is some data, do something with it." Like submitting a form.
  Example: sending a prompt to Claude and getting a response.

Every HTTP request can include:

- **Headers** — Metadata about the request (like your ID badge). The most important
  header for this server is `Authorization`, which carries your secret token.
- **Body** — The actual data you are sending (only for POST requests). This server
  expects the body in JSON format.

Every HTTP response comes with a **status code** — a number that tells you what
happened:

| Code | Meaning | When You Will See It |
|------|---------|---------------------|
| **200** | OK — everything worked | Successful request |
| **400** | Bad Request — something is wrong with your data | Missing required fields |
| **401** | Unauthorized — invalid or missing token | Wrong API token |
| **403** | Forbidden — you are not allowed to do that | Trying to read a blocked file or run a blocked command |
| **404** | Not Found — that endpoint or file does not exist | Typo in the URL, or file does not exist |
| **422** | Validation Error — data format is wrong | Empty prompt, empty messages array |
| **502** | Bad Gateway — the AI backend had an error | Claude CLI crashed or returned an error |
| **504** | Gateway Timeout — the request took too long | Bash command exceeded 30-second limit |

### Bearer Token

A Bearer token is like a password that you include in every request. It proves you
are allowed to use the server. The word "Bearer" is just the type — you literally
send the header: `Authorization: Bearer your-secret-token-here`.

### JSON (JavaScript Object Notation)

JSON is a text format for representing structured data. It looks like this:

```json
{
  "name": "Alice",
  "age": 30,
  "hobbies": ["reading", "coding"]
}
```

Both requests to and responses from this server use JSON. If you have worked with
Python dictionaries or JavaScript objects, JSON will look familiar.

### Streaming (SSE — Server-Sent Events)

Normally, when you ask Claude a question, you wait until the entire answer is ready,
then you get it all at once. With **streaming**, the server sends the answer
word-by-word (or chunk-by-chunk) as Claude generates it — just like watching someone
type in real time.

SSE (Server-Sent Events) is the technical format used for streaming. Each chunk
arrives as a line starting with `data: ` followed by a JSON object.

### MCP (Model Context Protocol)

MCP is a standard that lets AI tools connect to each other. When this server exposes
an MCP endpoint, other AI tools (like another instance of Claude Code) can discover
and use this server's capabilities automatically — without you writing any custom
integration code.

### OpenAI Compatibility

Many tools and libraries (LangChain, Open WebUI, Continue.dev, n8n) were built to
work with OpenAI's API format. This server can **pretend to be an OpenAI API**, so
all those existing tools work with Claude without any code changes — just point them
at this server instead of OpenAI.

---

## 3. Architecture Overview

### How a Request Flows Through the System

```
You (curl, Python, n8n, OpenAI SDK, etc.)
 |
 |  HTTP request
 v
+-------------------------------------------+
|  FastAPI Server (localhost:8020)           |
|                                           |
|  1. Check Bearer Token (authentication)   |
|  2. Validate request data                 |
|  3. Check security boundaries             |
|     (file paths, bash commands)           |
|  4. Convert to Claude CLI command         |
|                                           |
|  Concurrency gate: max 2 at a time       |
+-------------------+-----------------------+
                    |
                    |  async subprocess
                    v
+-------------------------------------------+
|  Claude Code CLI                          |
|  claude -p "your prompt here"             |
|  --output-format json                     |
|  --model claude-sonnet-4-5-20250929       |
|  --no-session-persistence                 |
|                                           |
|  (Uses your Max subscription — no extra   |
|   cost per request)                       |
+-------------------+-----------------------+
                    |
                    |  JSON response
                    v
+-------------------------------------------+
|  FastAPI Server                           |
|                                           |
|  5. Parse Claude's JSON output            |
|  6. Convert to the right response format  |
|     - Custom format for /api/* endpoints  |
|     - OpenAI format for /v1/* endpoints   |
|  7. Send response back to you             |
+-------------------------------------------+
```

### What Each File Does

The server is small — only 6 source files:

| File | Purpose |
|------|---------|
| `app/main.py` | Creates the FastAPI app, registers routes, sets up logging and MCP |
| `app/config.py` | Loads settings from the `.env` file (token, model, paths, limits) |
| `app/auth.py` | Checks the Bearer token on every request |
| `app/routers/api.py` | All endpoint definitions — both custom REST and OpenAI-compatible |
| `app/services/claude_service.py` | Calls the Claude CLI as a subprocess, parses its output |
| `.env` | Your secret settings (token, paths, defaults) — not checked into git |

Supporting files:

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `test_server.py` | Automated test suite for all endpoints |
| `claude-api-server.service` | Example systemd service unit file |

### The 3 API Flavors

This server offers three different ways to interact with it:

1. **Custom REST API** (`/api/*`) — Purpose-built endpoints with specific schemas for
   chat, code analysis, file reading, and bash commands. Best when you want explicit
   control.

2. **OpenAI-Compatible API** (`/v1/*`) — Drop-in replacement for the OpenAI API.
   Best when you are using tools that already support OpenAI (LangChain, n8n, etc.).

3. **MCP Protocol** (`/mcp`) — Auto-generated from the REST API by the `fastapi-mcp`
   library. Best for connecting AI tools together.

---

## 4. Setup & Running

### Prerequisites

- **Python 3.12** or newer
- **Claude Code CLI** installed and authenticated with a Max subscription
  (the `claude` command must work in your terminal)
- **Git** (to clone the repository)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repository-url> claude-api-server
cd claude-api-server

# 2. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure the `.env` File

Create a `.env` file in the project root with these settings:

```bash
# REQUIRED: A secret token that protects your API.
# Generate one with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
API_SECRET_TOKEN=your-secret-token-here

# Path to the Claude CLI executable.
# Find yours with: which claude
CLAUDE_CLI_PATH=/home/your-username/.local/bin/claude

# Default AI model when none is specified in the request.
# Options: haiku, sonnet, opus
DEFAULT_MODEL=sonnet

# Maximum number of requests processed at the same time.
# Higher = more throughput, but each request uses CPU and memory.
MAX_CONCURRENT=2

# How long (in seconds) to wait for Claude to respond before giving up.
REQUEST_TIMEOUT=120

# Comma-separated list of directory paths that the file-read and analyze
# endpoints are allowed to access. Anything outside these paths is blocked.
ALLOWED_PATHS=/home/your-username
```

### Start the Server

```bash
# Make sure you are in the project directory with the venv activated
source venv/bin/activate

# Start the server (foreground — you will see logs in your terminal)
uvicorn app.main:app --host 127.0.0.1 --port 8020

# OR start in the background (logs go to a file)
nohup uvicorn app.main:app --host 127.0.0.1 --port 8020 > /tmp/claude-api-server.log 2>&1 &
```

### Verify It Works

```bash
curl http://localhost:8020/health
```

You should see:

```json
{
  "status": "ok",
  "default_model": "sonnet",
  "max_concurrent": 2
}
```

If you see `Connection refused`, the server is not running. Check the steps above.

---

## 5. Authentication

Every endpoint (except `/health`) requires a **Bearer token** in the request header.
This token must match the `API_SECRET_TOKEN` value in your `.env` file.

### How to Include the Token

Add this header to every request:

```
Authorization: Bearer your-secret-token-here
```

### Example with curl

```bash
# This works (valid token)
curl http://localhost:8020/api/models \
  -H "Authorization: Bearer your-secret-token-here"

# This fails with 403 (no token)
curl http://localhost:8020/api/models

# This fails with 401 (wrong token)
curl http://localhost:8020/api/models \
  -H "Authorization: Bearer wrong-token"
```

### What Happens Without a Valid Token

| Scenario | HTTP Status | Error Message |
|----------|-------------|---------------|
| No `Authorization` header | 403 | Not authenticated |
| Invalid token value | 401 | Invalid authentication token |

---

## 6. Available Models

This server provides access to three Claude AI models, each with different strengths:

| Short Name | Full Model ID | Best For |
|------------|--------------|----------|
| `haiku` | claude-haiku-4-5-20251001 | Fast responses, simple tasks, low latency |
| `sonnet` | claude-sonnet-4-5-20250929 | **Default.** Balanced performance and quality |
| `opus` | claude-opus-4-6 | Complex reasoning, highest quality output |

**How to choose:**
- Use **haiku** when speed matters more than depth (quick classifications, short
  answers, simple translations).
- Use **sonnet** for most tasks — it is the best balance of speed and quality.
- Use **opus** for complex analysis, nuanced writing, or difficult coding problems.

### OpenAI Model Aliases

If you are using a tool that sends OpenAI model names, the server automatically
translates them to Claude models:

| OpenAI Model Name | Maps To | Claude Model ID |
|-------------------|---------|-----------------|
| `gpt-4o` | opus | claude-opus-4-6 |
| `gpt-4` | opus | claude-opus-4-6 |
| `gpt-4-turbo` | opus | claude-opus-4-6 |
| `gpt-4o-mini` | sonnet | claude-sonnet-4-5-20250929 |
| `gpt-3.5-turbo` | haiku | claude-haiku-4-5-20251001 |

You can also use the full Claude model IDs directly (e.g., `claude-opus-4-6`) or
the short names (`opus`, `sonnet`, `haiku`). All three naming styles work everywhere.

---

## 7. API Reference — Custom REST

These endpoints live under `/api/` and use a custom request/response format designed
specifically for this server.

### Standard Response Format

All `/api/chat`, `/api/code`, and `/api/analyze` endpoints return this structure:

```json
{
  "result": "Claude's response text goes here",
  "model": "claude-sonnet-4-5-20250929",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 142,
    "cost_usd": 0
  },
  "duration_ms": 4523,
  "is_error": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | The AI's response text |
| `model` | string | Which model actually processed the request |
| `usage` | object | Token counts and cost (always $0 with Max subscription) |
| `duration_ms` | integer | How long the request took in milliseconds |
| `is_error` | boolean | `true` if something went wrong |

---

### `GET /health` — Health Check

**What it does:** Tells you if the server is running. This is the only endpoint that
does NOT require authentication.

**When to use it:** To verify the server is up before sending real requests, or for
monitoring/health-check scripts.

```bash
curl http://localhost:8020/health
```

**Response:**

```json
{
  "status": "ok",
  "default_model": "sonnet",
  "max_concurrent": 2
}
```

---

### `GET /api/models` — List Available Models

**What it does:** Returns the list of AI models you can use.

**When to use it:** To discover which models are available and which one is the
default.

```bash
curl http://localhost:8020/api/models \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**

```json
{
  "models": [
    { "key": "opus",   "model_id": "claude-opus-4-6",              "default": false },
    { "key": "sonnet", "model_id": "claude-sonnet-4-5-20250929",   "default": true  },
    { "key": "haiku",  "model_id": "claude-haiku-4-5-20251001",    "default": false }
  ]
}
```

---

### `POST /api/chat` — General-Purpose AI Chat

**What it does:** Send a prompt to Claude and get a text response. This is the most
basic and commonly used endpoint.

**When to use it:** Any general question, text generation, translation, summarization,
or analysis that does not need to read files on the server.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Your question or instruction (1–100,000 characters) |
| `model` | string | No | `haiku`, `sonnet`, or `opus` (default: `sonnet`) |
| `system_prompt` | string | No | Instructions that shape Claude's behavior |
| `max_budget_usd` | number | No | Spending cap per request (default: 1.0) |

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what a REST API is in 3 sentences.",
    "model": "haiku"
  }'
```

**Response:**

```json
{
  "result": "A REST API is a way for programs to communicate over the web using standard HTTP methods like GET and POST. Each URL (endpoint) represents a resource, and you interact with it by sending requests and receiving responses in a structured format like JSON. It is the most common way to build web services because it is simple, stateless, and works with any programming language.",
  "model": "claude-haiku-4-5-20251001",
  "usage": { "input_tokens": 18, "output_tokens": 73, "cost_usd": 0 },
  "duration_ms": 3200,
  "is_error": false
}
```

**Using a system prompt:**

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What should I eat for dinner?",
    "system_prompt": "You are a nutritionist. Always suggest healthy meals.",
    "model": "sonnet"
  }'
```

---

### `POST /api/code` — Code Generation & Analysis

**What it does:** Like `/api/chat`, but Claude gets access to tools: it can read
files, search codebases, and run commands on the server. This makes it much more
powerful for coding tasks.

**When to use it:** Code generation, code review, project analysis, or any task where
Claude needs to look at actual files on the server.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Your coding question or instruction |
| `model` | string | No | `haiku`, `sonnet`, or `opus` |
| `allowed_tools` | list | No | Which tools Claude can use (default: `["Read", "Glob", "Grep", "Bash"]`) |
| `max_budget_usd` | number | No | Spending cap per request |

```bash
curl -X POST http://localhost:8020/api/code \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Read the file requirements.txt in the current project and list the dependencies",
    "model": "haiku"
  }'
```

**Default tools:**

| Tool | What It Does |
|------|-------------|
| `Read` | Read file contents |
| `Glob` | Find files by pattern (e.g., `*.py`) |
| `Grep` | Search inside files for text patterns |
| `Bash` | Run shell commands |

---

### `POST /api/analyze` — Document & Data Analysis

**What it does:** Reads one or more files from the server and sends their contents to
Claude along with your prompt. Great for analyzing documents without having to
copy-paste their contents.

**When to use it:** Comparing files, reviewing logs, analyzing data files, or any task
where Claude needs to see file contents as context.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | What you want Claude to do with the files |
| `context_files` | list | No | List of file paths on the server to include |
| `model` | string | No | `haiku`, `sonnet`, or `opus` |
| `max_budget_usd` | number | No | Spending cap per request |

**Important:** All file paths in `context_files` must be within the allowed paths
(configured in `.env`). Paths outside the whitelist will be rejected with a 403 error.

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize what this project depends on and why each dependency is needed.",
    "context_files": ["/home/gslee/claude-api-server/requirements.txt"],
    "model": "haiku"
  }'
```

---

### `POST /api/tools/file-read` — Read a Server File

**What it does:** Returns the raw contents of a file on the server. No AI involved —
this is a direct file read.

**When to use it:** When you need to fetch a file's contents from the server without
AI processing (e.g., checking a config file, previewing a log).

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Absolute file path on the server |

```bash
curl -X POST http://localhost:8020/api/tools/file-read \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/gslee/claude-api-server/requirements.txt"}'
```

**Response:**

```json
{
  "path": "/home/gslee/claude-api-server/requirements.txt",
  "content": "fastapi>=0.115.0\nuvicorn[standard]>=0.34.0\npydantic-settings>=2.0.0\nfastapi-mcp>=0.3.0\npython-dotenv>=1.0.0\n",
  "size": 107
}
```

**Error cases:**
- Path outside the whitelist: **403** Forbidden
- File does not exist: **404** Not Found

---

### `POST /api/tools/bash` — Run a Whitelisted Shell Command

**What it does:** Executes a shell command on the server and returns the output. Only
a specific set of safe, read-only commands are allowed.

**When to use it:** Checking server status (disk space, memory, uptime), listing
files, searching for text in files.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | The shell command to run (1–5,000 characters) |

```bash
curl -X POST http://localhost:8020/api/tools/bash \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command": "df -h"}'
```

**Response:**

```json
{
  "stdout": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        50G   23G   25G  48% /\n",
  "stderr": "",
  "returncode": 0
}
```

**Allowed commands** (the command must start with one of these):

`ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `date`, `uptime`, `df`, `du`,
`free`, `whoami`, `pwd`, `echo`, `file`, `stat`, `python3 -c`, `python3 -m json.tool`

Any other command (like `rm`, `mv`, `curl`, `wget`, `apt`, `pip`) will be rejected
with a **403** error. The timeout for bash commands is **30 seconds**.

---

## 8. API Reference — OpenAI-Compatible

These endpoints mimic OpenAI's API format, so tools built for OpenAI work with this
server without code changes.

### Why This Exists

Many popular tools and libraries expect the OpenAI API format:

- **OpenAI Python SDK** (`openai` package)
- **LangChain** / **LlamaIndex**
- **Open WebUI**
- **Continue.dev** (VS Code AI assistant)
- **n8n** (workflow automation)

Instead of rewriting all these integrations, this server speaks the same language as
OpenAI. You just change the URL and API key.

---

### `GET /v1/models` — List Models (OpenAI Format)

**What it does:** Returns the model list in OpenAI's format.

```bash
curl http://localhost:8020/v1/models \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**

```json
{
  "object": "list",
  "data": [
    { "id": "claude-opus-4-6",              "object": "model", "created": 1700000000, "owned_by": "anthropic" },
    { "id": "claude-sonnet-4-5-20250929",   "object": "model", "created": 1700000000, "owned_by": "anthropic" },
    { "id": "claude-haiku-4-5-20251001",    "object": "model", "created": 1700000000, "owned_by": "anthropic" },
    { "id": "gpt-4o",                       "object": "model", "created": 1700000000, "owned_by": "anthropic", "parent": "claude-opus-4-6" },
    { "id": "gpt-4o-mini",                  "object": "model", "created": 1700000000, "owned_by": "anthropic", "parent": "claude-sonnet-4-5-20250929" },
    { "id": "gpt-3.5-turbo",               "object": "model", "created": 1700000000, "owned_by": "anthropic", "parent": "claude-haiku-4-5-20251001" }
  ]
}
```

---

### `POST /v1/chat/completions` — Chat Completions

**What it does:** The main endpoint for conversation. Supports multi-turn messages,
streaming, and tool calling — all in OpenAI's format.

#### Non-Streaming Example

```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "claude-sonnet-4-5-20250929",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 equals 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

#### Streaming Example

Add `"stream": true` to get the response word-by-word via Server-Sent Events:

```bash
curl -N -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "haiku",
    "messages": [{"role": "user", "content": "Tell me a short joke."}],
    "stream": true
  }'
```

**Response** (each line arrives as the AI generates text):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Why"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" did the"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" chicken..."},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":25,"total_tokens":37}}

data: [DONE]
```

The stream ends with `data: [DONE]`.

#### Using with the Python OpenAI SDK

```python
from openai import OpenAI

# Point the SDK at this server instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="YOUR_TOKEN",  # Your Bearer token from .env
)

# Non-streaming
response = client.chat.completions.create(
    model="sonnet",  # or "gpt-4o-mini" — both work
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators in simple terms."},
    ],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="haiku",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()  # newline at the end
```

#### Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8020/v1",
    api_key="YOUR_TOKEN",
    model="sonnet",
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

#### Tool Calling (Function Calling)

You can define tools that the AI can decide to call. The server translates Claude's
responses into OpenAI's `tool_calls` format.

```bash
curl -X POST http://localhost:8020/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonnet",
    "messages": [{"role": "user", "content": "What is the weather in Seoul?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "The city name"}
          },
          "required": ["city"]
        }
      }
    }]
  }'
```

When Claude decides to call a tool, the response includes `tool_calls` and the
`finish_reason` is `"tool_calls"` instead of `"stop"`:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_a1b2c3d4e5f6",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Seoul\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

You then execute the tool yourself, send the result back in a follow-up message with
`"role": "tool"`, and Claude will use that result to formulate a final answer.

---

## 9. Security Boundaries

This server has multiple layers of protection to prevent misuse.

### File Path Whitelist

The `/api/analyze` and `/api/tools/file-read` endpoints can only access files within
the paths listed in the `ALLOWED_PATHS` setting in your `.env` file.

**Default:** `/home/gslee` (only files under this directory are accessible)

Trying to read a file outside the whitelist returns a **403 Forbidden** error:

```bash
# This works (path is under /home/gslee)
curl -X POST http://localhost:8020/api/tools/file-read \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"path": "/home/gslee/claude-api-server/requirements.txt"}'

# This is BLOCKED (path is outside /home/gslee)
curl -X POST http://localhost:8020/api/tools/file-read \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"path": "/etc/shadow"}'
# → 403: "Path not allowed: /etc/shadow"
```

### Bash Command Whitelist

The `/api/tools/bash` endpoint only allows commands that start with one of these
safe prefixes:

| Category | Commands |
|----------|----------|
| File viewing | `ls`, `cat`, `head`, `tail`, `file`, `stat` |
| Search | `grep`, `find`, `wc` |
| System info | `date`, `uptime`, `df`, `du`, `free`, `whoami`, `pwd` |
| Output | `echo` |
| Python | `python3 -c`, `python3 -m json.tool` |

**Blocked examples:** `rm`, `mv`, `cp`, `curl`, `wget`, `apt`, `pip`, `kill`,
`chmod`, `chown`, `sudo` — anything that could modify the system.

### Concurrency Limit

Only **2 requests** can be processed at the same time. If a 3rd request arrives while
2 are already running, it waits in a queue until one finishes. This prevents the
server from being overwhelmed.

### Timeouts

| Request Type | Timeout |
|-------------|---------|
| AI requests (`/api/chat`, `/api/code`, `/api/analyze`, `/v1/chat/completions`) | **120 seconds** |
| Bash commands (`/api/tools/bash`) | **30 seconds** |

If a request exceeds its timeout, it is killed and returns an error.

### Network Binding

The server only listens on `127.0.0.1` (localhost). This means it is **not accessible
from other machines** unless you put a reverse proxy (like Nginx or Caddy) in front of
it. This is a deliberate security choice.

### Stateless Requests

Every request is completely independent. The server uses `--no-session-persistence`
when calling Claude, so there is no conversation history between requests. Each
request starts fresh.

---

## 10. Testing

The project includes a comprehensive test tool (`test_server.py`) that verifies all
endpoints work correctly.

### Quick Test (No AI Calls)

This runs all tests except those that actually call Claude. Finishes in about 2
seconds. Great for verifying the server is properly configured.

```bash
python3 test_server.py --skip-ai
```

### Full Test

Runs all tests including AI calls. Takes 1-2 minutes because each AI call needs a few
seconds.

```bash
python3 test_server.py
```

### Test a Specific Category

```bash
python3 test_server.py --only connectivity
python3 test_server.py --only auth
python3 test_server.py --only api
python3 test_server.py --only security
python3 test_server.py --only openai
python3 test_server.py --only mcp
```

### Test a Remote Server

```bash
python3 test_server.py --url http://remote-server:8020 --token your-token-here
```

### Verbose Output

See the full response bodies (helpful for debugging):

```bash
python3 test_server.py --verbose --only auth
```

### Test Categories

| Category | What It Tests |
|----------|--------------|
| **connectivity** | Health check endpoint, response time |
| **auth** | Valid token, missing token, wrong token |
| **api** | Model listing, chat, code, analyze, input validation |
| **security** | File read (allowed/blocked), bash (allowed/blocked) |
| **openai** | OpenAI model list, chat completions, streaming, validation |
| **mcp** | MCP SSE endpoint availability |

### Reading Test Output

The output uses color-coded results:
- **PASS** (green) — test succeeded
- **FAIL** (red) — test failed, something is wrong
- **SKIP** (yellow) — test was skipped (usually because `--skip-ai` was used)

At the end, you get a summary like: `Results: 12 passed, 0 failed, 4 skipped (16 total)`

---

## 11. Common Use Cases

### Simple Question with curl

```bash
curl -X POST http://localhost:8020/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of South Korea?", "model": "haiku"}'
```

### Using Python `requests` Library

```python
import requests

URL = "http://localhost:8020"
TOKEN = "your-secret-token-here"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# Simple chat
response = requests.post(f"{URL}/api/chat", headers=HEADERS, json={
    "prompt": "Explain what Docker is in one paragraph.",
    "model": "sonnet",
})
data = response.json()
print(data["result"])

# Read a file from the server
response = requests.post(f"{URL}/api/tools/file-read", headers=HEADERS, json={
    "path": "/home/gslee/claude-api-server/requirements.txt",
})
print(response.json()["content"])

# Run a server command
response = requests.post(f"{URL}/api/tools/bash", headers=HEADERS, json={
    "command": "uptime",
})
print(response.json()["stdout"])
```

### OpenAI SDK Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8020/v1",
    api_key="your-secret-token-here",
)

# Just change model names — everything else works like OpenAI
response = client.chat.completions.create(
    model="sonnet",  # or "gpt-4o-mini" — aliases work too
    messages=[
        {"role": "user", "content": "Write a Python function to reverse a string."},
    ],
)
print(response.choices[0].message.content)
```

### n8n Workflow Integration

In n8n, create an **OpenAI credential** with:
- **API Key:** your Bearer token from `.env`
- **Base URL:** `http://localhost:8020/v1`

Then use the standard **OpenAI node** in your workflows. It will automatically talk to
this server instead of OpenAI.

Alternatively, use an **HTTP Request node** for direct control:

- **Method:** POST
- **URL:** `http://localhost:8020/api/chat`
- **Authentication:** Header Auth → `Authorization: Bearer YOUR_TOKEN`
- **Body (JSON):**
  ```json
  {
    "prompt": "{{ $json.email_body }}",
    "system_prompt": "Classify this email as: urgent, normal, or spam. Reply with only the category.",
    "model": "haiku"
  }
  ```

### Analyzing a File on the Server

```bash
curl -X POST http://localhost:8020/api/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize this file and list any potential issues.",
    "context_files": ["/home/gslee/some-project/config.py"],
    "model": "sonnet"
  }'
```

### Multi-Turn Conversation with OpenAI SDK

Since each request is stateless, you manage conversation history yourself:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="YOUR_TOKEN")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
]

# Turn 1
messages.append({"role": "user", "content": "What is a Python list comprehension?"})
response = client.chat.completions.create(model="sonnet", messages=messages)
assistant_msg = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_msg})
print("Turn 1:", assistant_msg)

# Turn 2 (builds on Turn 1)
messages.append({"role": "user", "content": "Can you give me a more complex example?"})
response = client.chat.completions.create(model="sonnet", messages=messages)
print("Turn 2:", response.choices[0].message.content)
```

---

## 12. Troubleshooting

### "Connection refused"

**Cause:** The server is not running.

**Fix:**
```bash
# Check if the server process is running
ps aux | grep uvicorn

# If not running, start it
cd /home/gslee/claude-api-server
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8020
```

### 401 Unauthorized / 403 Forbidden

**Cause:** Your token is missing, wrong, or the header format is incorrect.

**Fix:** Make sure your `Authorization` header looks exactly like this:
```
Authorization: Bearer your-actual-token-from-env-file
```

Common mistakes:
- Forgetting the word `Bearer` before the token
- Extra spaces or newlines in the token
- Using a different token than what is in `.env`

### 502 Bad Gateway

**Cause:** The Claude CLI encountered an error. This usually means:
- Claude CLI is not installed
- Claude CLI is not authenticated (Max subscription expired or not set up)
- The prompt triggered a safety filter

**Fix:**
```bash
# Verify Claude CLI works
claude -p "Hello" --output-format json --model claude-haiku-4-5-20251001

# If it fails, re-authenticate
claude auth login
```

### Timeout Errors

**Cause:** The request took longer than 120 seconds (AI) or 30 seconds (bash).

**Fix:**
- Use a simpler prompt
- Use the `haiku` model (faster than `sonnet` or `opus`)
- For bash commands, make sure the command finishes quickly
- Increase `REQUEST_TIMEOUT` in `.env` if needed (and restart the server)

### 422 Validation Error

**Cause:** Your request data is malformed.

**Common causes:**
- Empty `prompt` in `/api/chat`, `/api/code`, or `/api/analyze`
- Empty `messages` array in `/v1/chat/completions`
- Missing required fields
- `prompt` exceeds 100,000 characters

### MCP Not Working (404 on /mcp)

**Cause:** The `fastapi-mcp` package is not installed.

**Fix:**
```bash
source venv/bin/activate
pip install fastapi-mcp
# Restart the server
```

### Server Logs

Check the server logs for detailed error information:

```bash
# If running in foreground: errors appear in your terminal

# If running in background:
tail -f /tmp/claude-api-server.log
```

---

## 13. Glossary

Quick reference for all technical terms used in this manual.

| Term | Definition |
|------|-----------|
| **API** | A set of rules that lets programs talk to each other over a network |
| **Bearer Token** | A secret string sent in the `Authorization` header to prove identity |
| **CLI** | Command-Line Interface — a program you run in the terminal |
| **CORS** | Cross-Origin Resource Sharing — browser security rules for cross-site requests |
| **curl** | A command-line tool for making HTTP requests |
| **Endpoint** | A specific URL path that the server responds to (e.g., `/api/chat`) |
| **FastAPI** | The Python web framework this server is built with |
| **Gateway** | A server that sits between a client and another service, translating between them |
| **HTTP** | HyperText Transfer Protocol — the rules for web communication |
| **JSON** | JavaScript Object Notation — a text format for structured data |
| **MCP** | Model Context Protocol — a standard for AI tools to connect to each other |
| **Middleware** | Code that runs on every request before it reaches the endpoint (e.g., logging) |
| **OpenAI SDK** | OpenAI's Python library for calling AI models |
| **REST** | Representational State Transfer — an architectural style for web APIs |
| **Semaphore** | A concurrency control that limits how many things run at the same time |
| **SSE** | Server-Sent Events — a protocol for streaming data from server to client |
| **Stateless** | Each request is independent; the server does not remember previous requests |
| **Subprocess** | A separate program launched by the server (here: the Claude CLI) |
| **Token (auth)** | A secret string that proves you are authorized to use the API |
| **Token (AI)** | A unit of text (~4 characters or ~3/4 of a word) used to measure AI usage |
| **Uvicorn** | The ASGI server that runs the FastAPI application |
| **Whitelist** | A list of explicitly allowed items; everything else is blocked |

---

*Last updated: 2026-02-10*
