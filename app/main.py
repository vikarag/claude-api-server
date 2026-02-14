import logging
import shutil
import time
import uuid

from fastapi import FastAPI, Request

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
    version="2.1.0",
)

# Include REST API router
app.include_router(api.router)
app.include_router(api.openai_router)


# CORS middleware (off by default)
if settings.enable_cors:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", settings.cors_origins)


# Health check (no auth)
@app.get("/health")
async def health():
    cli_found = shutil.which(settings.claude_cli_path) is not None
    return {
        "status": "ok" if cli_found else "degraded",
        "claude_cli_found": cli_found,
        "default_model": settings.default_model,
        "max_concurrent": settings.max_concurrent,
    }


# Request logging middleware with request ID tracking
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    duration = int((time.monotonic() - start) * 1000)
    logger.info(
        "[%s] %s %s -> %d (%dms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    response.headers["X-Request-ID"] = request_id
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
