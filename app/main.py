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
