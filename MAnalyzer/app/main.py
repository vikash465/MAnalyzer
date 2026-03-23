import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pydantic import BaseModel, Field

from app.config import get_settings
from app.routers import report, analysis
from app.services.session import cleanup_expired_sessions
from app.services.crypto import encrypt_value

settings = get_settings()

# ── Logging ──

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("manalyzer")

# ── Rate limiter ──

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ── Lifespan (background cleanup) ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _session_cleanup_loop():
        while True:
            await asyncio.sleep(300)
            removed = cleanup_expired_sessions()
            if removed:
                logger.info("Cleaned up %d expired sessions", removed)

    task = asyncio.create_task(_session_cleanup_loop())
    logger.info("MAnalyzer started")
    yield
    task.cancel()

# ── App ──

app = FastAPI(
    title=settings.app_name,
    description="Upload medical reports (lab, radiology, pathology) and analyze them with AI.",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)
app.state.limiter = limiter


# ── Global exception handlers ──

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ── Auth middleware ──

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if settings.auth_enabled:
        public_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/api/encrypt-key"}
        if (
            not request.url.path.startswith("/static")
            and request.url.path not in public_paths
        ):
            token = request.headers.get("X-API-Key", "")
            if token != settings.api_key.get_secret_value():
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
    return await call_next(request)


# ── CORS ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True if settings.cors_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──

app.include_router(report.router)
app.include_router(analysis.router)

# ── Static files ──

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    return index.read_text(encoding="utf-8")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "auth_enabled": settings.auth_enabled,
    }


class EncryptKeyRequest(BaseModel):
    key: str = Field(..., min_length=10, max_length=200)
    provider: str = Field(..., pattern=r"^(openai|claude)$")


@app.post("/api/encrypt-key")
async def encrypt_api_key(req: EncryptKeyRequest):
    token = encrypt_value(f"{req.provider}:{req.key}")
    return {"token": token, "provider": req.provider}
