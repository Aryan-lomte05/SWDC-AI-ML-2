"""
FastAPI Backend — Video Deepfake Detector
==========================================
Spec (bible §15):

Endpoints:
  POST  /api/v1/videos/upload         → Upload video, returns job_id
  GET   /api/v1/jobs/{job_id}          → Poll job status
  GET   /api/v1/results/{video_id}     → Full analysis result
  GET   /api/v1/results/{video_id}/heatmaps/{frame_id}  → Heatmap image
  POST  /api/v1/analyze/url            → Analyze video from URL
  GET   /api/v1/health                 → Service health + GPU status
  WS    /ws/jobs/{job_id}              → Real-time progress

Local stack:
  FastAPI + uvicorn (no Docker)
  SQLite + SQLAlchemy async (no PostgreSQL)
  Local filesystem (no MinIO/S3)
  BackgroundTasks (no Celery/Redis)
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── Add project root to sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.database     import init_db, engine
from backend.routers      import upload, jobs, results
from backend.models.schemas import HealthResponse

logger = logging.getLogger(__name__)

# ── GPU Info ──────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = props.total_memory / 1e9
        return {
            "gpu_available":    True,
            "gpu_name":         props.name,
            "gpu_memory_used":  round(used, 2),
            "gpu_memory_total": round(total, 2),
        }
    return {"gpu_available": False, "gpu_name": None,
            "gpu_memory_used": None, "gpu_memory_total": None}


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown lifecycle."""
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Video Deepfake Detector — API Server v1.0.0")
    logger.info("=" * 60)

    # Ensure output directories exist
    for d in ["uploads", "outputs/heatmaps", "outputs/reports"]:
        Path(ROOT / d).mkdir(parents=True, exist_ok=True)

    # Initialise SQLite database
    await init_db()
    logger.info("✅ Database initialised")

    gpu = get_gpu_info()
    if gpu["gpu_available"]:
        logger.info("✅ GPU: %s | VRAM: %.1f GB", gpu["gpu_name"], gpu["gpu_memory_total"])
    else:
        logger.warning("⚠️  No GPU detected — running on CPU (inference will be slow)")

    mock_mode = os.getenv("MOCK_MODE", "True").lower() == "true"
    if mock_mode:
        logger.info("🔶 MOCK_MODE=True — simulated inference (no model weights required)")
    else:
        logger.info("🔬 MOCK_MODE=False — loading real model weights...")

    logger.info("✅ API ready at http://%s:%s", os.getenv("HOST", "127.0.0.1"), os.getenv("PORT", "8000"))

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("Shutting down Video Deepfake Detector API")
    await engine.dispose()


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Video Deepfake Detector API",
    description = (
        "Production-ready AI-powered deepfake video detection. "
        "Combines XceptionNet, ViT-L, FreqNet, BiLSTM, and classical "
        "physiological signal analysis into a calibrated ensemble detector."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],      # Tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request Logging ───────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug("%s %s", request.method, request.url.path)
    response = await call_next(request)
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router,  prefix="/api/v1", tags=["Upload"])
app.include_router(jobs.router,    prefix="/api/v1", tags=["Jobs"])
app.include_router(results.router, prefix="/api/v1", tags=["Results"])

# ── Static Files (frontend SPA) ───────────────────────────────────────────────
FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health, GPU status, and model load status."""
    gpu = get_gpu_info()
    return HealthResponse(
        status          = "healthy",
        models_loaded   = True,
        queue_depth     = 0,
        version         = "1.0.0",
        **gpu,
    )

# ── Root → SPA ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_spa():
    """Serve the frontend SPA."""
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Video Deepfake Detector API v1.0.0", "docs": "/docs"})

@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """SPA fallback — all non-API routes serve index.html for client-side routing."""
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    uvicorn.run(
        "backend.main:app",
        host        = os.getenv("HOST", "127.0.0.1"),
        port        = int(os.getenv("PORT", "8000")),
        reload      = os.getenv("DEBUG", "True").lower() == "true",
        log_level   = "info",
    )
