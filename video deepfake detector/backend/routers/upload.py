"""
Upload Router
==============
Spec (bible §15): POST /api/v1/videos/upload
"""

import uuid
import os
import math
import logging
from pathlib import Path
from datetime import datetime

import aiofiles
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database      import get_session, JobRecord, ResultRecord
from backend.models.schemas import AnalysisOptions, UploadResponse, JobStatus, AnalysisDepth
from backend.workers.task_runner import run_analysis_task

logger = logging.getLogger(__name__)
router = APIRouter()

ROOT       = Path(__file__).parent.parent.parent
UPLOAD_DIR = ROOT / "uploads"
MAX_MB     = int(os.getenv("MAX_UPLOAD_MB", "500"))

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEPTH_TIME_EST     = {"fast": 8, "standard": 20, "deep": 35}


@router.post("/videos/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    analysis_depth:   str        = Form(default="standard"),
    include_heatmaps: bool       = Form(default=True),
    session:          AsyncSession = Depends(get_session),
):
    """
    Upload a video for deepfake analysis.

    - **file**: Video file (mp4/mov/avi/mkv/webm, max 500MB)
    - **analysis_depth**: fast | standard | deep
    - **include_heatmaps**: Generate Grad-CAM heatmap images
    """
    # ── Validate ───────────────────────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code = 422,
            detail      = f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # ── Save file ──────────────────────────────────────────────────────────
    video_id = str(uuid.uuid4())
    save_dir = UPLOAD_DIR / video_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"video{ext}"

    total_mb = 0.0
    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):    # 1 MB chunks
            total_mb += len(chunk) / (1024 * 1024)
            if total_mb > MAX_MB:
                raise HTTPException(
                    status_code = 413,
                    detail      = f"File exceeds {MAX_MB}MB limit",
                )
            await f.write(chunk)

    # ── Create job record ──────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    options = {
        "analysis_depth":  analysis_depth,
        "include_heatmaps": include_heatmaps,
    }

    job = JobRecord(
        id           = job_id,
        video_id     = video_id,
        status       = JobStatus.QUEUED,
        stage        = "queued",
        progress     = 0,
        message      = "Job queued for processing",
        created_at   = datetime.utcnow(),
        updated_at   = datetime.utcnow(),
        filename     = file.filename,
        file_size_mb = round(total_mb, 2),
        options_json = options,
    )
    session.add(job)
    await session.commit()

    logger.info(
        "Video uploaded | job_id=%s | video_id=%s | size=%.1fMB | depth=%s",
        job_id, video_id, total_mb, analysis_depth,
    )

    # ── Queue background task ──────────────────────────────────────────────
    background_tasks.add_task(
        run_analysis_task,
        job_id    = job_id,
        video_id  = video_id,
        video_path= str(save_path),
        options   = options,
    )

    est_time = DEPTH_TIME_EST.get(analysis_depth, 20)

    return UploadResponse(
        job_id                  = job_id,
        status                  = JobStatus.QUEUED,
        estimated_time_seconds  = est_time,
        websocket_url           = f"/ws/jobs/{job_id}",
    )
