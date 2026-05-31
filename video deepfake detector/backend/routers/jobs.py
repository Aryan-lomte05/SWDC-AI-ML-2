"""
Jobs Router — Status Polling + WebSocket
==========================================
Spec (bible §15):
  GET /api/v1/jobs/{job_id}    → Poll status
  WS  /ws/jobs/{job_id}        → Real-time progress stream
"""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database       import get_session, JobRecord
from backend.models.schemas import JobStatusResponse, ProcessingStage

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory progress store (job_id → latest update dict)
# In production this would be Redis pub/sub
_progress_store: dict = {}


def update_job_progress(job_id: str, update: dict):
    """Called by the task runner to push progress updates."""
    _progress_store[job_id] = update


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id:  str,
    session: AsyncSession = Depends(get_session),
):
    """Poll job status by job_id."""
    result = await session.execute(
        select(JobRecord).where(JobRecord.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id     = job.id,
        status     = job.status,
        stage      = job.stage,
        progress   = job.progress,
        message    = job.message,
        created_at = job.created_at,
        updated_at = job.updated_at,
        error      = job.error,
    )


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.
    Spec (bible §15): streams stage, progress, partial_results.

    Protocol:
      Client connects → server sends updates as JSON every 0.5s
      until job is COMPLETED or FAILED, then closes.
    """
    await websocket.accept()
    logger.info("WebSocket connected: job_id=%s", job_id)

    last_sent = None

    try:
        while True:
            update = _progress_store.get(job_id)

            if update and update != last_sent:
                await websocket.send_json(update)
                last_sent = update

                # Close if terminal state reached
                status = update.get("status", "")
                if status in ("completed", "failed"):
                    await asyncio.sleep(0.5)
                    break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: job_id=%s", job_id)
    except Exception as e:
        logger.error("WebSocket error for job %s: %s", job_id, e)
        try:
            await websocket.send_json({"error": str(e), "status": "failed"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        # Cleanup progress store
        _progress_store.pop(job_id, None)
