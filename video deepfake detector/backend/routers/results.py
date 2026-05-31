"""
Results Router
================
Spec (bible §15):
  GET /api/v1/results/{video_id}                         → Full result JSON
  GET /api/v1/results/{video_id}/heatmaps/{frame_id}     → Heatmap image
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database       import get_session, ResultRecord
from backend.models.schemas import (
    AnalysisResult, OverallVerdict, SignalBreakdown,
    SuspiciousSegment, ExplainabilityInfo, VideoMetadata,
    RiskLevel, AnalysisDepth,
)

logger = logging.getLogger(__name__)
router = APIRouter()

ROOT       = Path(__file__).parent.parent.parent
HEATMAP_DIR = ROOT / "outputs" / "heatmaps"


@router.get("/results/{video_id}", response_model=AnalysisResult)
async def get_result(
    video_id: str,
    session:  AsyncSession = Depends(get_session),
):
    """
    Retrieve the full analysis result for a completed video job.

    Returns the complete JSON matching bible §10.2 output schema including:
    - Overall verdict + risk level
    - Per-signal breakdown
    - Frame-level suspicion timeline
    - Suspicious segments with timestamps
    - Explainability evidence
    """
    result = await session.execute(
        select(ResultRecord).where(ResultRecord.video_id == video_id)
    )
    rec = result.scalar_one_or_none()

    if not rec:
        raise HTTPException(
            status_code = 404,
            detail      = f"No result found for video_id={video_id}. "
                          f"Job may still be processing — poll /api/v1/jobs/{video_id}",
        )

    # Reconstruct Pydantic models from JSON columns
    overall = OverallVerdict(
        fake_probability    = rec.fake_probability or 0.0,
        realness_score      = rec.realness_score   or 1.0,
        uncertainty         = rec.uncertainty      or 0.0,
        risk_level          = RiskLevel(rec.risk_level or "AUTHENTIC"),
        verdict             = rec.verdict or "",
        confidence_interval = (rec.explainability or {}).get(
            "confidence_interval", {"lower": 0.0, "upper": 0.0}
        ),
    )

    sb = rec.signal_breakdown or {}
    signal_breakdown = SignalBreakdown(
        spatial_xception = sb.get("spatial_xception"),
        spatial_vit      = sb.get("spatial_vit"),
        frequency_domain = sb.get("frequency_domain"),
        temporal_lstm    = sb.get("temporal_lstm"),
        physiological    = sb.get("physiological"),
        lip_sync         = sb.get("lip_sync"),
        optical_flow     = sb.get("optical_flow"),
    )

    segs = [
        SuspiciousSegment(**s)
        for s in (rec.suspicious_segments or [])
    ]

    xai = rec.explainability or {}
    explainability = ExplainabilityInfo(
        top_evidence  = xai.get("top_evidence", []),
        risk_factors  = xai.get("risk_factors", []),
        heatmap_urls  = [
            f"/api/v1/results/{video_id}/heatmaps/{i}"
            for i in range(len(rec.heatmap_paths or []))
        ],
        top_regions = xai.get("top_regions", []),
    )

    vm = rec.video_metadata
    video_metadata = VideoMetadata(**vm) if vm else None

    return AnalysisResult(
        video_id            = rec.video_id,
        job_id              = rec.job_id,
        overall             = overall,
        signal_breakdown    = signal_breakdown,
        frame_scores        = rec.frame_scores        or [],
        timestamps          = rec.timestamps          or [],
        suspicious_segments = segs,
        explainability      = explainability,
        video_metadata      = video_metadata,
        analyzed_at         = rec.analyzed_at,
        analysis_depth      = AnalysisDepth(rec.analysis_depth or "standard"),
    )


@router.get("/results/{video_id}/heatmaps/{frame_id}")
async def get_heatmap(video_id: str, frame_id: int):
    """
    Return a Grad-CAM heatmap image for a specific frame.
    Used by the frontend heatmap gallery.
    """
    heatmap_path = HEATMAP_DIR / video_id / f"frame_{frame_id:04d}.jpg"

    if not heatmap_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"Heatmap frame {frame_id} not found for video {video_id}",
        )

    return FileResponse(
        path         = str(heatmap_path),
        media_type   = "image/jpeg",
        filename     = f"heatmap_frame_{frame_id}.jpg",
    )
