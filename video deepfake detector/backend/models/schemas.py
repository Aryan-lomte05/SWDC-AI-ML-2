"""
Pydantic v2 Schemas — API Request/Response Models
====================================================
Spec (bible §15): Full API schema matching all endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class AnalysisDepth(str, Enum):
    FAST     = "fast"       # XceptionNet only, ~5s
    STANDARD = "standard"   # All spatial + temporal, ~15s
    DEEP     = "deep"       # Full ensemble + XAI + MC Dropout, ~30s


class JobStatus(str, Enum):
    QUEUED      = "queued"
    PROCESSING  = "processing"
    COMPLETED   = "completed"
    FAILED      = "failed"


class RiskLevel(str, Enum):
    AUTHENTIC   = "AUTHENTIC"
    LOW_RISK    = "LOW_RISK"
    SUSPICIOUS  = "SUSPICIOUS"
    HIGH_RISK   = "HIGH_RISK"
    FAKE        = "FAKE"


class ProcessingStage(str, Enum):
    UPLOADING       = "uploading"
    PREPROCESSING   = "preprocessing"
    FACE_DETECTION  = "face_detection"
    INFERENCE       = "inference"
    SCORING         = "scoring"
    XAI             = "xai"
    COMPLETE        = "complete"


# ── Request Schemas ───────────────────────────────────────────────────────────

class AnalysisOptions(BaseModel):
    analysis_depth:          AnalysisDepth = AnalysisDepth.STANDARD
    include_heatmaps:        bool          = True
    include_physiological:   bool          = True
    include_lip_sync:        bool          = True
    include_optical_flow:    bool          = True
    include_frequency:       bool          = True
    mc_dropout_passes:       int           = Field(default=20, ge=1, le=50)
    frame_limit:             int           = Field(default=500, ge=10, le=500)


class AnalyzeUrlRequest(BaseModel):
    url:     str
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


# ── Response Schemas ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id:                str
    status:                JobStatus
    estimated_time_seconds: int
    websocket_url:         str


class JobStatusResponse(BaseModel):
    job_id:       str
    status:       JobStatus
    stage:        Optional[ProcessingStage] = None
    progress:     int = Field(default=0, ge=0, le=100)
    message:      str = ""
    created_at:   datetime
    updated_at:   datetime
    error:        Optional[str] = None


class SignalBreakdown(BaseModel):
    spatial_xception:  Optional[float] = None
    spatial_vit:       Optional[float] = None
    frequency_domain:  Optional[float] = None
    temporal_lstm:     Optional[float] = None
    physiological:     Optional[float] = None
    lip_sync:          Optional[float] = None
    optical_flow:      Optional[float] = None


class SuspiciousSegment(BaseModel):
    start_time: str   # "HH:MM:SS"
    end_time:   str
    score:      float
    reason:     str


class OverallVerdict(BaseModel):
    fake_probability:     float = Field(ge=0.0, le=1.0)
    realness_score:       float = Field(ge=0.0, le=1.0)
    uncertainty:          float = Field(ge=0.0)
    risk_level:           RiskLevel
    verdict:              str
    confidence_interval:  Dict[str, float]


class ExplainabilityInfo(BaseModel):
    top_evidence:    List[str]      = Field(default_factory=list)
    risk_factors:    List[Dict]     = Field(default_factory=list)
    heatmap_urls:    List[str]      = Field(default_factory=list)
    top_regions:     List[List]     = Field(default_factory=list)


class VideoMetadata(BaseModel):
    filename:        str
    duration_seconds: float
    fps:             float
    resolution:      str
    format:          str
    file_size_mb:    float
    frames_analyzed: int
    faces_detected:  int


class AnalysisResult(BaseModel):
    video_id:            str
    job_id:              str
    overall:             OverallVerdict
    signal_breakdown:    SignalBreakdown
    frame_scores:        List[float]            = Field(default_factory=list)
    timestamps:          List[float]            = Field(default_factory=list)
    suspicious_segments: List[SuspiciousSegment] = Field(default_factory=list)
    explainability:      ExplainabilityInfo
    video_metadata:      Optional[VideoMetadata] = None
    analyzed_at:         datetime                = Field(default_factory=datetime.utcnow)
    analysis_depth:      AnalysisDepth           = AnalysisDepth.STANDARD
    model_version:       str                     = "1.0.0"


class HealthResponse(BaseModel):
    status:          str
    gpu_available:   bool
    gpu_name:        Optional[str]    = None
    gpu_memory_used: Optional[float]  = None   # GB
    gpu_memory_total: Optional[float] = None   # GB
    models_loaded:   bool
    queue_depth:     int
    version:         str


# ── WebSocket progress update ─────────────────────────────────────────────────

class ProgressUpdate(BaseModel):
    job_id:          str
    stage:           ProcessingStage
    progress:        int = Field(ge=0, le=100)
    message:         str
    partial_results: Optional[Dict[str, Any]] = None
