"""
SQLite Database — Async SQLAlchemy
=====================================
Local database for jobs and results (no Docker/PostgreSQL needed).
"""

import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm            import declarative_base, sessionmaker

ROOT     = Path(__file__).parent.parent
DB_PATH  = ROOT / "deepfake_detector.db"
DB_URL   = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DB_URL,
    echo       = False,
    connect_args = {"check_same_thread": False},
)

AsyncSessionLocal = sessionmaker(
    bind       = engine,
    class_     = AsyncSession,
    expire_on_commit = False,
)

Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class JobRecord(Base):
    __tablename__ = "jobs"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id     = Column(String, unique=True, nullable=False)
    status       = Column(String, default="queued")
    stage        = Column(String, nullable=True)
    progress     = Column(Integer, default=0)
    message      = Column(String, default="")
    error        = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filename     = Column(String, nullable=True)
    file_size_mb = Column(Float, nullable=True)
    options_json = Column(JSON, nullable=True)


class ResultRecord(Base):
    __tablename__ = "results"

    video_id            = Column(String, primary_key=True)
    job_id              = Column(String, nullable=False)
    fake_probability    = Column(Float, nullable=True)
    realness_score      = Column(Float, nullable=True)
    uncertainty         = Column(Float, nullable=True)
    risk_level          = Column(String, nullable=True)
    verdict             = Column(Text, nullable=True)
    signal_breakdown    = Column(JSON, nullable=True)
    frame_scores        = Column(JSON, nullable=True)
    timestamps          = Column(JSON, nullable=True)
    suspicious_segments = Column(JSON, nullable=True)
    explainability      = Column(JSON, nullable=True)
    video_metadata      = Column(JSON, nullable=True)
    analyzed_at         = Column(DateTime, default=datetime.utcnow)
    analysis_depth      = Column(String, default="standard")
    heatmap_paths       = Column(JSON, nullable=True)


# ── Init ──────────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session
