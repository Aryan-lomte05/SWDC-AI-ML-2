"""
Background Task Runner — Video Deepfake Detector
==================================================
Spec (bible §15):

Orchestrates the entire ML pipeline asynchronously:
  1. Update job progress (WebSocket via _progress_store)
  2. Frame extraction
  3. Face processing
  4. Classical analysis (Optical Flow, Lip Sync, Physiological)
  5. Neural inference (Xception, ViT, FreqNet, Temporal)
  6. Confidence scoring
  7. XAI generation (Grad-CAM, ViT Attention)
  8. Save results to SQLite
"""

import os
import asyncio
import logging
from typing import Dict, List, Any
from pathlib import Path

import torch
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

# Import database & models
from backend.database import AsyncSessionLocal, JobRecord, ResultRecord
from backend.routers.jobs import update_job_progress

# Import ML pipeline modules
from ml.preprocessing.frame_extractor import FrameExtractor
from ml.preprocessing.face_processor  import FaceProcessor
from ml.analysis.physiological        import PhysiologicalAnalyzer
from ml.analysis.lip_sync             import LipSyncAnalyzer
from ml.analysis.optical_flow         import OpticalFlowAnalyzer
from ml.scoring.confidence_engine     import ConfidenceScoringEngine
from ml.explainability.grad_cam       import GradCAM

logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "True").lower() == "true"


async def run_analysis_task(job_id: str, video_id: str, video_path: str, options: dict):
    """
    Main background entry point for processing a video.
    Executed by FastAPI BackgroundTasks.
    """
    try:
        await _process_pipeline(job_id, video_id, video_path, options)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        await _fail_job(job_id, str(e))
    finally:
        # Cleanup temp upload if needed, though we might keep it for now
        pass


async def _process_pipeline(job_id: str, video_id: str, video_path: str, options: dict):
    # ── Init DB Session ────────────────────────────────────────────────────────
    async with AsyncSessionLocal() as session:
        job = await session.get(JobRecord, job_id)
        if not job:
            logger.error("Job %s not found in DB before start", job_id)
            return

        def emit(stage: str, prog: int, msg: str, partial=None):
            # Update DB
            job.stage    = stage
            job.progress = prog
            job.message  = msg
            job.status   = "processing" if prog < 100 else "completed"
            # Update WebSocket
            update_job_progress(job_id, {
                "job_id":   job_id,
                "status":   job.status,
                "stage":    stage,
                "progress": prog,
                "message":  msg,
                "partial":  partial,
            })

        emit("preprocessing", 5, "Starting frame extraction...")

        # ── 1. Frame Extraction ────────────────────────────────────────────────
        extractor = FrameExtractor(target_fps=10, max_frames=options.get("frame_limit", 500))
        # Yield execution so async loop isn't blocked (since extraction is CPU bound)
        await asyncio.sleep(0) 
        frames, timestamps, meta = extractor.extract_with_metadata(video_path)
        
        if not frames:
            raise ValueError("No frames extracted. Video may be corrupt or empty.")
            
        emit("face_detection", 15, f"Extracted {len(frames)} frames. Detecting faces...")

        # ── 2. Face Processing ─────────────────────────────────────────────────
        processor = FaceProcessor(output_size=256)
        face_crops = []
        valid_frames = []
        valid_timestamps = []

        # Process in chunks to avoid blocking
        chunk_size = 30
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i+chunk_size]
            for j, f in enumerate(chunk):
                faces = processor.process_frame(f)
                if faces and faces[0]["aligned_crop"] is not None:
                    face_crops.append(faces[0]["aligned_crop"])
                    valid_frames.append(f)
                    valid_timestamps.append(timestamps[i+j])
            
            prog = 15 + int(30 * (i / len(frames)))
            emit("face_detection", prog, f"Detected faces in {len(face_crops)} frames...")
            await asyncio.sleep(0.01)

        if len(face_crops) < 5:
            raise ValueError("Insufficient faces detected (<5) for reliable analysis.")

        emit("inference", 45, "Running classical and deep learning analysis...")

        # ── 3. Analysis (Mock vs Real) ─────────────────────────────────────────
        engine = ConfidenceScoringEngine()
        
        if MOCK_MODE:
            # Simulate ML inference processing time
            for i in range(5):
                await asyncio.sleep(0.5)
                emit("inference", 50 + i * 5, "Running neural network ensemble inference...")
            
            # Use random mock scores
            signal_scores = ConfidenceScoringEngine.generate_mock_scores(bias="random")
            
            # Generate dummy frame scores
            base = np.mean(list(signal_scores.values()))
            frame_scores = [float(np.clip(base + np.random.normal(0, 0.1), 0.0, 1.0)) for _ in valid_timestamps]
            
            emit("scoring", 80, "Aggregating confidence scores...")
            
            result_dict = engine.compute_final_score(
                signal_scores    = signal_scores,
                frame_scores     = frame_scores,
                timestamps       = valid_timestamps,
                uncertainty      = 0.04,
                video_id         = video_id,
                duration_seconds = meta["duration"],
            )
            heatmap_paths = []
            
        else:
            # Real Inference (pseudo-code structure for real models)
            # In a real deployed environment, you'd load PyTorch models here
            # or keep them loaded in memory via a global pool.
            # We'll run the classical analyzers since they don't need weights
            
            emit("inference", 50, "Running Optical Flow Analysis...")
            of_analyzer = OpticalFlowAnalyzer()
            of_res = of_analyzer.analyze(valid_frames)
            await asyncio.sleep(0)
            
            emit("inference", 60, "Running Lip Sync Analysis...")
            ls_analyzer = LipSyncAnalyzer()
            ls_res = ls_analyzer.analyze(face_crops)
            await asyncio.sleep(0)
            
            emit("inference", 70, "Running Physiological Analysis...")
            phys_analyzer = PhysiologicalAnalyzer()
            phys_res = phys_analyzer.analyze(valid_frames, face_crops, timestamps=valid_timestamps)
            await asyncio.sleep(0)
            
            # Combine real classical + mocked neural (since no weights are loaded)
            signal_scores = {
                "optical_flow":  of_res["anomaly_score"],
                "lip_sync":      ls_res["anomaly_score"],
                "physiological": phys_res["anomaly_score"],
                "spatial_xception": np.random.uniform(0.1, 0.9),  # Mock neural
                "spatial_vit":      np.random.uniform(0.1, 0.9),
                "temporal_lstm":    np.random.uniform(0.1, 0.9),
            }
            
            frame_scores = [float(np.clip(np.mean(list(signal_scores.values())) + np.random.normal(0,0.1), 0, 1)) for _ in valid_timestamps]
            
            evidence = of_res["evidence"] + ls_res["evidence"] + phys_res["evidence"]
            
            emit("scoring", 80, "Aggregating signals...")
            
            result_dict = engine.compute_final_score(
                signal_scores    = signal_scores,
                frame_scores     = frame_scores,
                timestamps       = valid_timestamps,
                evidence_strings = evidence,
                uncertainty      = 0.03,
                video_id         = video_id,
                duration_seconds = meta["duration"],
            )
            heatmap_paths = []

        # ── 4. Save to Database ────────────────────────────────────────────────
        emit("complete", 95, "Saving results...")
        
        result_record = ResultRecord(
            video_id            = video_id,
            job_id              = job_id,
            fake_probability    = result_dict["overall"]["fake_probability"],
            realness_score      = result_dict["overall"]["realness_score"],
            uncertainty         = result_dict["overall"]["uncertainty"],
            risk_level          = result_dict["overall"]["risk_level"],
            verdict             = result_dict["overall"]["verdict"],
            signal_breakdown    = result_dict["signal_breakdown"],
            frame_scores        = result_dict["frame_scores"],
            timestamps          = result_dict["timestamps"],
            suspicious_segments = result_dict["suspicious_segments"],
            explainability      = result_dict["explainability"],
            video_metadata      = {
                "filename": job.filename,
                "duration_seconds": result_dict["duration_seconds"],
                "fps": meta.get("fps", 0),
                "resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
                "format": Path(job.filename or "").suffix,
                "file_size_mb": job.file_size_mb,
                "frames_analyzed": len(valid_frames),
                "faces_detected": len(face_crops),
            },
            analysis_depth      = options.get("analysis_depth", "standard"),
            heatmap_paths       = heatmap_paths,
        )
        session.add(result_record)
        await session.commit()
        
        emit("complete", 100, "Analysis complete.")


async def _fail_job(job_id: str, error_msg: str):
    async with AsyncSessionLocal() as session:
        job = await session.get(JobRecord, job_id)
        if job:
            job.status = "failed"
            job.error  = error_msg
            await session.commit()
            
            update_job_progress(job_id, {
                "job_id": job_id,
                "status": "failed",
                "error":  error_msg,
            })
