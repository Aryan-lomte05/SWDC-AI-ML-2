"""
Frame Extractor
================
Extracts frames from a video at a target FPS, with scene-cut detection
to avoid flagging legitimate discontinuities as temporal anomalies.

Spec (bible §13 Step 2):
  - Full video: sample at 10 FPS
  - Scene cuts: always include boundary frames
  - Hard cap: 500 frames per video (speed constraint)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

TARGET_FPS  = 10
MAX_FRAMES  = 500
SCENE_THRESHOLD = 35.0   # Histogram diff threshold for scene cut detection


class FrameExtractor:
    """
    Extracts frames from a video file at a uniform target FPS,
    with optional scene-cut boundary injection.

    Args:
        target_fps       (int):   Frames per second to extract.
        max_frames       (int):   Maximum frames to return.
        scene_threshold  (float): Pixel-diff threshold for scene cut detection.
        resize           (tuple): Optional (W, H) to resize frames.
    """

    def __init__(
        self,
        target_fps:      int   = TARGET_FPS,
        max_frames:      int   = MAX_FRAMES,
        scene_threshold: float = SCENE_THRESHOLD,
        resize:          Optional[Tuple[int, int]] = None,
    ):
        self.target_fps      = target_fps
        self.max_frames      = max_frames
        self.scene_threshold = scene_threshold
        self.resize          = resize

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames + timestamps from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            frames     : List of BGR numpy arrays  [H, W, 3]
            timestamps : List of float seconds matching each frame
        """
        video_path = str(video_path)
        cap        = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, round(original_fps / self.target_fps))

        logger.info(
            "Extracting from %s | orig_fps=%.1f | interval=%d | total_frames=%d",
            Path(video_path).name, original_fps, frame_interval, total_frames,
        )

        frames, timestamps, scene_cuts = [], [], []
        prev_hist  = None
        frame_idx  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / original_fps

            # Scene cut detection (histogram difference)
            is_scene_cut = self._detect_scene_cut(frame, prev_hist)
            if is_scene_cut:
                scene_cuts.append(frame_idx)

            # Sample at target FPS or always include scene boundaries
            if frame_idx % frame_interval == 0 or is_scene_cut:
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                frames.append(frame)
                timestamps.append(timestamp)

            prev_hist = self._compute_hist(frame)
            frame_idx += 1

            if len(frames) >= self.max_frames:
                logger.info("Max frames cap (%d) reached.", self.max_frames)
                break

        cap.release()

        logger.info(
            "Extracted %d frames | %d scene cuts detected",
            len(frames), len(scene_cuts),
        )
        return frames, timestamps

    def extract_with_metadata(self, video_path: str) -> dict:
        """
        Extended extraction returning full metadata dict.
        """
        cap = cv2.VideoCapture(str(video_path))
        metadata = {
            "fps":          cap.get(cv2.CAP_PROP_FPS),
            "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_s":   cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            "codec":        int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()

        frames, timestamps = self.extract(video_path)
        return {
            "frames":     frames,
            "timestamps": timestamps,
            "metadata":   metadata,
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _compute_hist(self, frame: np.ndarray) -> np.ndarray:
        """Compute 3-channel colour histogram for scene cut detection."""
        hist = []
        for ch in range(3):
            h = cv2.calcHist([frame], [ch], None, [64], [0, 256])
            hist.append(h)
        return np.concatenate(hist).flatten()

    def _detect_scene_cut(
        self,
        frame:     np.ndarray,
        prev_hist: Optional[np.ndarray],
    ) -> bool:
        """Return True if this frame is a scene boundary."""
        if prev_hist is None:
            return False
        curr_hist = self._compute_hist(frame)
        diff = cv2.compareHist(
            prev_hist.astype(np.float32),
            curr_hist.astype(np.float32),
            cv2.HISTCMP_CHISQR,
        )
        return diff > self.scene_threshold


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path>")
        sys.exit(0)

    extractor = FrameExtractor(target_fps=10, max_frames=500)
    frames, ts = extractor.extract(sys.argv[1])
    print(f"✅ Extracted {len(frames)} frames")
    print(f"   Duration covered: {ts[-1]:.2f}s")
    print(f"   Frame shape: {frames[0].shape}")
