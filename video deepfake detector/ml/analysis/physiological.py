"""
Physiological Signal Analyzer
================================
Spec (bible §8.1):

Real humans have measurable biological signals visible in video.
Deepfakes cannot replicate them:

  1. rPPG (Remote Photoplethysmography)
       Heart pumps blood → cheek skin changes green-channel intensity
       at heartbeat frequency (0.75–3.5 Hz = 45–210 BPM).
       Low SNR = no heartbeat signal = likely fake.

  2. Eye Blink Analysis
       Normal blink rate: 15–20 per minute.
       Deepfakes: missing blinks (early GANs) or wrong timing.
       Asymmetric blinking (eyes blink independently) = fake.

  3. Head Micro-movements
       Real video has natural head sway; deepfakes composited onto
       a video may have face motion inconsistent with head motion.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# rPPG constants
RPPG_LOW_HZ  = 0.75    # 45 BPM
RPPG_HIGH_HZ = 3.5     # 210 BPM
VIDEO_FPS    = 10      # Our extracted frame rate

# Blink constants
NORMAL_BLINK_MIN = 5.0    # blinks/min (below = suspicious)
NORMAL_BLINK_MAX = 40.0   # blinks/min (above = suspicious)
EAR_THRESHOLD    = 0.21   # Eye Aspect Ratio below this = closed


class PhysiologicalAnalyzer:
    """
    Analyses biological signals in a face video sequence to detect
    signs of deepfake generation that breaks physiological coherence.
    """

    def analyze(
        self,
        face_regions: List[np.ndarray],
        fps: float = VIDEO_FPS,
    ) -> Dict:
        """
        Full physiological analysis on a sequence of face crops.

        Args:
            face_regions: List of aligned BGR face crops [H, W, 3]
            fps:          Effective frames-per-second of the sequence

        Returns:
            dict with:
              - anomaly_score    : float [0, 1] — higher = more suspicious
              - rppg             : dict
              - blink            : dict
              - head_motion      : dict
              - is_suspicious    : bool
              - evidence         : List[str]
        """
        if len(face_regions) < 10:
            return self._insufficient_data()

        rppg_result    = self.analyze_rppg(face_regions, fps)
        blink_result   = self.analyze_blink_patterns(face_regions, fps)
        motion_result  = self.analyze_head_motion(face_regions)

        # Aggregate anomaly score
        scores = []
        evidence = []

        if rppg_result["is_suspicious"]:
            scores.append(0.8)
            evidence.append(
                f"Absent heartbeat signal (rPPG SNR={rppg_result['rppg_snr']:.3f}, "
                f"expected >0.1 for real faces)"
            )
        else:
            scores.append(0.1)

        if blink_result["is_suspicious"]:
            scores.append(0.7)
            evidence.append(
                f"Abnormal blink rate ({blink_result['blink_rate']:.1f}/min, "
                f"normal: {NORMAL_BLINK_MIN}–{NORMAL_BLINK_MAX}/min)"
            )
        else:
            scores.append(0.1)

        if motion_result["is_suspicious"]:
            scores.append(0.6)
            evidence.append(
                f"Unnatural head micro-motion (variance={motion_result['motion_variance']:.4f})"
            )
        else:
            scores.append(0.1)

        anomaly_score = float(np.mean(scores))

        return {
            "anomaly_score":  anomaly_score,
            "rppg":           rppg_result,
            "blink":          blink_result,
            "head_motion":    motion_result,
            "is_suspicious":  anomaly_score > 0.4,
            "evidence":       evidence,
        }

    # ── rPPG ──────────────────────────────────────────────────────────────────

    def analyze_rppg(self, face_regions: List[np.ndarray], fps: float) -> Dict:
        """
        Remote Photoplethysmography: detect heart rate signal.
        Deepfakes lack coherent rPPG signals.

        Method:
          Extract mean green channel from cheek ROI per frame.
          Apply bandpass FFT to isolate heartbeat frequency band.
          Compute SNR in HR band vs total spectrum.
        """
        # Extract green channel mean from cheeks (top-centre strip)
        signals = []
        for face in face_regions:
            h, w = face.shape[:2]
            # Cheek ROI: central horizontal strip
            roi = face[int(0.40 * h): int(0.65 * h), int(0.10 * w): int(0.90 * w)]
            if roi.size == 0:
                signals.append(0.0)
                continue
            green_mean = float(np.mean(roi[:, :, 1]))   # Green channel
            signals.append(green_mean)

        signals = np.array(signals, dtype=np.float64)

        # Detrend
        signals -= np.mean(signals)

        # FFT
        n    = len(signals)
        fft  = np.abs(np.fft.rfft(signals))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)

        # HR band
        hr_mask     = (freqs >= RPPG_LOW_HZ) & (freqs <= RPPG_HIGH_HZ)
        hr_power    = fft[hr_mask].max() if hr_mask.any() else 0.0
        total_power = fft.sum() + 1e-8

        snr = float(hr_power / total_power)

        # Estimated heart rate
        est_hr = 0.0
        if hr_mask.any() and fft[hr_mask].size > 0:
            peak_freq = freqs[hr_mask][np.argmax(fft[hr_mask])]
            est_hr    = float(peak_freq * 60.0)

        return {
            "rppg_snr":     snr,
            "estimated_hr": est_hr,
            "is_suspicious": snr < 0.1,
            "signal":       signals.tolist()[:50],   # First 50 for UI
        }

    # ── Blink Detection ───────────────────────────────────────────────────────

    def analyze_blink_patterns(self, face_regions: List[np.ndarray], fps: float) -> Dict:
        """
        Detect blink events using Eye Aspect Ratio (EAR).
        EAR < 0.21 → eye closed → blink.

        Returns:
            blink_rate       : blinks per minute
            blink_frames     : list of frame indices where blink occurred
            symmetry_score   : 0–1 (1 = perfectly symmetric blinking)
            is_suspicious    : bool
        """
        ears      = []
        for face in face_regions:
            ear = self._compute_ear_approx(face)
            ears.append(ear)

        ears = np.array(ears)

        # Detect blink events (EAR drops below threshold for 1–4 frames)
        below = ears < EAR_THRESHOLD
        blink_frames = []
        in_blink = False
        for i, b in enumerate(below):
            if b and not in_blink:
                blink_frames.append(i)
                in_blink = True
            elif not b:
                in_blink = False

        # Compute rate
        duration_min = len(face_regions) / (fps * 60.0)
        blink_rate   = len(blink_frames) / max(duration_min, 1e-6)

        # Symmetry: check if blinks are regularly spaced (real) vs random
        if len(blink_frames) > 2:
            intervals       = np.diff(blink_frames)
            symmetry_score  = float(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-6))
            symmetry_score  = float(np.clip(symmetry_score, 0.0, 1.0))
        else:
            symmetry_score  = 0.5

        is_suspicious = (
            blink_rate < NORMAL_BLINK_MIN
            or blink_rate > NORMAL_BLINK_MAX
            or symmetry_score < 0.2
        )

        return {
            "blink_rate":      blink_rate,
            "blink_frames":    blink_frames,
            "symmetry_score":  symmetry_score,
            "ear_series":      ears.tolist()[:50],
            "is_suspicious":   is_suspicious,
        }

    # ── Head Motion ───────────────────────────────────────────────────────────

    def analyze_head_motion(self, face_regions: List[np.ndarray]) -> Dict:
        """
        Track head micro-motion across frames via feature point matching.
        Deepfakes composited on video may have face motion inconsistent
        with background head motion.
        """
        if len(face_regions) < 3:
            return {"motion_variance": 0.0, "is_suspicious": False}

        motions = []
        prev_gray = cv2.cvtColor(face_regions[0], cv2.COLOR_BGR2GRAY)

        for face in face_regions[1:]:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Lucas-Kanade optical flow on good features
            pts = cv2.goodFeaturesToTrack(prev_gray, 50, 0.01, 10)
            if pts is None:
                motions.append(0.0)
                prev_gray = gray
                continue
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
            good_new = next_pts[status == 1]
            good_old = pts[status == 1]
            if len(good_new) == 0:
                motions.append(0.0)
                prev_gray = gray
                continue
            motion = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            motions.append(float(motion))
            prev_gray = gray

        motions_arr    = np.array(motions)
        motion_variance = float(np.var(motions_arr))

        # Extremely low variance = totally static (deepfake with no natural sway)
        # Extremely high variance = jittery (deepfake warping artefact)
        is_suspicious = motion_variance < 0.01 or motion_variance > 50.0

        return {
            "motion_variance": motion_variance,
            "motion_profile":  motions[:50],
            "is_suspicious":   is_suspicious,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_ear_approx(self, face: np.ndarray) -> float:
        """
        Approximate Eye Aspect Ratio without 68-point landmarks.
        Uses eye region brightness contrast as proxy for openness.
        """
        h, w = face.shape[:2]
        # Eye strip
        eye_strip = face[int(0.35 * h): int(0.55 * h), :]
        gray      = cv2.cvtColor(eye_strip, cv2.COLOR_BGR2GRAY)
        # Normalised variance: low variance = eye closed (uniform dark)
        variance  = float(np.var(gray)) / (255.0 ** 2)
        ear       = float(np.clip(variance * 10.0, 0.0, 1.0))
        return ear

    def _insufficient_data(self) -> Dict:
        return {
            "anomaly_score":  0.0,
            "rppg":           {"rppg_snr": 0.0, "estimated_hr": 0.0, "is_suspicious": False},
            "blink":          {"blink_rate": 0.0, "is_suspicious": False},
            "head_motion":    {"motion_variance": 0.0, "is_suspicious": False},
            "is_suspicious":  False,
            "evidence":       ["Insufficient frames for physiological analysis (<10)"],
        }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Synthetic test: 60 random face-sized frames
    fake_faces = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(60)]
    analyzer   = PhysiologicalAnalyzer()
    result     = analyzer.analyze(fake_faces, fps=10)
    print(f"✅ PhysiologicalAnalyzer")
    print(f"   anomaly_score: {result['anomaly_score']:.4f}")
    print(f"   rPPG SNR:      {result['rppg']['rppg_snr']:.4f}")
    print(f"   blink_rate:    {result['blink']['blink_rate']:.2f}/min")
    print(f"   evidence:      {result['evidence']}")
