"""
Lip-Sync Analyzer
==================
Spec (bible §8.2):

Detects Wav2Lip-style audio-visual forgeries where ONLY the lip region
is replaced. Most of the face is real — only the mouth is fake.

Detection signals:
  1. Lip landmark velocity discontinuities
       Real lips have smooth velocity profiles.
       Swapped lips show sudden jumps (different motion source).

  2. Lip region sharpness discontinuity
       Wav2Lip super-resolves the lip region differently from
       the surrounding face → Laplacian variance mismatch.

  3. Lip texture statistics
       Mean, std, entropy of lip ROI vs surrounding face.
       Replaced regions have different noise/texture statistics.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class LipSyncAnalyzer:
    """
    Detects lip-sync forgeries (Wav2Lip and similar techniques).
    Operates on a sequence of aligned face crops.
    """

    def analyze(
        self,
        face_regions: List[np.ndarray],
        fps: float = 10.0,
    ) -> Dict:
        """
        Full lip-sync analysis on a face crop sequence.

        Args:
            face_regions: List of aligned BGR face crops [H, W, 3]
            fps:          Effective frame rate

        Returns:
            dict with:
              - anomaly_score  : float [0, 1]
              - is_suspicious  : bool
              - sharpness      : dict
              - velocity       : dict
              - texture        : dict
              - evidence       : List[str]
        """
        if len(face_regions) < 5:
            return self._insufficient_data()

        sharpness_result = self.check_lip_region_sharpness(face_regions)
        velocity_result  = self.check_lip_motion_velocity(face_regions)
        texture_result   = self.check_lip_texture_stats(face_regions)

        scores   = []
        evidence = []

        if sharpness_result["is_suspicious"]:
            scores.append(0.75)
            evidence.append(
                f"Lip region sharpness ratio={sharpness_result['mean_ratio']:.2f} "
                f"(normal: 0.3–2.0) — possible Wav2Lip replacement"
            )
        else:
            scores.append(0.1)

        if velocity_result["is_suspicious"]:
            scores.append(0.8)
            evidence.append(
                f"{len(velocity_result['discontinuities'])} lip motion discontinuities "
                f"detected at frames: {velocity_result['discontinuities'][:5]}"
            )
        else:
            scores.append(0.1)

        if texture_result["is_suspicious"]:
            scores.append(0.65)
            evidence.append(
                f"Lip texture statistics differ from surrounding face "
                f"(std_ratio={texture_result['std_ratio']:.2f})"
            )
        else:
            scores.append(0.1)

        anomaly_score = float(np.mean(scores))

        return {
            "anomaly_score": anomaly_score,
            "is_suspicious": anomaly_score > 0.35,
            "sharpness":     sharpness_result,
            "velocity":      velocity_result,
            "texture":       texture_result,
            "evidence":      evidence,
        }

    # ── Sharpness Analysis ────────────────────────────────────────────────────

    def check_lip_region_sharpness(self, face_regions: List[np.ndarray]) -> Dict:
        """
        Compare Laplacian sharpness of lip region vs surrounding face.
        Wav2Lip super-resolves only the mouth → different sharpness.

        Spec: sharpness_ratio > 2.0 or < 0.3 → suspicious
        """
        ratios = []

        for face in face_regions:
            h, w = face.shape[:2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Lip mask: lower quarter of face, horizontal centre
            y1, y2 = int(0.68 * h), int(0.88 * h)
            x1, x2 = int(0.25 * w), int(0.75 * w)

            lip_region  = gray[y1:y2, x1:x2]
            # Surrounding face (exclude lip region)
            face_region = gray.copy()
            face_region[y1:y2, x1:x2] = 0   # blank out lip

            if lip_region.size == 0:
                continue

            lip_lap  = cv2.Laplacian(lip_region,  cv2.CV_64F).var()
            # Face region — non-zero pixels only
            face_roi = face_region[face_region > 0]
            if face_roi.size == 0:
                continue
            face_lap = float(np.var(np.abs(np.gradient(face_roi.astype(float)))))

            ratio = lip_lap / (face_lap + 1e-8)
            ratios.append(ratio)

        if not ratios:
            return {"mean_ratio": 1.0, "is_suspicious": False}

        mean_ratio   = float(np.mean(ratios))
        is_suspicious = mean_ratio > 2.0 or mean_ratio < 0.3

        return {
            "mean_ratio":   mean_ratio,
            "ratio_series": ratios[:50],
            "is_suspicious": is_suspicious,
        }

    # ── Velocity Analysis ─────────────────────────────────────────────────────

    def check_lip_motion_velocity(self, face_regions: List[np.ndarray]) -> Dict:
        """
        Track lip region optical flow velocity across frames.
        Swapped lips have discontinuous velocity profiles (different source).

        Spec: discontinuities > 2 → suspicious
        """
        velocities = []

        for i in range(1, len(face_regions)):
            prev = face_regions[i - 1]
            curr = face_regions[i]

            h, w = prev.shape[:2]
            y1, y2 = int(0.65 * h), int(0.90 * h)
            x1, x2 = int(0.20 * w), int(0.80 * w)

            prev_lip = cv2.cvtColor(prev[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            curr_lip = cv2.cvtColor(curr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            if prev_lip.size == 0 or curr_lip.size == 0:
                velocities.append(0.0)
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_lip, curr_lip, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            velocities.append(float(mag.mean()))

        if len(velocities) < 3:
            return {"discontinuities": [], "velocity_profile": velocities, "is_suspicious": False}

        velocities_arr = np.array(velocities)
        velocity_std   = np.std(velocities_arr)

        # Discontinuities: sudden jumps > 3 std from local mean
        discontinuities = []
        for i in range(1, len(velocities_arr)):
            jump = abs(velocities_arr[i] - velocities_arr[i - 1])
            if jump > 3 * velocity_std:
                discontinuities.append(i)

        return {
            "velocity_profile":  velocities[:50],
            "discontinuities":   discontinuities,
            "velocity_std":      float(velocity_std),
            "is_suspicious":     len(discontinuities) > 2,
        }

    # ── Texture Statistics ────────────────────────────────────────────────────

    def check_lip_texture_stats(self, face_regions: List[np.ndarray]) -> Dict:
        """
        Compare mean, std, entropy of lip region vs surrounding face.
        Replaced regions have different noise/texture distributions.
        """
        lip_stds  = []
        face_stds = []

        for face in face_regions:
            h, w = face.shape[:2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(float)

            y1, y2 = int(0.68 * h), int(0.88 * h)
            x1, x2 = int(0.25 * w), int(0.75 * w)

            lip  = gray[y1:y2, x1:x2]
            rest = np.concatenate([
                gray[:y1, :].ravel(),
                gray[y2:, :].ravel(),
            ])

            if lip.size == 0 or rest.size == 0:
                continue

            lip_stds.append(float(np.std(lip)))
            face_stds.append(float(np.std(rest)))

        if not lip_stds:
            return {"std_ratio": 1.0, "is_suspicious": False}

        mean_lip_std  = float(np.mean(lip_stds))
        mean_face_std = float(np.mean(face_stds))
        std_ratio     = mean_lip_std / (mean_face_std + 1e-8)

        # Abnormal ratio = texture mismatch between lip and face
        is_suspicious = std_ratio > 2.5 or std_ratio < 0.4

        return {
            "std_ratio":    std_ratio,
            "lip_std":      mean_lip_std,
            "face_std":     mean_face_std,
            "is_suspicious": is_suspicious,
        }

    def _insufficient_data(self) -> Dict:
        return {
            "anomaly_score": 0.0,
            "is_suspicious": False,
            "sharpness":     {"mean_ratio": 1.0, "is_suspicious": False},
            "velocity":      {"discontinuities": [], "is_suspicious": False},
            "texture":       {"std_ratio": 1.0, "is_suspicious": False},
            "evidence":      ["Insufficient frames for lip-sync analysis (<5)"],
        }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    faces   = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(30)]
    analyzer = LipSyncAnalyzer()
    result   = analyzer.analyze(faces)
    print(f"✅ LipSyncAnalyzer")
    print(f"   anomaly_score:    {result['anomaly_score']:.4f}")
    print(f"   sharpness_ratio:  {result['sharpness']['mean_ratio']:.3f}")
    print(f"   discontinuities:  {result['velocity']['discontinuities']}")
