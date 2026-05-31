"""
Confidence Scoring Engine
===========================
Spec (bible §10):

The brain of the system. Aggregates ALL signals from neural models
and classical analyzers into a single calibrated, uncertainty-aware
deepfake probability score.

Signal weights (from bible):
  spatial_xception  0.25
  spatial_vit       0.20
  frequency_domain  0.10
  temporal_lstm     0.20
  physiological     0.15
  lip_sync          0.10
  ─────────────────────
  Total             1.00

Output schema matches bible §10.2 JSON exactly:
  - fake_probability      [0, 1]
  - realness_score        [0, 1]
  - uncertainty           float
  - risk_level            AUTHENTIC|LOW_RISK|SUSPICIOUS|HIGH_RISK|FAKE
  - confidence_interval   (lower, upper)
  - signal_breakdown      dict
  - verdict               human-readable string
  - suspicious_segments   list of (start_ts, end_ts, score, reason)
  - frame_scores          per-frame scores
  - evidence              top evidence strings
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import timedelta

logger = logging.getLogger(__name__)


# ── Constants from bible §10.1 ────────────────────────────────────────────────

SIGNAL_WEIGHTS: Dict[str, float] = {
    "spatial_xception":  0.25,
    "spatial_vit":       0.20,
    "frequency_domain":  0.10,
    "temporal_lstm":     0.20,
    "physiological":     0.15,
    "lip_sync":          0.10,
}

RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "AUTHENTIC":  (0.00, 0.20),
    "LOW_RISK":   (0.20, 0.40),
    "SUSPICIOUS": (0.40, 0.65),
    "HIGH_RISK":  (0.65, 0.85),
    "FAKE":       (0.85, 1.00),
}

VERDICT_TEMPLATES: Dict[str, str] = {
    "AUTHENTIC":  "✅ Media appears AUTHENTIC. No significant manipulation detected.",
    "LOW_RISK":   "⚠️  LOW RISK. Minor anomalies detected. Likely authentic with caveats.",
    "SUSPICIOUS": "🔶 SUSPICIOUS. Multiple inconsistencies found. Manual review strongly recommended.",
    "HIGH_RISK":  "🚨 HIGH RISK. Strong multi-signal evidence of manipulation detected.",
    "FAKE":       "❌ FAKE DETECTED. Media is very likely AI-generated or manipulated.",
}

# Evidence region descriptions (for human-readable report)
REGION_DESCRIPTIONS: Dict[str, str] = {
    "left_eye":   "Left eye region showed unnatural blending or generation artefacts",
    "right_eye":  "Right eye region exhibited texture inconsistency",
    "mouth":      "Mouth/lip area showed synthesis boundary artefacts",
    "left_cheek": "Left cheek skin texture inconsistent with surrounding face",
    "right_cheek":"Right cheek exhibited colour/texture discontinuity",
    "forehead":   "Forehead region showed blending seam or GAN fingerprint",
    "jaw_left":   "Left jaw boundary exhibited face-mask blending artefact",
    "jaw_right":  "Right jaw boundary showed compositing inconsistency",
    "nose":       "Nose region exhibited lighting or texture anomaly",
}


@dataclass
class SuspiciousSegment:
    start_time: str    # "HH:MM:SS"
    end_time:   str
    score:      float
    reason:     str


class ConfidenceScoringEngine:
    """
    Aggregates all detection signals into a final calibrated verdict.

    Usage:
        engine = ConfidenceScoringEngine()
        result = engine.compute_final_score(
            signal_scores    = {...},
            frame_scores     = [...],
            timestamps       = [...],
            evidence_strings = [...],
            uncertainty      = 0.04,
        )
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or SIGNAL_WEIGHTS
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning("Signal weights sum to %.3f (expected 1.0) — normalising", total)
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ── Core Scoring ──────────────────────────────────────────────────────────

    def compute_final_score(
        self,
        signal_scores:    Dict[str, float],
        frame_scores:     Optional[List[float]]   = None,
        timestamps:       Optional[List[float]]   = None,
        evidence_strings: Optional[List[str]]     = None,
        uncertainty:      float = 0.0,
        video_id:         str   = "unknown",
        duration_seconds: float = 0.0,
    ) -> Dict:
        """
        Compute the final score report matching bible §10.2 JSON schema.

        Args:
            signal_scores    : Dict with signal name → score [0,1]
            frame_scores     : Per-frame fake probability list
            timestamps       : Timestamps for each frame (seconds)
            evidence_strings : Human-readable evidence from all modules
            uncertainty      : MC Dropout std (from EnsembleDetector)
            video_id         : Video identifier string
            duration_seconds : Video duration in seconds

        Returns:
            Full result dict matching the bible's output schema
        """
        # ── Weighted aggregation ──────────────────────────────────────────
        weighted_score = 0.0
        used_weights   = 0.0

        for signal, weight in self.weights.items():
            if signal in signal_scores:
                val = float(np.clip(signal_scores[signal], 0.0, 1.0))
                weighted_score += val * weight
                used_weights   += weight

        if used_weights > 0:
            weighted_score /= used_weights
        weighted_score = float(np.clip(weighted_score, 0.0, 1.0))

        # ── Risk classification ───────────────────────────────────────────
        risk_level = self._classify_risk(weighted_score)

        # ── Confidence interval ───────────────────────────────────────────
        ci_lower = max(0.0, weighted_score - 2 * uncertainty)
        ci_upper = min(1.0, weighted_score + 2 * uncertainty)

        # ── Suspicious segments ───────────────────────────────────────────
        suspicious_segments = []
        if frame_scores and timestamps:
            suspicious_segments = self._detect_suspicious_segments(
                frame_scores, timestamps, signal_scores
            )

        # ── Evidence generation ───────────────────────────────────────────
        auto_evidence = self._generate_evidence(signal_scores, weighted_score)
        all_evidence  = (evidence_strings or []) + auto_evidence
        # Deduplicate while preserving order
        seen_ev = set()
        unique_evidence = []
        for e in all_evidence:
            if e not in seen_ev:
                seen_ev.add(e)
                unique_evidence.append(e)

        # ── Build output dict (bible §10.2 schema) ────────────────────────
        result = {
            "video_id":          video_id,
            "duration_seconds":  round(duration_seconds, 2),
            "overall": {
                "fake_probability": round(weighted_score, 4),
                "realness_score":   round(1.0 - weighted_score, 4),
                "uncertainty":      round(uncertainty, 4),
                "risk_level":       risk_level,
                "verdict":          VERDICT_TEMPLATES[risk_level],
                "confidence_interval": {
                    "lower": round(ci_lower, 4),
                    "upper": round(ci_upper, 4),
                },
            },
            "signal_breakdown":  {k: round(float(v), 4) for k, v in signal_scores.items()},
            "frame_scores":      [round(float(s), 4) for s in (frame_scores or [])],
            "suspicious_segments": [asdict(seg) for seg in suspicious_segments],
            "explainability": {
                "top_evidence":    unique_evidence[:8],
                "risk_factors":    self._risk_factors(signal_scores),
            },
            "metadata": {
                "signals_used":  list(signal_scores.keys()),
                "weights_used":  self.weights,
                "model_version": "1.0.0",
            },
        }

        logger.info(
            "Scored video %s | fake_prob=%.4f | risk=%s | uncertainty=%.4f",
            video_id, weighted_score, risk_level, uncertainty,
        )

        return result

    # ── Suspicious Segment Detection ──────────────────────────────────────────

    def _detect_suspicious_segments(
        self,
        frame_scores: List[float],
        timestamps:   List[float],
        signal_scores: Dict[str, float],
        threshold:    float = 0.55,
        min_length_s: float = 0.5,
    ) -> List[SuspiciousSegment]:
        """
        Identify contiguous time segments where frame scores exceed threshold.
        Groups adjacent suspicious frames into segments.
        """
        segments     = []
        in_segment   = False
        seg_start    = 0.0
        seg_frames   = []

        for i, (score, ts) in enumerate(zip(frame_scores, timestamps)):
            if score >= threshold:
                if not in_segment:
                    in_segment = True
                    seg_start  = ts
                    seg_frames = []
                seg_frames.append(score)
            else:
                if in_segment:
                    seg_end = timestamps[i - 1] if i > 0 else ts
                    if (seg_end - seg_start) >= min_length_s:
                        reason = self._segment_reason(signal_scores, np.mean(seg_frames))
                        segments.append(SuspiciousSegment(
                            start_time = self._format_time(seg_start),
                            end_time   = self._format_time(seg_end),
                            score      = round(float(np.mean(seg_frames)), 4),
                            reason     = reason,
                        ))
                    in_segment = False

        # Close open segment
        if in_segment and seg_frames:
            reason = self._segment_reason(signal_scores, np.mean(seg_frames))
            segments.append(SuspiciousSegment(
                start_time = self._format_time(seg_start),
                end_time   = self._format_time(timestamps[-1]),
                score      = round(float(np.mean(seg_frames)), 4),
                reason     = reason,
            ))

        return segments[:10]  # Cap at 10 segments

    def _segment_reason(self, signal_scores: Dict[str, float], seg_score: float) -> str:
        """Generate a human-readable reason for the suspicious segment."""
        # Identify dominant signal
        dominant = max(signal_scores.items(), key=lambda x: x[1]) if signal_scores else ("unknown", 0)

        reason_map = {
            "spatial_xception":  "Face blending artefact (pixel-level GAN fingerprint)",
            "spatial_vit":       "Global facial inconsistency (lighting/geometry mismatch)",
            "frequency_domain":  "GAN frequency artefact (checkerboard pattern in FFT)",
            "temporal_lstm":     "Temporal coherence anomaly (motion discontinuity)",
            "physiological":     "Biological signal absence (rPPG/blink anomaly)",
            "lip_sync":          "Lip-sync inconsistency (Wav2Lip-style forgery)",
        }
        return reason_map.get(dominant[0], "Multi-signal manipulation evidence")

    # ── Evidence Generation ───────────────────────────────────────────────────

    def _generate_evidence(
        self,
        signal_scores: Dict[str, float],
        final_score:   float,
    ) -> List[str]:
        """Generate automatic evidence strings from signal scores."""
        evidence = []

        thresholds = {
            "spatial_xception":  0.60,
            "spatial_vit":       0.60,
            "frequency_domain":  0.55,
            "temporal_lstm":     0.60,
            "physiological":     0.50,
            "lip_sync":          0.55,
        }

        descriptions = {
            "spatial_xception":  "XceptionNet detected pixel-level manipulation artefacts (blending seam / GAN fingerprint)",
            "spatial_vit":       "ViT-L attention reveals global facial inconsistency across patch regions",
            "frequency_domain":  "Frequency domain analysis detected GAN upsampling artefacts in FFT spectrum",
            "temporal_lstm":     "Bidirectional LSTM flagged temporal motion discontinuity across frame sequence",
            "physiological":     "Physiological analysis: absent/abnormal rPPG signal or anomalous blink pattern",
            "lip_sync":          "Lip-sync analysis: sharpness discontinuity at mouth boundary (Wav2Lip signature)",
        }

        for signal, score in sorted(signal_scores.items(), key=lambda x: -x[1]):
            if score >= thresholds.get(signal, 0.6):
                evidence.append(f"{descriptions.get(signal, signal)} — score: {score:.2%}")

        if final_score > 0.85:
            evidence.insert(0, "⚡ STRONG MULTI-SIGNAL CONSENSUS: All major detection pathways agree on manipulation")

        return evidence

    def _risk_factors(self, signal_scores: Dict[str, float]) -> List[Dict]:
        """Return sorted risk factors for display in the frontend."""
        factors = []
        for signal, score in sorted(signal_scores.items(), key=lambda x: -x[1]):
            weight = self.weights.get(signal, 0)
            factors.append({
                "signal":      signal,
                "score":       round(float(score), 4),
                "weight":      weight,
                "contribution": round(float(score * weight), 4),
                "level":       "HIGH" if score > 0.65 else "MEDIUM" if score > 0.40 else "LOW",
            })
        return factors

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify_risk(self, score: float) -> str:
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "FAKE"

    @staticmethod
    def _format_time(seconds: float) -> str:
        return str(timedelta(seconds=int(seconds))).zfill(8)   # HH:MM:SS

    # ── Mock Score (MOCK_MODE) ────────────────────────────────────────────────

    @staticmethod
    def generate_mock_scores(
        seed: Optional[int] = None,
        bias: str = "fake",     # 'fake' | 'real' | 'random'
    ) -> Dict[str, float]:
        """
        Generate realistic simulated signal scores for demo/testing.
        Used when MOCK_MODE=True in .env.

        Args:
            seed : Random seed for reproducibility
            bias : 'fake' → high scores, 'real' → low scores

        Returns:
            Dict of signal_name → score [0, 1]
        """
        rng = np.random.default_rng(seed)

        if bias == "fake":
            base   = rng.uniform(0.70, 0.95)
            noise  = 0.08
        elif bias == "real":
            base   = rng.uniform(0.05, 0.25)
            noise  = 0.06
        else:
            base   = rng.uniform(0.20, 0.80)
            noise  = 0.15

        return {
            signal: float(np.clip(base + rng.normal(0, noise), 0.0, 1.0))
            for signal in SIGNAL_WEIGHTS
        }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    engine = ConfidenceScoringEngine()

    # Test with mock FAKE scores
    fake_signals = ConfidenceScoringEngine.generate_mock_scores(seed=42, bias="fake")
    frame_scores = [0.8 + np.random.normal(0, 0.05) for _ in range(45)]
    timestamps   = [i * 0.1 for i in range(45)]

    result = engine.compute_final_score(
        signal_scores    = fake_signals,
        frame_scores     = frame_scores,
        timestamps       = timestamps,
        video_id         = "test_vid_001",
        duration_seconds = 4.5,
    )

    print(f"✅ ConfidenceScoringEngine")
    print(f"   fake_probability : {result['overall']['fake_probability']:.4f}")
    print(f"   risk_level       : {result['overall']['risk_level']}")
    print(f"   verdict          : {result['overall']['verdict']}")
    print(f"   suspicious segs  : {len(result['suspicious_segments'])}")
    print(f"   evidence[0]      : {result['explainability']['top_evidence'][0] if result['explainability']['top_evidence'] else 'none'}")

    # Test with REAL scores
    real_signals = ConfidenceScoringEngine.generate_mock_scores(seed=7, bias="real")
    result2 = engine.compute_final_score(real_signals, video_id="real_vid_001")
    print(f"\n   [REAL] fake_prob={result2['overall']['fake_probability']:.4f} | risk={result2['overall']['risk_level']}")
