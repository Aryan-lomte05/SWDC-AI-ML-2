"""
Ensemble Detector — Multi-Model Fusion
=========================================
Combines outputs from all 4 backbone detectors into a single
calibrated final score.

Fusion strategy:
  1. Per-model Temperature Scaling  →  calibrated probabilities
  2. Learnable MLP fusion           →  weighted combination
  3. MC Dropout (N=20 passes)       →  uncertainty estimation
  4. Final sigmoid                  →  [0, 1] fake probability

Signal weights (from bible):
  spatial_xception  0.25
  spatial_vit       0.20
  frequency_domain  0.10
  temporal_lstm     0.20
  (physiological + lip_sync added by ConfidenceScoringEngine externally)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class EnsembleDetector(nn.Module):
    """
    Learnable ensemble fusion over all backbone model scores.

    Args:
        xception    : XceptionDetector instance
        vit         : ViTDeepfakeDetector instance
        freq_net    : FrequencyAnalyzer instance
        temporal    : TemporalDeepfakeDetector instance
        mc_passes   : Number of MC-Dropout inference passes for uncertainty
    """

    def __init__(
        self,
        xception,
        vit,
        freq_net,
        temporal,
        mc_passes: int = 20,
    ):
        super().__init__()

        self.xception  = xception
        self.vit       = vit
        self.freq_net  = freq_net
        self.temporal  = temporal
        self.mc_passes = mc_passes

        # ── Temperature Scaling (one per model, learnable) ────────────────────
        # Start at 1.0 (no scaling), fine-tuned during calibration
        self.temperatures = nn.Parameter(torch.ones(4))

        # ── Learnable Fusion MLP ──────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _calibrate(self, score: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling: logit / T → sigmoid."""
        logit = torch.logit(score.clamp(1e-6, 1 - 1e-6))
        return torch.sigmoid(logit / temp.abs().clamp(min=0.1))

    def forward(
        self,
        frames: torch.Tensor,
        frame_features: torch.Tensor,
    ) -> Dict:
        """
        Args:
            frames        : [B, 3, H, W]      — face crops for spatial models
            frame_features: [1, T, 2048]       — XceptionNet embeddings for temporal

        Returns dict:
            final_score   : [1]     — calibrated ensemble fake probability
            model_scores  : [4]     — individual model scores
            attention_map : [B,1,N] — ViT patch attention (XAI)
            frame_weights : [1, T]  — per-frame suspicion (timeline)
        """
        # ── Individual model inference ─────────────────────────────────────
        p_xception, _ = self.xception(frames)         # [B, 1]
        p_vit,  attn  = self.vit(frames)              # [B, 1], [B,1,N]
        p_freq        = self.freq_net(frames)          # [B, 1]
        p_temporal, fw = self.temporal(frame_features) # [1, 1], [1, T]

        # Aggregate spatial models over batch (mean across frames)
        p_xception = p_xception.mean()
        p_vit_s    = p_vit.mean()
        p_freq_s   = p_freq.mean()
        p_temp_s   = p_temporal.squeeze()

        # ── Temperature calibration ────────────────────────────────────────
        scores = torch.stack([
            self._calibrate(p_xception.unsqueeze(0), self.temperatures[0]),
            self._calibrate(p_vit_s.unsqueeze(0),    self.temperatures[1]),
            self._calibrate(p_freq_s.unsqueeze(0),   self.temperatures[2]),
            self._calibrate(p_temp_s.unsqueeze(0),   self.temperatures[3]),
        ], dim=-1)  # [1, 4]

        # ── Fusion ─────────────────────────────────────────────────────────
        final_score = self.fusion(scores)  # [1, 1]

        return {
            "final_score":   final_score.squeeze().item(),
            "model_scores":  {
                "spatial_xception": p_xception.item(),
                "spatial_vit":      p_vit_s.item(),
                "frequency_domain": p_freq_s.item(),
                "temporal_lstm":    p_temp_s.item(),
            },
            "attention_map":  attn,    # [B, 1, N] — for XAI
            "frame_weights":  fw,      # [1, T]    — for timeline
        }

    @torch.no_grad()
    def mc_uncertainty(
        self,
        frames: torch.Tensor,
        frame_features: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Monte Carlo Dropout uncertainty estimation.
        Enable dropout at inference; run N passes; compute std.

        Returns:
            mean_score  : float — mean prediction across passes
            uncertainty : float — std across passes
        """
        self.train()   # Enable dropout
        scores = []
        for _ in range(self.mc_passes):
            result = self.forward(frames, frame_features)
            scores.append(result["final_score"])
        self.eval()

        scores_arr = np.array(scores)
        return float(scores_arr.mean()), float(scores_arr.std())


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

    from ml.models.xception_detector import XceptionDetector
    from ml.models.vit_detector      import ViTDeepfakeDetector
    from ml.models.freq_analyzer     import FrequencyAnalyzer
    from ml.models.temporal_lstm     import TemporalDeepfakeDetector

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xception = XceptionDetector(pretrained=False).to(device)
    vit      = ViTDeepfakeDetector(pretrained=False).to(device)
    freq     = FrequencyAnalyzer().to(device)
    temporal = TemporalDeepfakeDetector().to(device)

    ensemble = EnsembleDetector(xception, vit, freq, temporal).to(device)

    frames   = torch.randn(8, 3, 224, 224).to(device)
    feats    = torch.randn(1, 8, 2048).to(device)

    result = ensemble(frames, feats)
    print(f"✅ EnsembleDetector | final_score: {result['final_score']:.4f}")
    print(f"   model_scores: {result['model_scores']}")
    print(f"   frame_weights: {result['frame_weights'].shape}")
