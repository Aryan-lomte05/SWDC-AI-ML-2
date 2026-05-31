"""
Bidirectional LSTM Temporal Deepfake Detector
===============================================
Most amateur detectors analyse INDIVIDUAL frames.
This model operates on SEQUENCES of frame embeddings — the key differentiator.

Key insights:
  - Deepfakes score consistently high across frames (no variance)
  - Real face manipulation only in a short segment → temporal localisation
  - Biological motion (blink timing, expression dynamics) is captured
    across frames, not within any single frame

Architecture:
  Frame embeddings [B, T, 2048]  (from XceptionNet)
  → BiLSTM(2 layers, hidden=512, bidirectional)  →  [B, T, 1024]
  → Temporal Attention                            →  [B, 1024]  +  attn weights [B, T]
  → FC(1024→128) → ReLU → Dropout → FC(128→1) → Sigmoid

The temporal attention weights tell us WHICH FRAMES are most suspicious —
critical for the frame-level timeline in the UI.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TemporalDeepfakeDetector(nn.Module):
    """
    Bidirectional LSTM with temporal attention for video-level detection.

    Args:
        feature_dim (int): Dimension of per-frame embedding (2048 from Xception).
        hidden_dim  (int): LSTM hidden state size.
        num_layers  (int): Number of stacked LSTM layers.
        dropout     (float): Dropout between LSTM layers.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim:  int = 512,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()

        logger.info(
            "Initialising TemporalDeepfakeDetector | feature_dim=%d hidden_dim=%d layers=%d",
            feature_dim, hidden_dim, num_layers,
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = feature_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2   # bidirectional → 1024

        # ── Temporal Attention ────────────────────────────────────────────────
        # Learns a suspicion score per frame
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Layer Norm for stability ──────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

    def forward(self, frame_features: torch.Tensor):
        """
        Args:
            frame_features: [B, T, feature_dim] — sequence of per-frame embeddings

        Returns:
            logit         : [B, 1]   — video-level fake probability
            attn_weights  : [B, T]   — per-frame suspicion weights (for timeline UI)
        """
        # BiLSTM over sequence
        lstm_out, _ = self.lstm(frame_features)   # [B, T, 1024]
        lstm_out    = self.layer_norm(lstm_out)

        # Temporal attention
        raw_attn     = self.attention(lstm_out)              # [B, T, 1]
        attn_weights = torch.softmax(raw_attn, dim=1)        # [B, T, 1]
        context      = (attn_weights * lstm_out).sum(dim=1)  # [B, 1024]

        logit = self.classifier(context)  # [B, 1]

        return logit, attn_weights.squeeze(-1)  # [B,1], [B,T]

    def get_frame_suspicion_scores(self, frame_features: torch.Tensor):
        """
        Returns per-frame raw suspicion scores (before softmax normalisation).
        More useful than normalised weights for absolute frame flagging.
        """
        lstm_out, _ = self.lstm(frame_features)
        lstm_out    = self.layer_norm(lstm_out)
        raw_attn    = self.attention(lstm_out).squeeze(-1)    # [B, T]
        return torch.sigmoid(raw_attn)                         # [B, T] in [0,1]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalDeepfakeDetector().to(device)

    # Simulate 30 frames of XceptionNet embeddings
    dummy = torch.randn(2, 30, 2048).to(device)

    logit, attn = model(dummy)
    print(f"✅ TemporalDeepfakeDetector | logit: {logit.shape} | attn: {attn.shape}")
    print(f"   Attn sum (should ≈ 1.0): {attn.sum(dim=1)}")
    # Expected: logit [2,1], attn [2,30]
