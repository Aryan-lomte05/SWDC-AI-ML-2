"""
XceptionNet Deepfake Detector
=================================
Gold standard for spatial deepfake detection (FaceForensics++ baseline).

Depthwise separable convolutions are uniquely suited for capturing
per-channel manipulation artifacts — blending seams, GAN fingerprints,
lighting inconsistencies — that standard CNNs miss.

Architecture:
  ImageNet-pretrained XceptionNet backbone (timm)
  → Global Average Pooling  →  [B, 2048]
  → Dropout(0.5)
  → FC(2048→512) + ReLU + Dropout(0.3)
  → FC(512→1) + Sigmoid
  Returns: (logit, embedding) — embedding fed to TemporalLSTM
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class XceptionDetector(nn.Module):
    """
    XceptionNet-based spatial deepfake detector.

    Args:
        num_classes (int): Output classes. 1 for binary fake/real.
        pretrained   (bool): Load ImageNet weights.
        dropout      (float): Dropout rate before classifier head.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        logger.info("Initialising XceptionDetector (pretrained=%s)", pretrained)

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,          # Remove original classifier
            global_pool="avg",      # [B, 2048]
        )

        self.embedding_dim = self.backbone.num_features  # 2048

        # ── Classification Head ───────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] — normalised face crops (224×224 or 299×299)

        Returns:
            logit     : [B, 1]     — fake probability
            embedding : [B, 2048]  — feature vector for temporal model
        """
        embedding = self.backbone(x)          # [B, 2048]
        embedding = self.dropout(embedding)
        logit     = self.classifier(embedding) # [B, 1]
        return logit, embedding

    def get_target_layer(self):
        """Return last conv layer for Grad-CAM visualisation."""
        return self.backbone.act4   # Last activation block in Xception


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = XceptionDetector(pretrained=False).to(device)
    dummy = torch.randn(4, 3, 299, 299).to(device)

    logit, emb = model(dummy)
    print(f"✅ XceptionDetector | logit: {logit.shape} | embedding: {emb.shape}")
    # Expected: logit [4,1], embedding [4,2048]
