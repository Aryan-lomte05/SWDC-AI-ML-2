"""
Vision Transformer Deepfake Detector (ViT-L/16)
=================================================
ViT's global self-attention captures inconsistencies that CNNs miss:
  - Left/right face lighting mismatch
  - One ear different from the other
  - Forehead incompatible with chin lighting

Self-attention lets any patch directly attend to any other patch,
catching long-range global inconsistencies in a single forward pass.
Attention maps also serve as built-in XAI (no Grad-CAM needed).

Architecture:
  ViT-L/16 pretrained (timm)  →  [B, N+1, 1024]  (N=196 patches + CLS)
  Cross-attention pool (CLS ← patches)  →  [B, 1024]
  LayerNorm → FC(1024→256) → GELU → FC(256→1) → Sigmoid
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class ViTDeepfakeDetector(nn.Module):
    """
    ViT-L/16 based global-inconsistency detector.

    Args:
        model_name (str): timm model identifier.
        pretrained (bool): Load ImageNet-21k weights.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()

        logger.info("Initialising ViTDeepfakeDetector (%s, pretrained=%s)", model_name, pretrained)

        # ── ViT Backbone ──────────────────────────────────────────────────────
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # Remove head; we add our own
        )
        embed_dim = self.vit.embed_dim   # 1024 for ViT-L

        # ── Cross-Attention Pool (CLS queries patches) ────────────────────────
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=False,
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] — face crops (224×224 for ViT-L/16)

        Returns:
            logit       : [B, 1]         — fake probability
            attn_weights: [B, 1, N]      — attention map over patches (XAI)
        """
        # Extract all token features
        tokens = self.vit.forward_features(x)    # [B, N+1, 1024]

        cls_token    = tokens[:, 0:1, :]          # [B, 1, 1024]
        patch_tokens = tokens[:, 1:, :]           # [B, N, 1024]

        # Transpose for MultiheadAttention API: (seq, batch, dim)
        q = cls_token.transpose(0, 1)             # [1, B, 1024]
        k = patch_tokens.transpose(0, 1)          # [N, B, 1024]
        v = patch_tokens.transpose(0, 1)          # [N, B, 1024]

        attn_out, attn_weights = self.attention_pool(q, k, v)
        # attn_out: [1, B, 1024]  |  attn_weights: [B, 1, N]

        pooled = attn_out.squeeze(0)              # [B, 1024]
        logit  = self.head(pooled)                # [B, 1]

        return logit, attn_weights

    def get_attention_map(self, x: torch.Tensor, layer_idx: int = -1):
        """
        Extract raw multi-head attention from a specific transformer block.
        Used by vit_attention.py explainability module.

        Returns:
            attn: [B, heads, N+1, N+1]
        """
        attention_maps = []

        def _hook(module, inp, out):
            # timm stores attention weights in forward output tuple
            attention_maps.append(out)

        # Hook into the target block's attention
        hook = self.vit.blocks[layer_idx].attn.register_forward_hook(_hook)
        with torch.no_grad():
            _ = self.vit.forward_features(x)
        hook.remove()

        return attention_maps[0] if attention_maps else None


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTDeepfakeDetector(pretrained=False).to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)

    logit, attn = model(dummy)
    print(f"✅ ViTDeepfakeDetector | logit: {logit.shape} | attn: {attn.shape}")
    # Expected: logit [2,1], attn [2,1,196]
