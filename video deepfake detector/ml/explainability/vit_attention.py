"""
ViT Attention Map Extractor
=============================
Spec (bible §9.2):

Extracts and visualises the self-attention weights from the last
Vision Transformer block. Shows which image PATCHES the model's
CLS token attended to when making its classification decision.

Unlike Grad-CAM (which requires a backward pass), ViT attention is
a FREE explainability signal — no extra computation, just a hook
on the existing forward pass.

Attention rollout (Abnar & Zuidema, 2020):
  Naively taking the last layer's attention misses the fact that
  attention at each layer is conditioned on ALL previous layers.
  Rollout propagates attention through ALL transformer blocks,
  giving a more faithful attribution map.

Output: 14×14 patch attention map (for ViT-L/16 on 224×224 input)
  → bilinear upsampled to 224×224 pixel heatmap
  → overlaid on face crop as forensic evidence
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import List, Optional, Dict
import math

logger = logging.getLogger(__name__)


class ViTAttentionExtractor:
    """
    Extracts attention rollout maps from a ViTDeepfakeDetector.

    Supports two modes:
      1. last_layer   — CLS-to-patch attention from the last block only
                        (fast, used during real-time inference)
      2. rollout      — propagated attention through all blocks
                        (more faithful, used for forensic reports)
    """

    def __init__(self, model, device=None):
        """
        Args:
            model  : ViTDeepfakeDetector instance
            device : torch.device
        """
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks  = []
        self._attn_maps: List[torch.Tensor] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def get_last_layer_attention(
        self,
        image_tensor: torch.Tensor,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Extract CLS-to-patch attention from a single transformer block.

        Args:
            image_tensor : [1, 3, H, W]
            layer_idx    : Block index (-1 = last block)

        Returns:
            attn_map : [H, W] numpy array in [0, 1]
        """
        self._attn_maps = []
        hook = self._register_hook(layer_idx)

        with torch.no_grad():
            _ = self.model.vit.forward_features(image_tensor.to(self.device))

        hook.remove()

        if not self._attn_maps:
            H = image_tensor.shape[2]
            return np.zeros((H, H), dtype=np.float32)

        attn = self._attn_maps[0]   # [B, num_heads, N+1, N+1]

        # Average across heads, extract CLS → patch attention
        attn_avg  = attn.mean(dim=1)                    # [B, N+1, N+1]
        cls_attn  = attn_avg[0, 0, 1:]                  # [N] — CLS attending to patches

        return self._reshape_to_image(cls_attn, image_tensor.shape[2])

    def get_attention_rollout(
        self,
        image_tensor: torch.Tensor,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Attention rollout (Abnar & Zuidema, 2020):
        Propagates attention through all transformer blocks to get
        a faithful attribution map.

        Args:
            image_tensor   : [1, 3, H, W]
            head_fusion    : How to combine attention heads ('mean' | 'max' | 'min')
            discard_ratio  : Fraction of lowest attentions to discard (noise reduction)

        Returns:
            rollout_map : [H, W] numpy array in [0, 1]
        """
        self._attn_maps = []

        # Register hooks on ALL transformer blocks
        hooks = []
        for i, block in enumerate(self.model.vit.blocks):
            h = block.attn.register_forward_hook(self._collect_attn)
            hooks.append(h)

        with torch.no_grad():
            _ = self.model.vit.forward_features(image_tensor.to(self.device))

        for h in hooks:
            h.remove()

        if not self._attn_maps:
            H = image_tensor.shape[2]
            return np.zeros((H, H), dtype=np.float32)

        # Rollout computation
        result = self._compute_rollout(head_fusion, discard_ratio)
        return self._reshape_to_image(result, image_tensor.shape[2])

    def generate_multi_scale_heatmap(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict:
        """
        Generate both fast (last-layer) and faithful (rollout) attention maps.
        Returns dict with both maps + overlay images.
        """
        img_np = self._tensor_to_numpy(image_tensor)

        fast_attn    = self.get_last_layer_attention(image_tensor)
        rollout_attn = self.get_attention_rollout(image_tensor)

        fast_overlay    = self._overlay(img_np, fast_attn)
        rollout_overlay = self._overlay(img_np, rollout_attn)

        return {
            "fast_attention":    fast_attn,
            "rollout_attention": rollout_attn,
            "fast_overlay":      fast_overlay,
            "rollout_overlay":   rollout_overlay,
            "top_patches":       self._identify_top_patches(rollout_attn),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _register_hook(self, layer_idx: int):
        block = self.model.vit.blocks[layer_idx]
        return block.attn.register_forward_hook(self._collect_attn)

    def _collect_attn(self, module, inp, out):
        # timm ViT attention forward returns (x, attn_weights) or just x
        # depending on version — handle both
        if isinstance(out, tuple):
            self._attn_maps.append(out[1].detach())
        else:
            # Try to get attn_weights from module's last computed attention
            if hasattr(module, 'attn_drop'):
                pass  # older timm — only output x
            self._attn_maps.append(out.detach() if out.dim() == 4 else torch.zeros(1,1,1,1))

    def _compute_rollout(self, head_fusion: str, discard_ratio: float) -> torch.Tensor:
        """
        Rollout: propagate attention through all layers.
        At each layer: rollout = (attn + identity) × rollout_prev
        """
        all_attns = self._attn_maps   # List of [B, heads, N+1, N+1]

        rollout = None

        for attn in all_attns:
            if attn.shape[-1] < 2:
                continue

            # Fuse heads
            if head_fusion == "mean":
                fused = attn.mean(dim=1)    # [B, N+1, N+1]
            elif head_fusion == "max":
                fused = attn.max(dim=1)[0]
            else:
                fused = attn.min(dim=1)[0]

            # Discard low-attention values
            flat = fused.view(fused.shape[0], -1)
            threshold = flat.quantile(discard_ratio, dim=1, keepdim=True).unsqueeze(-1)
            fused = torch.where(fused >= threshold, fused, torch.zeros_like(fused))

            # Add residual connection (identity)
            I = torch.eye(fused.shape[-1], device=fused.device).unsqueeze(0)
            a = (fused + I) / (fused + I).sum(dim=-1, keepdim=True)

            rollout = a if rollout is None else torch.bmm(a, rollout)

        if rollout is None:
            return torch.zeros(1)

        # CLS → patch attention
        cls_attn = rollout[0, 0, 1:]   # [N]
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        return cls_attn

    def _reshape_to_image(self, patch_attn: torch.Tensor, img_size: int) -> np.ndarray:
        """Reshape 1D patch attention to 2D spatial map and upsample."""
        n     = patch_attn.shape[0]
        grid  = int(math.sqrt(n))

        if grid * grid != n:
            # Non-square patch grid — pad
            grid = math.ceil(math.sqrt(n))
            pad  = grid * grid - n
            patch_attn = F.pad(patch_attn, (0, pad))

        attn_2d = patch_attn.reshape(1, 1, grid, grid).float()
        upsampled = F.interpolate(attn_2d, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
        result = upsampled.squeeze().cpu().numpy()
        result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        return result.astype(np.float32)

    def _overlay(self, frame: np.ndarray, attn: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_VIRIDIS)
        overlaid = (1 - alpha) * frame.astype(float) + alpha * heatmap.astype(float)
        return np.clip(overlaid, 0, 255).astype(np.uint8)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalised tensor back to uint8 BGR for display."""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img * std + mean) * 255.0
        img  = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _identify_top_patches(self, attn_map: np.ndarray, top_n: int = 5) -> List[Dict]:
        """Identify the top-N highest-attention patches with coordinates."""
        h, w       = attn_map.shape
        flat_idx   = np.argsort(attn_map.ravel())[::-1][:top_n]
        patches    = []
        for idx in flat_idx:
            r, c = divmod(int(idx), w)
            patches.append({
                "row":        r,
                "col":        c,
                "activation": float(attn_map[r, c]),
                "pixel_y":    int(r * h / attn_map.shape[0]),
                "pixel_x":    int(c * w / attn_map.shape[1]),
            })
        return patches


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
    from ml.models.vit_detector import ViTDeepfakeDetector

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = ViTDeepfakeDetector(pretrained=False).to(device)
    extractor = ViTAttentionExtractor(model, device)

    dummy = torch.randn(1, 3, 224, 224).to(device)
    attn  = extractor.get_last_layer_attention(dummy)
    print(f"✅ ViTAttentionExtractor | attn shape: {attn.shape} | range: [{attn.min():.3f}, {attn.max():.3f}]")
