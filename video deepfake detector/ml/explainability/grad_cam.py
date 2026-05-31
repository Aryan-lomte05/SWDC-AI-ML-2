"""
Grad-CAM Explainability
========================
Spec (bible §9.1):

Generate class activation maps showing WHICH spatial regions of the face
triggered the deepfake classification decision. Makes the model's reasoning
transparent and legally defensible.

Algorithm (Selvaraju et al., 2017 — Grad-CAM):
  1. Forward pass → record activations at target convolutional layer
  2. Backward pass → record gradients flowing back through target layer
  3. Weight each activation channel by its mean gradient (global avg pool)
  4. Sum weighted activations → raw CAM
  5. ReLU (keep only positive contributions)
  6. Bilinear upsample to input resolution
  7. Normalise to [0, 1]
  8. Overlay as JET colourmap heatmap on original frame

Why this matters for patent/legal:
  Every flagged frame gets a forensic heatmap showing exactly which
  facial region (eyes, lips, cheeks, blending boundary) triggered the
  detection. This level of evidence is unprecedented in consumer tools.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN-based detectors.

    Compatible with XceptionDetector and FrequencyAnalyzer.
    For ViT, use ViTAttentionExtractor instead.

    Args:
        model        : The CNN model (XceptionDetector instance)
        target_layer : The nn.Module conv layer to hook into
        device       : torch.device
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device=None):
        self.model        = model
        self.target_layer = target_layer
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor]  = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

        logger.debug("GradCAM hooks registered on %s", target_layer.__class__.__name__)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core Generation ───────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : [1, 3, H, W] — single face crop, normalised
            target_class : Class index to explain (0 = fake probability)

        Returns:
            cam : [H, W] numpy array, values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]   # (logit, embedding) → use logit

        # Backward pass
        self.model.zero_grad()
        output[:, target_class].sum().backward(retain_graph=True)

        # Grad-CAM computation
        gradients   = self._gradients   # [1, C, h, w]
        activations = self._activations  # [1, C, h, w]

        if gradients is None or activations is None:
            logger.warning("GradCAM: hooks did not capture gradients/activations")
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((H, W), dtype=np.float32)

        # Global average pool gradients → per-channel weights [1, C, 1, 1]
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam  = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.astype(np.float32)

    def generate_batch(
        self,
        face_crops: List[torch.Tensor],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Generate Grad-CAM for top-k most suspicious face crops.

        Args:
            face_crops : List of [1, 3, H, W] tensors
            top_k      : Number of frames to generate heatmaps for

        Returns:
            List of dicts: {frame_idx, cam, score}
        """
        results = []
        for i, crop in enumerate(face_crops[:top_k]):
            cam   = self.generate(crop)
            score = float(self.model(crop.to(self.device))[0].item()
                         if isinstance(self.model(crop.to(self.device)), tuple)
                         else self.model(crop.to(self.device)).item())
            results.append({"frame_idx": i, "cam": cam, "score": score})
        return results

    # ── Overlay Rendering ─────────────────────────────────────────────────────

    @staticmethod
    def overlay_on_frame(
        frame: np.ndarray,
        cam:   np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto original frame.

        Args:
            frame    : BGR numpy array [H, W, 3]
            cam      : [H, W] float array in [0, 1]
            alpha    : Heatmap opacity (0=invisible, 1=solid)
            colormap : OpenCV colourmap constant

        Returns:
            overlaid : BGR numpy array [H, W, 3]
        """
        # Resize CAM to match frame if needed
        if cam.shape != frame.shape[:2]:
            cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)  # BGR
        frame_f = frame.astype(np.float32)
        heat_f  = heatmap.astype(np.float32)

        overlaid = (1.0 - alpha) * frame_f + alpha * heat_f
        return np.clip(overlaid, 0, 255).astype(np.uint8)

    @staticmethod
    def save_heatmap(
        frame:      np.ndarray,
        cam:        np.ndarray,
        output_path: str,
        alpha:      float = 0.45,
    ) -> str:
        """Save Grad-CAM overlaid frame to disk. Returns the saved path."""
        overlaid = GradCAM.overlay_on_frame(frame, cam, alpha)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, overlaid)
        return output_path

    @staticmethod
    def annotate_regions(
        frame: np.ndarray,
        cam:   np.ndarray,
        threshold: float = 0.6,
    ) -> Dict:
        """
        Identify which facial regions (eyes, nose, mouth, cheeks) are
        most highlighted in the CAM. Returns region importance scores
        for the evidence report.

        Args:
            frame     : [H, W, 3]
            cam       : [H, W] in [0, 1]
            threshold : Activation threshold for region flagging

        Returns:
            dict: region_name -> mean_activation
        """
        h, w = cam.shape

        regions = {
            "forehead":     cam[int(0.05*h): int(0.25*h), int(0.2*w): int(0.8*w)],
            "left_eye":     cam[int(0.25*h): int(0.45*h), int(0.1*w): int(0.40*w)],
            "right_eye":    cam[int(0.25*h): int(0.45*h), int(0.60*w): int(0.9*w)],
            "nose":         cam[int(0.40*h): int(0.60*h), int(0.35*w): int(0.65*w)],
            "left_cheek":   cam[int(0.45*h): int(0.70*h), int(0.05*w): int(0.35*w)],
            "right_cheek":  cam[int(0.45*h): int(0.70*h), int(0.65*w): int(0.95*w)],
            "mouth":        cam[int(0.65*h): int(0.85*h), int(0.25*w): int(0.75*w)],
            "jaw_left":     cam[int(0.75*h): int(0.95*h), int(0.05*w): int(0.40*w)],
            "jaw_right":    cam[int(0.75*h): int(0.95*h), int(0.60*w): int(0.95*w)],
        }

        scores = {}
        for name, region in regions.items():
            scores[name] = float(region.mean()) if region.size > 0 else 0.0

        # Sort by activation
        sorted_regions = sorted(scores.items(), key=lambda x: -x[1])
        flagged = [(r, s) for r, s in sorted_regions if s > threshold]

        return {
            "region_scores":  scores,
            "top_regions":    sorted_regions[:3],
            "flagged_regions": flagged,
        }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
    from ml.models.xception_detector import XceptionDetector

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = XceptionDetector(pretrained=False).to(device)
    target = model.get_target_layer()
    gcam   = GradCAM(model, target, device)

    dummy  = torch.randn(1, 3, 299, 299).to(device)
    cam    = gcam.generate(dummy)
    print(f"✅ GradCAM | cam shape: {cam.shape} | range: [{cam.min():.3f}, {cam.max():.3f}]")

    regions = GradCAM.annotate_regions(np.zeros((299, 299, 3), dtype=np.uint8), cam)
    print(f"   top regions: {regions['top_regions']}")
    gcam.remove_hooks()
