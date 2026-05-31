"""
Deepfake Detection Loss Function
==================================
Spec (bible §17):

Standard BCE alone is insufficient for deepfake detection:
  1. Class imbalance (more fakes than reals in DFDC)
  2. Hard negatives — near-perfect deepfakes that look real
  3. Label noise — some training labels are uncertain

Solution: Focal Loss + Label Smoothing + Consistency Regularisation

focal_loss     : Downweights easy examples, focuses training on hard
                 near-boundary cases. gamma=2.0 (Retinaface paper default)
label_smoothing: Prevents overconfidence; a label of 1.0 becomes 0.9,
                 which improves calibration for uncertainty estimation.
consistency_reg: Adjacent frames of the same video should have similar
                 scores — penalises inconsistent frame-level predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DeepfakeLoss(nn.Module):
    """
    Combined focal loss + label smoothing + temporal consistency regularisation.

    Args:
        focal_gamma       (float): Focal loss gamma. 0 = standard BCE.
        label_smoothing   (float): Smoothing factor. 0 = no smoothing.
        consistency_weight(float): Weight for temporal consistency term.
        pos_weight        (float): Weight for positive (fake) class.
                                   >1 if reals are underrepresented.
    """

    def __init__(
        self,
        focal_gamma:        float = 2.0,
        label_smoothing:    float = 0.1,
        consistency_weight: float = 0.1,
        pos_weight:         float = 1.0,
    ):
        super().__init__()
        self.gamma              = focal_gamma
        self.label_smoothing    = label_smoothing
        self.consistency_weight = consistency_weight

        if pos_weight != 1.0:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

        logger.info(
            "DeepfakeLoss | gamma=%.1f | smoothing=%.2f | consistency_w=%.2f | pos_w=%.1f",
            focal_gamma, label_smoothing, consistency_weight, pos_weight,
        )

    def forward(
        self,
        pred:   torch.Tensor,     # [B, 1] predicted probabilities
        target: torch.Tensor,     # [B]    binary labels (0=real, 1=fake)
        frame_preds: torch.Tensor = None,  # [B, T] per-frame predictions (optional)
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            pred        : [B, 1] — video-level fake probabilities
            target      : [B]   — binary labels
            frame_preds : [B, T] — per-frame predictions for consistency loss

        Returns:
            total_loss : scalar tensor
        """
        pred   = pred.squeeze(-1).clamp(1e-6, 1 - 1e-6)   # [B]
        target = target.float()

        # ── Label Smoothing ───────────────────────────────────────────────
        smooth_target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # ── Binary Focal Loss ─────────────────────────────────────────────
        bce = F.binary_cross_entropy(pred, smooth_target, reduction="none")  # [B]

        # pt = probability of the correct class
        pt           = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss   = (focal_weight * bce).mean()

        # ── Temporal Consistency Regularisation ───────────────────────────
        consistency_loss = torch.tensor(0.0, device=pred.device)
        if frame_preds is not None and self.consistency_weight > 0:
            consistency_loss = self._consistency_reg(frame_preds)

        total = focal_loss + self.consistency_weight * consistency_loss

        return total

    def _consistency_reg(self, frame_preds: torch.Tensor) -> torch.Tensor:
        """
        Penalise large frame-to-frame prediction jumps within the same video.
        Encourages smooth temporal predictions.

        frame_preds: [B, T] per-frame probabilities
        """
        if frame_preds.shape[1] < 2:
            return torch.tensor(0.0, device=frame_preds.device)

        diffs = frame_preds[:, 1:] - frame_preds[:, :-1]   # [B, T-1]
        return (diffs ** 2).mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls real face embeddings together and pushes fake embeddings apart.
    Used as an auxiliary loss on XceptionNet embeddings to improve
    generalisation across deepfake methods.

    temp (float): Temperature scaling. Lower = harder negatives.
    """

    def __init__(self, temp: float = 0.07):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        features: torch.Tensor,  # [B, D] L2-normalised embeddings
        labels:   torch.Tensor,  # [B] binary labels
    ) -> torch.Tensor:
        B = features.shape[0]
        features = F.normalize(features, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.mm(features, features.T) / self.temp   # [B, B]

        # Mask: same class pairs (positive pairs)
        labels    = labels.view(-1, 1)
        pos_mask  = (labels == labels.T).float()
        neg_mask  = 1.0 - pos_mask
        # Remove diagonal (self-similarity)
        eye       = torch.eye(B, device=features.device)
        pos_mask  = pos_mask * (1 - eye)

        # Log-softmax
        exp_sim   = torch.exp(sim_matrix) * (1 - eye)
        log_prob  = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability over positive pairs
        num_pos = pos_mask.sum(dim=1)
        loss    = -(pos_mask * log_prob).sum(dim=1) / (num_pos.clamp(min=1))

        return loss.mean()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    criterion = DeepfakeLoss(focal_gamma=2.0, label_smoothing=0.1)
    pred      = torch.sigmoid(torch.randn(8, 1))
    target    = torch.randint(0, 2, (8,)).float()
    fps       = torch.sigmoid(torch.randn(8, 30))   # 30 frames

    loss = criterion(pred, target, fps)
    print(f"✅ DeepfakeLoss | loss={loss.item():.4f}")

    sup_con = SupConLoss(temp=0.07)
    feats   = torch.randn(8, 2048)
    sc_loss = sup_con(feats, target.long())
    print(f"✅ SupConLoss   | loss={sc_loss.item():.4f}")
