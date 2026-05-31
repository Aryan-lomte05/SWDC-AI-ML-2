"""
Augmentation Pipeline
======================
Spec (bible §5 — Data Augmentation Strategy):

  - Compression artifacts (simulate social media H.264 CRF 23–40)
  - Geometric: RandomCrop, HorizontalFlip, RandomRotation(±10°)
  - Color/lighting: ColorJitter, RandomGrayscale(5%)
  - Noise: GaussianNoise, JPEGCompression(50–95)
  - Temporal: FrameDropout(10%), TemporalJitter(±2 frames)

Uses albumentations for speed (10–50× faster than torchvision).
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False
    logger.warning("albumentations not installed; using basic OpenCV augmentation.")


# ── Albumentations Pipeline ───────────────────────────────────────────────────

def build_train_transform(image_size: int = 224) -> "A.Compose":
    """
    Full training augmentation pipeline for face crops.
    Simulates all real-world degradations deepfakes go through.
    """
    assert _HAS_ALBUMENTATIONS, "Install albumentations: pip install albumentations"

    return A.Compose([
        # ── Geometric ──────────────────────────────────────────────────
        A.RandomResizedCrop(
            height=image_size, width=image_size,
            scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5,
        ),

        # ── Color / Lighting ───────────────────────────────────────────
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.7
        ),
        A.ToGray(p=0.05),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # ── Blur & Sharpness ───────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # ── Noise ──────────────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(0, 0.02 * 255 ** 2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),

        # ── Compression (simulate social media) ───────────────────────
        A.ImageCompression(
            quality_lower=50, quality_upper=95,
            compression_type=A.ImageCompression.ImageCompressionType.JPEG,
            p=0.5,
        ),

        # ── Final ──────────────────────────────────────────────────────
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_val_transform(image_size: int = 224) -> "A.Compose":
    """
    Validation / inference transform — only resize + normalise.
    """
    assert _HAS_ALBUMENTATIONS, "Install albumentations"

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Temporal Augmentation ─────────────────────────────────────────────────────

class TemporalAugmenter:
    """
    Augmentations that operate on a SEQUENCE of frames.
    Applied during training to improve temporal model robustness.
    """

    def __init__(
        self,
        dropout_prob:    float = 0.1,   # Probability of dropping a frame
        jitter_max:      int   = 2,     # Max frames to shift sequence
        reverse_prob:    float = 0.1,   # Probability of reversing sequence
        speed_prob:      float = 0.2,   # Probability of temporal speed change
    ):
        self.dropout_prob = dropout_prob
        self.jitter_max   = jitter_max
        self.reverse_prob = reverse_prob
        self.speed_prob   = speed_prob

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal augmentations to a frame sequence."""
        frames = list(frames)

        # Temporal Jitter — random start offset
        if self.jitter_max > 0 and len(frames) > self.jitter_max:
            shift = random.randint(0, self.jitter_max)
            frames = frames[shift:]

        # Frame Dropout — randomly drop frames (model must be robust)
        if self.dropout_prob > 0:
            frames = [f for f in frames if random.random() > self.dropout_prob]
            if len(frames) == 0:
                frames = [frames[0]] if frames else []

        # Reverse sequence (temporal flip)
        if random.random() < self.reverse_prob:
            frames = frames[::-1]

        # Speed change — subsample to simulate faster/slower playback
        if random.random() < self.speed_prob:
            factor = random.choice([0.5, 0.75, 1.25, 1.5])
            step   = max(1, round(factor))
            frames = frames[::step]

        return frames


# ── Video Compression Simulation ─────────────────────────────────────────────

def simulate_video_compression(
    frames: List[np.ndarray],
    crf:    int = 28,
    codec:  str = "mp4v",
) -> List[np.ndarray]:
    """
    Simulate H.264 video compression artifacts.
    Writes frames to a temp file, re-reads them.

    Args:
        frames: List of BGR numpy frames
        crf:    Quality factor (23=good, 40=heavy compression)
        codec:  FourCC codec string

    Returns:
        Compressed frames as numpy arrays
    """
    import tempfile, os

    if len(frames) == 0:
        return frames

    h, w = frames[0].shape[:2]
    tmp  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(tmp.name, fourcc, 10.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    # Re-read
    cap = cv2.VideoCapture(tmp.name)
    out = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.append(frame)
    cap.release()
    os.unlink(tmp.name)

    return out if out else frames


# ── Numpy fallback (no albumentations) ───────────────────────────────────────

def basic_transform(frame: np.ndarray, image_size: int = 224) -> np.ndarray:
    """
    Minimal transform using only OpenCV + numpy.
    Used when albumentations is not installed.
    """
    frame = cv2.resize(frame, (image_size, image_size))
    frame = frame.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame[..., ::-1] - mean) / std   # BGR→RGB + normalise

    return frame.transpose(2, 0, 1)   # [C, H, W]
