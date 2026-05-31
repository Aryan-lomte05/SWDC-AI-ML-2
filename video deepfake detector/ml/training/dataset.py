"""
Deepfake Detection Dataset Loader
===================================
Spec (bible §17 — Curriculum Learning + §5 Datasets):

Supports FaceForensics++, DFDC, Celeb-DF v2, WildDeepfake.

Dataset expected directory structure:
  data/
    real/
      ff_plusplus/  *.mp4 or pre-extracted *.jpg frames
      dfdc/
      celeb_df/
    fake/
      ff_plusplus/deepfakes/
      ff_plusplus/face2face/
      ff_plusplus/faceswap/
      ff_plusplus/neuraltextures/
      dfdc/
      celeb_df/
      wilddeepfake/

CurriculumDataset (bible §17):
  Epoch 0-30%  : Easy samples (raw/lightly compressed, obvious artefacts)
  Epoch 30-70% : Medium samples (standard compression, DFDC diversity)
  Epoch 70-100%: Hard samples (Celeb-DF high-quality, WildDeepfake in-the-wild)
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ml.preprocessing.face_processor import FaceProcessor
from ml.preprocessing.augmentation  import build_train_transform, build_val_transform, TemporalAugmenter

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224   # Unified input resolution


# ── Base Frame Dataset (single frame per sample) ─────────────────────────────

class DeepfakeFrameDataset(Dataset):
    """
    Frame-level dataset. Each sample is a single aligned face crop.

    Args:
        samples   : List of (image_path, label) or (frame_array, label)
        transform : Albumentations transform
        image_size: Resize target
    """

    def __init__(
        self,
        samples:    List[Tuple],
        transform=None,
        image_size: int = IMAGE_SIZE,
    ):
        self.samples    = samples
        self.transform  = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item, label = self.samples[idx]

        if isinstance(item, (str, Path)):
            frame = cv2.imread(str(item))
            if frame is None:
                frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = item   # Already numpy array

        frame = cv2.resize(frame, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=frame)
            tensor    = augmented["image"]   # Already ToTensorV2
        else:
            tensor = torch.from_numpy(
                frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return tensor, torch.tensor(label, dtype=torch.float32)


# ── Sequence Dataset (T frames per sample for LSTM) ──────────────────────────

class DeepfakeSequenceDataset(Dataset):
    """
    Sequence-level dataset. Each sample is T consecutive frames from a video.
    Used for training the TemporalDeepfakeDetector.

    Args:
        video_samples : List of (video_path, label)
        seq_len       : Number of frames per sequence (T)
        transform     : Frame-level transform
        temporal_aug  : TemporalAugmenter instance
    """

    def __init__(
        self,
        video_samples: List[Tuple],
        seq_len:       int = 30,
        transform=None,
        temporal_aug:  Optional[TemporalAugmenter] = None,
        image_size:    int = IMAGE_SIZE,
    ):
        self.video_samples = video_samples
        self.seq_len       = seq_len
        self.transform     = transform
        self.temporal_aug  = temporal_aug
        self.image_size    = image_size
        self._face_proc    = FaceProcessor(output_size=image_size)

    def __len__(self) -> int:
        return len(self.video_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.video_samples[idx]

        frames = self._extract_frames(str(video_path))

        if self.temporal_aug:
            frames = self.temporal_aug(frames)

        # Pad / truncate to seq_len
        if len(frames) < self.seq_len:
            frames += [frames[-1]] * (self.seq_len - len(frames))
        else:
            # Random crop of seq_len consecutive frames
            start  = random.randint(0, len(frames) - self.seq_len)
            frames = frames[start: start + self.seq_len]

        # Convert frames → tensors [T, 3, H, W]
        tensor_frames = []
        for f in frames:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if self.transform:
                f = self.transform(image=f)["image"]
            else:
                f = torch.from_numpy(
                    cv2.resize(f, (self.image_size, self.image_size))
                    .transpose(2, 0, 1)
                    .astype(np.float32) / 255.0
                )
            tensor_frames.append(f)

        seq_tensor = torch.stack(tensor_frames)   # [T, 3, H, W]

        return seq_tensor, torch.tensor(label, dtype=torch.float32)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap    = cv2.VideoCapture(video_path)
        frames = []
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps / 10))   # Target 10 FPS

        frame_idx = 0
        while cap.isOpened() and len(frames) < self.seq_len * 2:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(frame)
            frame_idx += 1

        cap.release()
        return frames if frames else [np.zeros((256, 256, 3), dtype=np.uint8)]


# ── Curriculum Dataset ────────────────────────────────────────────────────────

class CurriculumDataset(Dataset):
    """
    Curriculum learning wrapper (bible §17).

    Progressively introduces harder samples as training epochs advance:
      Phase 1 (epoch < 30%): Easy — raw/C0 FaceForensics++ fakes
      Phase 2 (epoch 30-70%): Medium — C23 compressed + DFDC diversity
      Phase 3 (epoch > 70%): Hard — Celeb-DF v2, WildDeepfake, adversarial

    Args:
        easy_samples   : List of (path, label) — obvious artefacts
        medium_samples : List of (path, label) — standard compression
        hard_samples   : List of (path, label) — near-realistic
        epoch          : Current training epoch
        max_epochs     : Total training epochs
        transform      : Data transform
    """

    def __init__(
        self,
        easy_samples:   List[Tuple],
        medium_samples: List[Tuple],
        hard_samples:   List[Tuple],
        epoch:          int = 0,
        max_epochs:     int = 50,
        transform=None,
        image_size:     int = IMAGE_SIZE,
    ):
        self.easy_samples   = easy_samples
        self.medium_samples = medium_samples
        self.hard_samples   = hard_samples
        self.epoch          = epoch
        self.max_epochs     = max_epochs
        self.transform      = transform
        self.image_size     = image_size

        self._update_active_samples()

    def set_epoch(self, epoch: int):
        """Call at the start of each training epoch to update sample mix."""
        self.epoch = epoch
        self._update_active_samples()
        logger.info(
            "Curriculum epoch %d/%d | difficulty=%.2f | active_samples=%d",
            epoch, self.max_epochs, self.difficulty, len(self.active_samples),
        )

    def _update_active_samples(self):
        self.difficulty = self.epoch / max(self.max_epochs, 1)

        if self.difficulty < 0.30:
            self.active_samples = self.easy_samples
            logger.debug("Curriculum: EASY phase")
        elif self.difficulty < 0.70:
            self.active_samples = self.easy_samples + self.medium_samples
            logger.debug("Curriculum: MEDIUM phase")
        else:
            self.active_samples = self.easy_samples + self.medium_samples + self.hard_samples
            logger.debug("Curriculum: HARD phase (all samples)")

    def __len__(self) -> int:
        return len(self.active_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item, label = self.active_samples[idx]

        if isinstance(item, (str, Path)):
            frame = cv2.imread(str(item))
            if frame is None:
                frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = item

        frame = cv2.resize(frame, (self.image_size, self.image_size))

        if self.transform:
            tensor = self.transform(image=frame)["image"]
        else:
            tensor = torch.from_numpy(
                frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return tensor, torch.tensor(label, dtype=torch.float32)


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_dataset_from_directory(
    data_root: str,
    split:     str = "train",
    train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    image_size: int = IMAGE_SIZE,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build train/val/test datasets from a structured data directory.

    Expected layout:
      data_root/real/  — subdirs or .jpg files
      data_root/fake/  — subdirs or .jpg files

    Returns: (train_dataset, val_dataset, test_dataset)
    """
    data_root = Path(data_root)
    real_paths = sorted((data_root / "real").rglob("*.jpg")) + \
                 sorted((data_root / "real").rglob("*.png"))
    fake_paths = sorted((data_root / "fake").rglob("*.jpg")) + \
                 sorted((data_root / "fake").rglob("*.png"))

    logger.info(
        "Dataset: %d real frames, %d fake frames", len(real_paths), len(fake_paths)
    )

    all_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.seed(seed)
    random.shuffle(all_samples)

    n     = len(all_samples)
    n_tr  = int(n * train_val_test[0])
    n_val = int(n * train_val_test[1])

    train_s = all_samples[:n_tr]
    val_s   = all_samples[n_tr: n_tr + n_val]
    test_s  = all_samples[n_tr + n_val:]

    train_tf = build_train_transform(image_size)
    val_tf   = build_val_transform(image_size)

    return (
        DeepfakeFrameDataset(train_s, train_tf, image_size),
        DeepfakeFrameDataset(val_s,   val_tf,   image_size),
        DeepfakeFrameDataset(test_s,  val_tf,   image_size),
    )


def build_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to balance real/fake classes.
    Critical for imbalanced datasets like DFDC (more fakes than reals).
    """
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_fake = sum(1 for l in labels if l == 1)
    n_real = len(labels) - n_fake

    weight_real = 1.0 / max(n_real, 1)
    weight_fake = 1.0 / max(n_fake, 1)

    weights = [weight_fake if l == 1 else weight_real for l in labels]
    return WeightedRandomSampler(
        weights     = weights,
        num_samples = len(weights),
        replacement = True,
    )
