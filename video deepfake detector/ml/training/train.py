"""
Training Script — Video Deepfake Detector
==========================================
Spec (bible §17):

Full training loop with:
  - AdamW optimizer + cosine annealing with warmup
  - Mixed precision (fp16) via torch.cuda.amp  →  RTX 4060 optimised
  - Gradient accumulation (effective_batch = 128)
  - Curriculum learning (easy → medium → hard)
  - MLflow experiment tracking
  - Best model checkpoint saving (by AUC-ROC)
  - TensorBoard logging
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.xception_detector import XceptionDetector
from ml.models.vit_detector       import ViTDeepfakeDetector
from ml.models.freq_analyzer      import FrequencyAnalyzer
from ml.models.temporal_lstm      import TemporalDeepfakeDetector
from ml.models.ensemble           import EnsembleDetector
from ml.training.losses           import DeepfakeLoss, SupConLoss
from ml.training.dataset          import (
    build_dataset_from_directory,
    build_balanced_sampler,
    CurriculumDataset,
)

logger = logging.getLogger(__name__)

# ── Default config path ───────────────────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "training_config.yaml"


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── LR Scheduler with Warmup ──────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Cosine annealing with linear warmup (spec §17)."""
    import math

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ── Adversarial Training (PGD) ────────────────────────────────────────────────

class AdversarialTrainer:
    """Spec (bible §19): PGD adversarial training."""

    def __init__(self, eps: float = 8 / 255, steps: int = 10):
        self.eps   = eps
        self.steps = steps

    def pgd_attack(self, model, frames, labels, criterion):
        """Projected Gradient Descent attack — generates adversarial examples."""
        delta = torch.zeros_like(frames).uniform_(-self.eps, self.eps)
        delta.requires_grad_(True)

        for _ in range(self.steps):
            with autocast():
                out = model(frames + delta)
                if isinstance(out, tuple):
                    out = out[0]
                loss = criterion(out, labels)

            loss.backward()
            with torch.no_grad():
                delta.data = (
                    delta.data + (self.eps / self.steps) * delta.grad.sign()
                ).clamp(-self.eps, self.eps)
            delta.grad.zero_()

        return (frames + delta).detach()


# ── Main Trainer ──────────────────────────────────────────────────────────────

class DeepfakeTrainer:
    """
    Full training orchestration.

    Usage:
        trainer = DeepfakeTrainer.from_config("configs/training_config.yaml")
        trainer.train()
    """

    def __init__(self, config: dict):
        self.cfg    = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info("Training device: %s", self.device)
        if self.device.type == "cuda":
            logger.info(
                "GPU: %s | VRAM: %.1f GB",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

        self.model     = None
        self.optimizer = None
        self.scheduler = None
        self.scaler    = GradScaler(enabled=self.cfg["training"].get("fp16", True))
        self.criterion = DeepfakeLoss(
            focal_gamma        = self.cfg["training"].get("focal_gamma", 2.0),
            label_smoothing    = self.cfg["training"].get("label_smoothing", 0.1),
            consistency_weight = self.cfg["training"].get("consistency_weight", 0.1),
        )
        self.adv_trainer   = AdversarialTrainer()
        self.best_auc      = 0.0
        self.output_dir    = Path(self.cfg.get("output_dir", "runs/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self._setup_mlflow()

    @classmethod
    def from_config(cls, config_path: str = str(DEFAULT_CONFIG)) -> "DeepfakeTrainer":
        return cls(load_config(config_path))

    # ── Setup ──────────────────────────────────────────────────────────────────

    def build_model(self) -> nn.Module:
        """Instantiate backbone model from config."""
        backbone = self.cfg["model"]["backbone"]
        pretrained = self.cfg["model"].get("pretrained", True)

        model_map = {
            "xception":      XceptionDetector,
            "vit":           ViTDeepfakeDetector,
            "freq":          FrequencyAnalyzer,
            "temporal_lstm": TemporalDeepfakeDetector,
        }

        if backbone not in model_map:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(model_map)}")

        model = model_map[backbone](pretrained=pretrained)
        logger.info("Built model: %s | pretrained=%s", backbone, pretrained)
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module):
        """AdamW with weight decay, separate param groups for backbone vs head."""
        cfg_opt  = self.cfg["optimization"]
        lr       = float(self.cfg["training"]["learning_rate"])
        wd       = float(self.cfg["training"].get("weight_decay", 1e-5))

        # Separate backbone (lower lr) from head (higher lr)
        backbone_params = []
        head_params     = []

        for name, param in model.named_parameters():
            if "classifier" in name or "head" in name or "fusion" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1},   # 10× lower for backbone
            {"params": head_params,     "lr": lr},
        ]

        return torch.optim.AdamW(
            param_groups,
            betas        = (cfg_opt.get("beta1", 0.9), cfg_opt.get("beta2", 0.999)),
            eps          = float(cfg_opt.get("eps", 1e-8)),
            weight_decay = wd,
        )

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self):
        """Full training loop."""
        cfg_t = self.cfg["training"]

        # Data
        train_ds, val_ds, _ = build_dataset_from_directory(
            self.cfg["data"]["root"],
            image_size = self.cfg["data"].get("image_size", 224),
        )
        sampler    = build_balanced_sampler(train_ds)
        train_dl   = DataLoader(
            train_ds,
            batch_size  = cfg_t["batch_size"],
            sampler     = sampler,
            num_workers = 4,
            pin_memory  = True,
        )
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

        # Model + optimizer
        self.model     = self.build_model()
        self.optimizer = self.build_optimizer(self.model)

        epochs        = cfg_t["epochs"]
        warmup_epochs = cfg_t.get("warmup_epochs", 5)
        accum_steps   = cfg_t.get("gradient_accumulation_steps", 4)

        total_steps  = epochs * len(train_dl)
        warmup_steps = warmup_epochs * len(train_dl)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        logger.info(
            "Training | epochs=%d | batch=%d | effective_batch=%d | steps=%d",
            epochs, cfg_t["batch_size"], cfg_t["batch_size"] * accum_steps, total_steps,
        )

        start_epoch = 0
        if self.cfg.get("resume_checkpoint"):
            start_epoch = self._load_checkpoint(self.cfg["resume_checkpoint"])

        for epoch in range(start_epoch, epochs):
            train_loss, train_auc = self._train_epoch(
                train_dl, epoch, epochs, accum_steps
            )
            val_loss, val_auc = self._validate(val_dl)

            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f train_auc=%.4f | val_loss=%.4f val_auc=%.4f",
                epoch + 1, epochs, train_loss, train_auc, val_loss, val_auc,
            )

            # Checkpoint
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self._save_checkpoint(epoch, val_auc, is_best=True)
                logger.info("★ New best AUC: %.4f — checkpoint saved", val_auc)
            
            # Save latest checkpoint for resuming
            self._save_checkpoint(epoch, val_auc, is_best=False)

            self._log_metrics(epoch, train_loss, train_auc, val_loss, val_auc)

        logger.info("Training complete. Best AUC: %.4f", self.best_auc)

    def _train_epoch(
        self,
        loader,
        epoch:       int,
        max_epochs:  int,
        accum_steps: int,
    ) -> tuple:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        step = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

        self.optimizer.zero_grad()

        for frames, labels in pbar:
            frames = frames.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Adversarial augmentation on half the batch (spec §19)
            if epoch >= int(max_epochs * 0.3):   # Start AT after 30% of training
                half = len(frames) // 2
                adv  = self.adv_trainer.pgd_attack(
                    self.model, frames[:half], labels[:half], self.criterion
                )
                frames = torch.cat([frames, adv])
                labels = torch.cat([labels, labels[:half]])

            with autocast(enabled=self.cfg["training"].get("fp16", True)):
                output = self.model(frames)
                pred   = output[0] if isinstance(output, tuple) else output
                loss   = self.criterion(pred, labels) / accum_steps

            self.scaler.scale(loss).backward()

            step += 1
            if step % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss   += loss.item() * accum_steps
            all_preds.extend(pred.squeeze().detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        return total_loss / len(loader), auc

    @torch.no_grad()
    def _validate(self, loader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for frames, labels in loader:
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.cfg["training"].get("fp16", True)):
                output = self.model(frames)
                pred   = output[0] if isinstance(output, tuple) else output
                loss   = self.criterion(pred, labels)

            total_loss   += loss.item()
            all_preds.extend(pred.squeeze().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        return total_loss / len(loader), auc

    def _save_checkpoint(self, epoch: int, auc: float, is_best: bool = False):
        state = {
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scaler_state":self.scaler.state_dict(),
            "scheduler":   self.scheduler.state_dict() if self.scheduler else None,
            "auc":         auc,
            "best_auc":    self.best_auc,
            "config":      self.cfg,
        }
        
        # Always save the latest for daily resuming
        latest_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(state, latest_path)
        
        if is_best:
            path = self.output_dir / f"best_model_epoch{epoch+1}_auc{auc:.4f}.pth"
            torch.save(state, path)
            torch.save(self.model.state_dict(), self.output_dir / "best.pth")

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Loads state from checkpoint and returns the epoch to resume from."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("Checkpoint %s not found. Starting from scratch.", path)
            return 0
            
        logger.info("Loading checkpoint from %s...", path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optim_state"])
        if "scaler_state" in checkpoint and hasattr(self, "scaler"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        if "scheduler" in checkpoint and self.scheduler and checkpoint["scheduler"]:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            
        self.best_auc = checkpoint.get("best_auc", checkpoint.get("auc", 0.0))
        resumed_epoch = checkpoint["epoch"] + 1
        
        logger.info("Successfully resumed from epoch %d (Best AUC: %.4f)", resumed_epoch, self.best_auc)
        return resumed_epoch

    def _setup_logging(self):
        logging.basicConfig(
            level   = logging.INFO,
            format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
        )

    def _setup_mlflow(self):
        try:
            import mlflow
            mlflow.set_experiment(
                self.cfg.get("experiment_name", "deepfake-detector")
            )
            self._mlflow = mlflow
        except ImportError:
            self._mlflow = None

    def _log_metrics(self, epoch, tr_loss, tr_auc, val_loss, val_auc):
        if self._mlflow:
            with self._mlflow.start_run(run_name=f"epoch_{epoch+1}", nested=True):
                self._mlflow.log_metrics({
                    "train_loss": tr_loss, "train_auc": tr_auc,
                    "val_loss":   val_loss, "val_auc":   val_auc,
                }, step=epoch)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Deepfake Detector")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config YAML path")
    parser.add_argument("--resume", action="store_true", help="Automatically resume from runs/latest_checkpoint.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume_checkpoint"] = str(Path(cfg.get("output_dir", "runs/")) / "latest_checkpoint.pth")

    trainer = DeepfakeTrainer(cfg)
    trainer.train()
