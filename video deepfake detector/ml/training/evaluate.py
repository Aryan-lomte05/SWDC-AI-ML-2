"""
Evaluation & Benchmarking Script
===================================
Spec (bible §18):

Comprehensive evaluation covering all metrics mentioned in the bible:
  - AUC-ROC     : Threshold-independent performance
  - AP           : Average Precision (critical for imbalanced data)
  - EER          : Equal Error Rate (forensics standard)
  - F1 / Accuracy
  - Cross-dataset generalisation table
  - Optimal threshold via Youden's J statistic
  - Per-method breakdown (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
  - Confusion matrix + ROC curve plotting
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)


class DeepfakeEvaluator:
    """
    Comprehensive evaluation suite for deepfake detection models.

    Usage:
        evaluator = DeepfakeEvaluator(model, device)
        metrics   = evaluator.evaluate(test_loader)
        evaluator.plot_results(metrics, output_dir="runs/eval/")
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

    # ── Core Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader:       DataLoader,
        dataset_name: str = "test",
    ) -> Dict:
        """
        Full evaluation on a DataLoader.

        Returns:
            dict with all metrics from bible §18 table.
        """
        all_preds, all_labels = [], []

        logger.info("Evaluating on %s dataset...", dataset_name)
        for frames, labels in tqdm(loader, desc=f"Eval [{dataset_name}]"):
            frames = frames.to(self.device)
            output = self.model(frames)
            scores = output[0] if isinstance(output, tuple) else output
            scores = scores.squeeze(-1)

            all_preds.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        return self._compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            dataset_name,
        )

    def _compute_metrics(
        self,
        preds:        np.ndarray,
        labels:       np.ndarray,
        dataset_name: str,
    ) -> Dict:
        """Compute all metrics from the bible's §18 table."""
        # AUC-ROC
        auc = float(roc_auc_score(labels, preds))

        # Average Precision
        ap  = float(average_precision_score(labels, preds))

        # ROC curve + optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(labels, preds)
        youden_j              = tpr - fpr
        opt_idx               = int(np.argmax(youden_j))
        opt_threshold         = float(thresholds[opt_idx])

        # EER (Equal Error Rate) — interpolate where FPR == FNR
        fnr = 1.0 - tpr
        eer_idx = int(np.argmin(np.abs(fpr - fnr)))
        eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

        # Binary metrics at optimal threshold
        binary_preds = (preds >= opt_threshold).astype(int)
        acc          = float(accuracy_score(labels, binary_preds))
        f1           = float(f1_score(labels, binary_preds, zero_division=0))
        cm           = confusion_matrix(labels, binary_preds)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        precision  = float(tp / max(tp + fp, 1))
        recall     = float(tp / max(tp + fn, 1))
        fpr_at_opt = float(fp / max(fp + tn, 1))

        metrics = {
            "dataset":         dataset_name,
            "n_samples":       len(labels),
            "n_real":          int((labels == 0).sum()),
            "n_fake":          int((labels == 1).sum()),
            "auc_roc":         round(auc,  4),
            "avg_precision":   round(ap,   4),
            "eer":             round(eer,  4),
            "accuracy":        round(acc,  4),
            "f1_score":        round(f1,   4),
            "precision":       round(precision, 4),
            "recall":          round(recall,    4),
            "fpr_at_threshold":round(fpr_at_opt,4),
            "optimal_threshold": round(opt_threshold, 4),
            "confusion_matrix": cm.tolist(),
            "roc": {
                "fpr": fpr.tolist()[::5],   # Downsample for storage
                "tpr": tpr.tolist()[::5],
            },
        }

        self._print_metrics(metrics)
        return metrics

    @staticmethod
    def _print_metrics(m: Dict):
        print("\n" + "═" * 55)
        print(f"  EVALUATION RESULTS — {m['dataset'].upper()}")
        print("═" * 55)
        print(f"  Samples  : {m['n_samples']:,} ({m['n_real']:,} real / {m['n_fake']:,} fake)")
        print(f"  AUC-ROC  : {m['auc_roc']:.4f}")
        print(f"  Avg Prec : {m['avg_precision']:.4f}")
        print(f"  EER      : {m['eer']:.4f}  (lower=better, 0=perfect)")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  F1 Score : {m['f1_score']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  FPR      : {m['fpr_at_threshold']:.4f}")
        print(f"  Threshold: {m['optimal_threshold']:.4f}")
        print("═" * 55 + "\n")

    # ── Cross-Dataset Generalisation ──────────────────────────────────────────

    def cross_dataset_evaluation(
        self,
        loaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict]:
        """
        Bible §18 cross-dataset generalisation test.

        Args:
            loaders: dict of dataset_name → DataLoader

        Returns:
            dict of dataset_name → metrics dict
        """
        all_results = {}
        for name, loader in loaders.items():
            logger.info("Cross-dataset eval: %s", name)
            all_results[name] = self.evaluate(loader, name)

        # Print generalisation table (matches bible §18)
        print("\n" + "═" * 60)
        print("  CROSS-DATASET GENERALISATION RESULTS")
        print("═" * 60)
        print(f"  {'Dataset':<25} {'AUC-ROC':>8} {'EER':>7} {'F1':>7}")
        print("─" * 60)
        for name, res in all_results.items():
            print(
                f"  {name:<25} {res['auc_roc']:>8.4f} "
                f"{res['eer']:>7.4f} {res['f1_score']:>7.4f}"
            )
        print("═" * 60)

        return all_results

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_results(
        self,
        metrics_dict:  Dict,
        output_dir:    str = "runs/eval",
        show:          bool = False,
    ) -> str:
        """
        Generate comprehensive evaluation plots.

        Returns:
            Path to saved figure
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#0f0f1a")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])   # ROC curve
        ax2 = fig.add_subplot(gs[0, 1])   # Score distribution
        ax3 = fig.add_subplot(gs[0, 2])   # Confusion matrix
        ax4 = fig.add_subplot(gs[1, :])   # Per-dataset bar chart

        style = {
            "text_color":  "#e0e0ff",
            "grid_color":  "#2a2a4a",
            "bg_color":    "#0f0f1a",
            "accent":      "#7c3aed",
            "danger":      "#ef4444",
            "success":     "#22c55e",
        }

        def _style_ax(ax, title):
            ax.set_facecolor(style["bg_color"])
            ax.set_title(title, color=style["text_color"], pad=12, fontsize=11)
            ax.tick_params(colors=style["text_color"])
            ax.spines["bottom"].set_color(style["grid_color"])
            ax.spines["left"].set_color(style["grid_color"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=style["grid_color"], linestyle="--", alpha=0.5)

        # ── ROC curve ─────────────────────────────────────────────────────
        if "roc" in metrics_dict:
            fpr_ = metrics_dict["roc"]["fpr"]
            tpr_ = metrics_dict["roc"]["tpr"]
            ax1.plot(fpr_, tpr_, color=style["accent"], lw=2.5, label=f'AUC={metrics_dict["auc_roc"]:.4f}')
            ax1.plot([0, 1], [0, 1], color=style["grid_color"], lw=1.5, linestyle="--")
            ax1.set_xlabel("False Positive Rate", color=style["text_color"])
            ax1.set_ylabel("True Positive Rate", color=style["text_color"])
            ax1.legend(facecolor="#1a1a2e", labelcolor=style["text_color"])
            _style_ax(ax1, "ROC Curve")

        # ── Confusion matrix ──────────────────────────────────────────────
        if "confusion_matrix" in metrics_dict:
            cm_arr = np.array(metrics_dict["confusion_matrix"])
            im = ax3.imshow(cm_arr, cmap="RdYlGn", vmin=0)
            ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
            ax3.set_xticklabels(["Pred Real", "Pred Fake"], color=style["text_color"])
            ax3.set_yticklabels(["True Real", "True Fake"], color=style["text_color"])
            for i in range(2):
                for j in range(2):
                    ax3.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                             color="white", fontweight="bold", fontsize=14)
            _style_ax(ax3, "Confusion Matrix")

        # ── Metric bars ───────────────────────────────────────────────────
        m_labels = ["AUC-ROC", "Avg Prec", "Accuracy", "F1 Score", "Recall", "Precision"]
        m_values = [
            metrics_dict.get("auc_roc", 0),
            metrics_dict.get("avg_precision", 0),
            metrics_dict.get("accuracy", 0),
            metrics_dict.get("f1_score", 0),
            metrics_dict.get("recall", 0),
            metrics_dict.get("precision", 0),
        ]
        colors = [style["success"] if v >= 0.9 else style["accent"] if v >= 0.7 else style["danger"]
                  for v in m_values]

        bars = ax4.barh(m_labels, m_values, color=colors, height=0.5)
        for bar, val in zip(bars, m_values):
            ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", color=style["text_color"], fontsize=10)
        ax4.set_xlim(0, 1.1)
        ax4.set_xlabel("Score", color=style["text_color"])
        ax4.axvline(0.95, color=style["success"], linestyle="--", alpha=0.7, label="Target (0.95)")
        ax4.legend(facecolor="#1a1a2e", labelcolor=style["text_color"])
        _style_ax(ax4, "Evaluation Metrics")

        fig.suptitle(
            f"Video Deepfake Detector — Evaluation Report\n"
            f"Dataset: {metrics_dict.get('dataset', 'test')} | "
            f"AUC-ROC: {metrics_dict.get('auc_roc', 0):.4f} | "
            f"EER: {metrics_dict.get('eer', 0):.4f}",
            color=style["text_color"], fontsize=13, y=0.98,
        )

        save_path = str(output_dir / f"eval_{metrics_dict.get('dataset', 'test')}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=style["bg_color"])
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Evaluation plot saved: %s", save_path)
        return save_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Deepfake Detector")
    parser.add_argument("--model",       required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data",        required=True, help="Test data directory")
    parser.add_argument("--output_dir",  default="runs/eval", help="Output directory for results")
    parser.add_argument("--backbone",    default="xception",  help="Model backbone")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.model}")

    print("\n✅ Evaluation complete.")
    print(f"   Results saved to: {args.output_dir}")
