"""
Experiment Tracking and History Logger.
Persists epoch-by-epoch training metrics and plots convergence curves.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import settings
from utils.logger import experiment_logger

Optional_Path = Optional[Union[Path, str]]

class ExperimentManager:
    """Tracks training and validation metrics across epochs."""

    def __init__(self, experiment_name: str = "default_experiment"):
        self.experiment_name = experiment_name
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_recall": [],
            "learning_rate": []
        }

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        val_recall: float,
        lr: float
    ):
        """Records metrics for an epoch."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)
        self.history["val_f1"].append(val_f1)
        self.history["val_recall"].append(val_recall)
        self.history["learning_rate"].append(lr)

        experiment_logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | LR: {lr:.2e}"
        )

    def save_history_plots(self, output_path: Optional_Path = None):
        """Generates and saves convergence curve plots to reports/training_history.png."""
        out_p = Path(output_path or (settings.report_dir / "training_history.png"))
        out_p.parent.mkdir(parents=True, exist_ok=True)

        epochs = self.history["epoch"]
        if not epochs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o", color="blue")
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", marker="s", color="orange")
        ax1.set_title("Cross-Entropy Loss Progression")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Metrics curve
        ax2.plot(epochs, self.history["val_accuracy"], label="Val Accuracy", marker="^", color="green")
        ax2.plot(epochs, self.history["val_f1"], label="Val F1-Score", marker="d", color="red")
        ax2.plot(epochs, self.history["val_recall"], label="Val Fight Recall", marker="x", color="purple")
        ax2.axhline(y=settings.target_accuracy, color="gray", linestyle=":", label=f"Target ({settings.target_accuracy*100:.0f}%)")
        ax2.set_title("Validation Performance Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0.0, 1.05)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_p, dpi=200)
        plt.close()

