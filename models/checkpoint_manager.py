"""
Model Checkpoint Manager.
Handles saving and loading of model weights and complete experiment state metadata
(optimizer, scheduler, metrics, hardware config, and software versions).
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from config.settings import settings
from config.labels import ID_TO_LABEL, LABEL_TO_ID
from utils.logger import app_logger

class CheckpointManager:
    """Manages serialization and deserialization of model checkpoints."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = Path(checkpoint_dir or settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, Any],
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        hardware_profile: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None
    ) -> Path:
        """
        Saves a comprehensive training checkpoint containing model weights and experiment metadata.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required to save checkpoints.")

        checkpoint_data = {
            "epoch": epoch,
            "architecture": "VideoMAE-V2-Base-Classifier",
            "model_id": settings.model_id,
            "id_to_label": ID_TO_LABEL,
            "label_to_id": LABEL_TO_ID,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "hardware_profile": hardware_profile,
            "random_seed": settings.random_seed,
            "python_version": sys.version,
            "torch_version": torch.__version__ if torch else "N/A"
        }

        # 1. Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pth"
        torch.save(checkpoint_data, latest_path)

        # 2. Save best model if flagged
        best_path = self.checkpoint_dir / "best_model.pth"
        if is_best:
            torch.save(checkpoint_data, best_path)
            app_logger.info(f"Updated best model checkpoint at: {best_path} (F1: {metrics.get('f1_score', 'N/A')})")

        # 3. Optional tag or experiment specific file
        if tag:
            tagged_path = self.checkpoint_dir / f"checkpoint_{tag}.pth"
            torch.save(checkpoint_data, tagged_path)
            return tagged_path

        return latest_path

    def load_checkpoint(
        self,
        model: Any,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Loads model weights from checkpoint into the provided model instance.
        If checkpoint_path is None, looks for best_model.pth, then latest_model.pth.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required to load checkpoints.")

        target_path = None
        if checkpoint_path and Path(checkpoint_path).exists():
            target_path = Path(checkpoint_path)
        else:
            best_p = self.checkpoint_dir / "best_model.pth"
            latest_p = self.checkpoint_dir / "latest_model.pth"
            if best_p.exists():
                target_path = best_p
            elif latest_p.exists():
                target_path = latest_p

        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_dir}")

        app_logger.info(f"Loading checkpoint from: {target_path} onto {device}")
        checkpoint = torch.load(target_path, map_location=device)

        # Handle potential key discrepancies
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "architecture": checkpoint.get("architecture", "Unknown"),
            "model_id": checkpoint.get("model_id", settings.model_id),
            "checkpoint_file": str(target_path)
        }

        return model, metadata

    def get_available_checkpoints(self) -> Dict[str, Path]:
        """Scans checkpoint directory and returns dictionary of available checkpoints."""
        found = {}
        for p in self.checkpoint_dir.glob("*.pth"):
            found[p.stem] = p
        return found

