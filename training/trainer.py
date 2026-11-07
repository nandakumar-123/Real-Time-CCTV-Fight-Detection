"""
Hardware-Aware Staged Trainer for CCTV Fight Detection.
Implements AdamW optimizer with warmup + cosine decay, weighted cross-entropy,
gradient accumulation, automatic OOM protection, and validation check-points.
"""

import math
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from config.settings import settings
from config.labels import NUM_CLASSES
from models.classifier import CCTVFightClassifier
from models.checkpoint_manager import CheckpointManager
from training.early_stopping import EarlyStopping
from training.experiment_manager import ExperimentManager
from utils.logger import train_logger
from utils.resource_manager import resource_manager
from utils.metrics import calculate_classification_metrics
from utils.hardware_detector import detect_hardware

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Creates a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

class CCTVTrainer:
    """Orchestrates staged fine-tuning of VideoMAE V2 Base."""

    def __init__(
        self,
        model: CCTVFightClassifier,
        train_loader,
        val_loader,
        epochs: int = 30,
        stage: int = 1,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.stage = stage

        # Hardware detection & profile
        self.profile = detect_hardware()
        self.device = self.profile.device_type
        self.model.to(self.device)

        # Staged unfreezing
        if stage == 1:
            self.model.freeze_backbone()
        elif stage == 2:
            self.model.unfreeze_final_blocks(num_blocks=2)
        else:
            self.model.unfreeze_all()

        # Loss function with optional class weighting
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer with differential learning rates
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.profile.recommended_lr_backbone,
            lr_head=self.profile.recommended_lr_head,
            weight_decay=0.05
        )
        self.optimizer = AdamW(param_groups)

        # Scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        # AMP GradScaler
        self.use_amp = self.profile.use_mixed_precision and (self.device == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.grad_accum_steps = self.profile.gradient_accumulation_steps
        self.checkpoint_manager = CheckpointManager()
        self.experiment_manager = ExperimentManager(experiment_name=f"stage_{stage}")
        self.early_stopping = EarlyStopping(patience=7, mode="max")

    def train_epoch(self, epoch: int) -> float:
        """Executes one training epoch with gradient accumulation and OOM resilience."""
        self.model.train()
        total_loss = 0.0
        successful_steps = 0
        self.optimizer.zero_grad()

        for step, (tensors, labels, _) in enumerate(self.train_loader):
            # Check resource thresholds
            is_safe, msg = resource_manager.check_safety_thresholds()
            if not is_safe:
                train_logger.warning(f"Resource safety warning: {msg}. Freeing cache.")
                resource_manager.release_memory()

            try:
                tensors = tensors.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(tensors)
                    loss = self.criterion(logits, labels)
                    loss_scaled = loss / self.grad_accum_steps

                self.scaler.scale(loss_scaled).backward()

                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                total_loss += loss.item()
                successful_steps += 1

            except (RuntimeError, Exception) as e:
                err_str = str(e).lower()
                if "out of memory" in err_str or "oom" in err_str:
                    train_logger.warning(f"CUDA OOM caught at epoch {epoch}, step {step}. Recovering...")
                    self.optimizer.zero_grad()
                    resource_manager.release_memory()
                    continue
                else:
                    train_logger.error(f"Unexpected training step error: {e}")
                    raise e

        return total_loss / max(1, successful_steps)

    def validate(self) -> Dict[str, Any]:
        """Runs validation and computes performance metrics."""
        self.model.eval()
        total_loss = 0.0
        all_true = []
        all_pred = []
        all_prob = []

        with torch.no_grad():
            for tensors, labels, _ in self.val_loader:
                tensors = tensors.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    logits = self.model(tensors)
                    loss = self.criterion(logits, labels)

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                total_loss += loss.item()
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())
                all_prob.extend(probs[:, 1].cpu().numpy().tolist())

        val_metrics = calculate_classification_metrics(all_true, all_pred, all_prob)
        val_metrics["val_loss"] = round(total_loss / max(1, len(self.val_loader)), 4)
        return val_metrics

    def fit(self):
        """Runs complete training across epochs with early stopping and best model saving."""
        train_logger.info(f"Starting Stage {self.stage} training for {self.epochs} epochs on {self.device.upper()}...")
        print(f"\n--- Starting Training Stage {self.stage} ({self.profile.scenario}) ---")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_score"]
            val_loss = val_metrics["val_loss"]
            val_rec = val_metrics["recall"]

            curr_lr = self.scheduler.get_last_lr()[0]
            self.experiment_manager.record_epoch(
                epoch, train_loss, val_loss, val_acc, val_f1, val_rec, curr_lr
            )

            is_best = self.early_stopping.step(val_f1)
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                epoch=epoch,
                metrics=val_metrics,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                is_best=is_best
            )

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s"
                + (" (Best Checkpoint Saved)" if is_best else "")
            )

            if self.early_stopping.early_stop:
                print(f"Early stopping condition met. Halting training at epoch {epoch}.")
                break

        # Save history convergence plot
        self.experiment_manager.save_history_plots()
        train_logger.info("Training cycle completed.")

