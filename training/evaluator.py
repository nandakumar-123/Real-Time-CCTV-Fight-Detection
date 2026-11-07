"""
Independent Model Evaluator and Error Analysis Engine.
Calculates unbiased test performance, generates confusion matrix and ROC curves,
saves full classification reports, and isolates CCTV-only sub-evaluations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from config.settings import settings
from config.labels import ID_TO_LABEL, LABEL_FIGHT, LABEL_NON_FIGHT
from models.classifier import CCTVFightClassifier
from models.checkpoint_manager import CheckpointManager
from data.dataset_loader import FightDetectionDataset, create_dataloader
from utils.logger import app_logger
from utils.metrics import calculate_classification_metrics

class Evaluator:
    """Evaluates fine-tuned model against test sets and held-out CCTV recordings."""

    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_dataloader(self, dataloader) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs complete evaluation over provided dataloader.
        Returns:
            metrics (dict): Comprehensive evaluation metrics
            predictions_df (DataFrame): Sample-level predictions for error analysis
        """
        all_true = []
        all_pred = []
        all_prob = []
        records = []

        with torch.no_grad():
            for batch_idx, (tensors, labels, metas) in enumerate(dataloader):
                tensors = tensors.to(self.device)
                logits = self.model(tensors)
                probs = F.softmax(logits, dim=-1) # [B, 2]

                batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
                batch_probs_fight = probs[:, LABEL_FIGHT].cpu().numpy()
                batch_labels = labels.cpu().numpy()

                all_true.extend(batch_labels.tolist())
                all_pred.extend(batch_preds.tolist())
                all_prob.extend(batch_probs_fight.tolist())

                # Collect per-sample records for error analysis
                for i in range(len(batch_labels)):
                    records.append({
                        "video_path": metas["video_path"][i],
                        "source_id": metas["source_id"][i],
                        "start_sec": float(metas["start_sec"][i]),
                        "end_sec": float(metas["end_sec"][i]),
                        "true_label": int(batch_labels[i]),
                        "true_label_name": ID_TO_LABEL[int(batch_labels[i])],
                        "predicted_label": int(batch_preds[i]),
                        "predicted_label_name": ID_TO_LABEL[int(batch_preds[i])],
                        "fight_probability": float(batch_probs_fight[i]),
                        "is_correct": bool(batch_labels[i] == batch_preds[i])
                    })

        metrics = calculate_classification_metrics(all_true, all_pred, all_prob)
        predictions_df = pd.DataFrame(records)
        return metrics, predictions_df

def plot_confusion_matrix(cm: List[List[int]], output_path: Path):
    """Plots and saves confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    labels = ["Non-Fight", "Fight"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — CCTV Fight Detection")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_roc_curve(y_true: List[int], y_prob: List[float], output_path: Path):
    """Plots and saves Receiver Operating Characteristic (ROC) curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="ROC baseline")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — CCTV Fight Detection")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def run_evaluation(checkpoint_file: Optional[Path] = None, split_name: str = "test"):
    """
    Runs full evaluation on the designated split ('test' or 'validation').
    Generates all required report artifacts in reports/.
    """
    app_logger.info(f"Initiating evaluation on '{split_name}' split...")

    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    model = CCTVFightClassifier()
    ckpt_mgr = CheckpointManager()

    try:
        model, meta = ckpt_mgr.load_checkpoint(model, checkpoint_file, device=device)
        app_logger.info(f"Loaded checkpoint for evaluation: {meta.get('checkpoint_file')}")
    except Exception as e:
        app_logger.warning(f"Could not load trained checkpoint: {e}. Running with initialized weights.")

    split_csv = settings.splits_dir / f"{split_name}.csv"
    if not split_csv.exists():
        app_logger.error(f"Split CSV not found: {split_csv}. Please run 'python -m data.dataset_splitter' first.")
        print(f"Error: Split CSV not found at {split_csv}")
        return

    dataset = FightDetectionDataset(split_csv_path=split_csv, is_training=False)
    if len(dataset) == 0:
        print(f"Warning: The {split_name} split dataset is empty.")
        return

    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

    evaluator = Evaluator(model, device=device)
    metrics, pred_df = evaluator.evaluate_dataloader(dataloader)

    # Save reports
    rep_dir = settings.report_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final metrics JSON
    json_path = rep_dir / "final_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 2. Classification report TXT
    txt_path = rep_dir / "classification_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("CCTV AI FIGHT DETECTION — FINAL TEST CLASSIFICATION REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Accuracy:            {metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision:           {metrics['precision'] * 100:.2f}%\n")
        f.write(f"Recall (Fight):      {metrics['recall'] * 100:.2f}%\n")
        f.write(f"F1-Score:            {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:             {metrics['roc_auc']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['false_positive_rate'] * 100:.2f}%\n")
        f.write(f"False Negative Rate: {metrics['false_negative_rate'] * 100:.2f}%\n\n")
        f.write("Scikit-Learn Detailed Report:\n")
        f.write(metrics["classification_report"])

    # 3. Confusion Matrix plot
    cm_path = rep_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    # 4. ROC curve plot
    roc_path = rep_dir / "roc_curve.png"
    plot_roc_curve(pred_df["true_label"].tolist(), pred_df["fight_probability"].tolist(), roc_path)

    # 5. Error analysis CSV (false positives and false negatives)
    errors_df = pred_df[~pred_df["is_correct"]].copy()
    error_csv = rep_dir / "error_analysis.csv"
    errors_df.to_csv(error_csv, index=False)

    # 6. CCTV-Only subset evaluation
    if "is_cctv" in pred_df.columns or "video_path" in pred_df.columns:
        cctv_mask = pred_df["video_path"].str.contains("cctv|surveillance|rwf", case=False, na=False)
        cctv_sub = pred_df[cctv_mask]
        if len(cctv_sub) > 0:
            cctv_metrics = calculate_classification_metrics(
                cctv_sub["true_label"].tolist(),
                cctv_sub["predicted_label"].tolist(),
                cctv_sub["fight_probability"].tolist()
            )
            cctv_json = rep_dir / "cctv_only_evaluation.json"
            with open(cctv_json, "w", encoding="utf-8") as f:
                json.dump(cctv_metrics, f, indent=4)

    app_logger.info(f"Evaluation complete. Test Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']}")
    print(f"Evaluation Completed:")
    print(f"  - Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"  - Precision:  {metrics['precision'] * 100:.2f}%")
    print(f"  - Recall:     {metrics['recall'] * 100:.2f}%")
    print(f"  - F1-Score:   {metrics['f1_score']}")
    print(f"  - ROC-AUC:    {metrics['roc_auc']}")
    print(f"Reports saved in: {rep_dir}")

if __name__ == "__main__":
    run_evaluation()

