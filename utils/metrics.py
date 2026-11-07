"""
Evaluation Metrics Utility.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, FPR, FNR,
and produces structured confusion matrix representations.
"""

from typing import Dict, Any, List, Union
import numpy as np

def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    y_true: Ground truth binary labels (0 or 1)
    y_pred: Predicted binary labels (0 or 1)
    y_prob: Predicted probability for class 1 (Fight)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Core scores
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Error rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC AUC (requires at least one positive and one negative sample)
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    report_str = classification_report(
        y_true, y_pred, target_names=["Non-Fight", "Fight"], zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "fight_recall": round(rec, 4), # explicit fight recall
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str
    }

