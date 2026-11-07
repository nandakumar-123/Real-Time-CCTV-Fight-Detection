"""
Metrics & Evaluation UI Renderer for Streamlit.
Displays confusion matrix, ROC curves, loss progression, and validation metrics.
"""

from pathlib import Path
import json
import streamlit as st
from config.settings import settings

def render_evaluation_dashboard():
    """Renders held-out test evaluation reports and plots."""
    st.subheader("Model Evaluation on Held-Out Test Set")
    st.caption("All metrics shown below are calculated strictly against unseen test recordings.")

    rep_dir = settings.report_dir
    json_path = rep_dir / "final_metrics.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
            col2.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
            col3.metric("Fight Recall", f"{metrics.get('recall', 0.0) * 100:.1f}%")
            col4.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
            col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")
        except Exception as e:
            st.warning(f"Could not parse metrics JSON: {e}")
    else:
        st.info("No evaluation metrics generated yet. Run 'python -m training.evaluator' to populate.")

    st.markdown("---")

    # Image plots
    col_a, col_b = st.columns(2)
    cm_path = rep_dir / "confusion_matrix.png"
    roc_path = rep_dir / "roc_curve.png"

    with col_a:
        st.markdown("##### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.caption("Run evaluation to generate Confusion Matrix.")

    with col_b:
        st.markdown("##### ROC Curve")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.caption("Run evaluation to generate ROC Curve.")

    st.markdown("##### Training History & Convergence")
    hist_path = rep_dir / "training_history.png"
    if hist_path.exists():
        st.image(str(hist_path), use_container_width=True)
    else:
        st.caption("Training history curves will display once training is conducted.")

