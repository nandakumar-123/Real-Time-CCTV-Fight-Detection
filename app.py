"""
Streamlit Web Application: CCTV AI Fight Detection System.
Provides 6-page interactive dashboard for video upload, hardware monitoring,
temporal fight analysis, held-out evaluation, and system diagnostics.
"""

import os
import json
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

from config.settings import settings
from utils.hardware_detector import detect_hardware
from utils.resource_manager import resource_manager
from utils.video_utils import get_video_metadata, validate_video_path
from models.classifier import CCTVFightClassifier
from models.checkpoint_manager import CheckpointManager
from inference.video_inference import VideoInferenceEngine
from dashboard.components import render_alert_banner, render_metric_cards
from dashboard.metrics_view import render_evaluation_dashboard
from dashboard.results_view import render_timeline_chart, render_events_table
from diagnostics.report_generator import DiagnosticReportGenerator

# App Page Configuration
st.set_page_config(
    page_title="CCTV AI Fight Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_cached_model(checkpoint_path: str = None, device_type: str = "cpu"):
    """Caches loaded VideoMAE V2 model in memory to prevent reloading across queries."""
    model = CCTVFightClassifier()
    ckpt_mgr = CheckpointManager()
    target_p = Path(checkpoint_path) if checkpoint_path else None
    try:
        model, meta = ckpt_mgr.load_checkpoint(model, target_p, device=device_type)
        return model, meta
    except Exception as e:
        # Fallback to initialized model
        return model, {"checkpoint_file": "None (Using Initialized Weights)", "metrics": {}}

# Sidebar Navigation
st.sidebar.title("🛡️ CCTV Fight AI")
st.sidebar.caption("VideoMAE V2 Base Surveillance System")

pages = [
    "1. Dashboard",
    "2. Upload & Analyze Video",
    "3. Detection Results",
    "4. Temporal Timeline",
    "5. Model Evaluation",
    "6. System Logs & Diagnostics"
]
selected_page = st.sidebar.radio("Navigation", pages)

# Sidebar System Health Snapshot
profile = detect_hardware()
res_snap = resource_manager.get_resource_snapshot()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{profile.device_type.upper()}` ({profile.gpu_name or profile.cpu_model[:18]})")
st.sidebar.markdown(f"**RAM Usage:** `{res_snap['ram_used_gb']} / {res_snap['ram_total_gb']} GB` ({res_snap['ram_percent']}%)")
if profile.device_type == "cuda":
    st.sidebar.markdown(f"**VRAM Allocated:** `{res_snap['gpu_allocated_gb']} / {profile.gpu_total_vram_gb} GB`")

# Available Checkpoint Selection
ckpt_mgr = CheckpointManager()
available_ckpts = ckpt_mgr.get_available_checkpoints()
ckpt_options = list(available_ckpts.keys()) or ["None"]
selected_ckpt_name = st.sidebar.selectbox("Active Checkpoint", ckpt_options)
selected_ckpt_path = str(available_ckpts[selected_ckpt_name]) if selected_ckpt_name != "None" else None

# Load Model
model, model_meta = load_cached_model(selected_ckpt_path, device_type=profile.device_type)

# Initialize Session State
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "current_video_path" not in st.session_state:
    st.session_state["current_video_path"] = None

# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if selected_page == "1. Dashboard":
    st.title("🛡️ CCTV AI Fight Detection Dashboard")
    st.markdown("Automated surveillance pipeline utilizing **VideoMAE V2 Base** for physical violence detection.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System & Hardware Status")
        st.write(f"- **Operating System:** {profile.os_name} ({profile.python_version})")
        st.write(f"- **Execution Device:** `{profile.device_type.upper()}`")
        st.write(f"- **CPU:** {profile.cpu_model} ({profile.cpu_physical_cores} Cores)")
        st.write(f"- **GPU Model:** {profile.gpu_name or 'None (CPU Mode)'}")
        st.write(f"- **CUDA Version:** {profile.cuda_version or 'N/A'}")
        st.write(f"- **Hardware Scenario:** {profile.scenario}")

    with col2:
        st.subheader("Model Status")
        st.write(f"- **Architecture:** VideoMAE V2 Base")
        st.write(f"- **Pretrained Backbone:** `{settings.model_id}`")
        st.write(f"- **Active Checkpoint:** `{model_meta.get('checkpoint_file', 'Default')}`")
        st.write(f"- **Input Shape:** 16 Frames @ 224×224 RGB")
        st.write(f"- **Target Accuracy:** {settings.target_accuracy * 100:.0f}%")

    st.markdown("---")
    st.subheader("Operational Statistics")
    c1, c2, c3, c4 = st.columns(4)
    pred_files = list(settings.predictions_dir.glob("*.json"))
    event_files = list(settings.events_dir.glob("*.json"))
    c1.metric("Analyzed Videos", len(pred_files))
    c2.metric("Recorded Fight Incidents", len(event_files))
    c3.metric("RAM Available", f"{profile.available_ram_gb} GB")
    c4.metric("GPU Memory", f"{profile.gpu_total_vram_gb} GB" if profile.device_type == "cuda" else "CPU")

# ==============================================================================
# PAGE 2: UPLOAD & ANALYZE VIDEO
# ==============================================================================
elif selected_page == "2. Upload & Analyze Video":
    st.title("📹 Upload Video for Analysis")
    st.markdown("Upload CCTV surveillance footage (`.mp4`, `.avi`, `.mov`, `.mkv`).")

    uploaded_file = st.file_uploader("Select Video File", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        save_path = settings.upload_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["current_video_path"] = str(save_path)

        # Video preview & metadata
        col_v, col_m = st.columns([1.5, 1])
        with col_v:
            st.video(str(save_path))
        with col_m:
            st.subheader("Video Metadata")
            try:
                v_meta = get_video_metadata(save_path)
                st.write(f"- **Filename:** {v_meta['file_name']}")
                st.write(f"- **Resolution:** {v_meta['width']} × {v_meta['height']}")
                st.write(f"- **Duration:** {v_meta['duration_seconds']}s")
                st.write(f"- **Framerate:** {v_meta['fps']} FPS")
                st.write(f"- **File Size:** {v_meta['file_size_mb']} MB")
            except Exception as e:
                st.error(f"Error reading video metadata: {e}")

        st.markdown("---")
        # Analysis Controls
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            threshold = st.slider("Fight Confidence Threshold", min_value=0.20, max_value=0.90, value=0.50, step=0.05)
        with col_c2:
            stride = st.slider("Sliding Window Stride (seconds)", min_value=0.5, max_value=2.0, value=1.0, step=0.25)

        if st.button("🚀 ANALYSE VIDEO", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_hook(fraction: float, message: str):
                progress_bar.progress(fraction)
                status_text.text(message)

            try:
                engine = VideoInferenceEngine(
                    model=model,
                    device=profile.device_type,
                    window_duration=2.0,
                    stride=stride,
                    threshold=threshold
                )
                result = engine.process_video(save_path, progress_callback=progress_hook)
                st.session_state["last_result"] = result
                st.success("Analysis Completed Successfully! Switch to 'Detection Results' page to view findings.")
            except Exception as err:
                st.error(f"Inference Failure: {err}")

# ==============================================================================
# PAGE 3: DETECTION RESULTS
# ==============================================================================
elif selected_page == "3. Detection Results":
    st.title("📊 Fight Detection Results")
    res = st.session_state["last_result"]

    if res is None:
        st.info("No video has been analyzed yet. Please go to 'Upload & Analyze Video' page.")
    else:
        render_alert_banner(res["fight_detected"], res["peak_fight_probability"])
        render_metric_cards(res)

        st.markdown("---")
        col_res1, col_res2 = st.columns([1.5, 1])

        with col_res1:
            st.subheader("Event Summary")
            render_events_table(res["events"])

        with col_res2:
            st.subheader("Export Results")
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(res, indent=4),
                file_name=f"detection_{Path(res['file_name']).stem}.json",
                mime="application/json"
            )
            df_windows = pd.DataFrame(res["window_predictions"])
            st.download_button(
                label="📥 Download Window Predictions (CSV)",
                data=df_windows.to_csv(index=False),
                file_name=f"windows_{Path(res['file_name']).stem}.csv",
                mime="text/csv"
            )

# ==============================================================================
# PAGE 4: TEMPORAL TIMELINE
# ==============================================================================
elif selected_page == "4. Temporal Timeline":
    st.title("⏱️ Temporal Detection Timeline")
    res = st.session_state["last_result"]

    if res is None:
        st.info("Analyze a video first to display temporal window progression.")
    else:
        st.markdown(f"**Analyzed Footage:** `{res['file_name']}` ({res['metadata']['duration_seconds']}s)")
        render_timeline_chart(res["window_predictions"], threshold=0.50)

        st.markdown("---")
        st.subheader("Per-Window Predictions Breakdown")
        st.dataframe(pd.DataFrame(res["window_predictions"]), use_container_width=True)

# ==============================================================================
# PAGE 5: MODEL EVALUATION
# ==============================================================================
elif selected_page == "5. Model Evaluation":
    st.title("📈 Model Evaluation & Validation")
    render_evaluation_dashboard()

# ==============================================================================
# PAGE 6: SYSTEM LOGS & DIAGNOSTICS
# ==============================================================================
elif selected_page == "6. System Logs & Diagnostics":
    st.title("📋 System Logs & Diagnostics")

    log_files = {
        "Application": settings.log_dir / "application.log",
        "Training": settings.log_dir / "training.log",
        "Inference": settings.log_dir / "inference.log",
        "Errors": settings.log_dir / "errors.log",
        "Hardware": settings.log_dir / "hardware.log"
    }

    col_l1, col_l2 = st.columns([1, 3])
    with col_l1:
        selected_log = st.selectbox("Select Log Stream", list(log_files.keys()))
        level_filter = st.selectbox("Filter Level", ["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"])
        if st.button("Run Diagnostic Analyzer"):
            gen = DiagnosticReportGenerator()
            report_paths = gen.generate_reports()
            st.success(f"Report generated: {report_paths['markdown_report']}")

    with col_l2:
        log_p = log_files[selected_log]
        if log_p.exists():
            with open(log_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-300:]

            if level_filter != "ALL":
                lines = [l for l in lines if level_filter in l]

            st.text_area(f"Recent entries in {selected_log} ({len(lines)} lines)", "".join(lines), height=450)
        else:
            st.info(f"Log file '{selected_log}' is empty or does not exist yet.")

