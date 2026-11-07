"""
Reusable UI Components for Streamlit CCTV Fight Detection.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

def render_alert_banner(fight_detected: bool, peak_conf: float):
    """Renders prominent status banner for fight detection state."""
    if fight_detected:
        st.error(
            f"### 🚨 < FIGHT DETECTED >\n"
            f"**Physical altercation detected.** Peak confidence: **{peak_conf * 100:.1f}%**",
            icon="🚨"
        )
    else:
        st.success(
            "### ✅ < NO FIGHT DETECTED >\n"
            "**Normal surveillance activity. No violent altercations identified.**",
            icon="✅"
        )

def render_metric_cards(metrics: Dict[str, Any]):
    """Renders 4-column metric summary cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fight Probability", f"{metrics.get('peak_fight_probability', 0.0) * 100:.1f}%")
    with col2:
        st.metric("Non-Fight Probability", f"{(1.0 - metrics.get('peak_fight_probability', 0.0)) * 100:.1f}%")
    with col3:
        st.metric("Detected Fight Events", metrics.get("num_fight_events", 0))
    with col4:
        st.metric("Processing Time", f"{metrics.get('processing_time_sec', 0.0):.2f}s")

