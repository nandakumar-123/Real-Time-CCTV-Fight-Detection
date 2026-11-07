"""
Inference Results and Temporal Timeline Visualizer for Streamlit.
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_timeline_chart(window_predictions: list, threshold: float = 0.50):
    """Renders confidence progression over video timestamps."""
    if not window_predictions:
        st.info("No window predictions available.")
        return

    df = pd.DataFrame(window_predictions)
    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(df["start_sec"], df["fight_prob"], color="#e74c3c", lw=2.5, label="Fight Probability")
    ax.axhline(y=threshold, color="#34495e", linestyle="--", lw=1.5, label=f"Detection Threshold ({threshold})")

    # Fill above threshold
    ax.fill_between(df["start_sec"], df["fight_prob"], threshold, where=(df["fight_prob"] >= threshold),
                    color="#e74c3c", alpha=0.25, label="Fight Region")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Temporal Fight Probability Profile")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    st.pyplot(fig)
    plt.close()

def render_events_table(events: list):
    """Renders table of detected physical fight events."""
    if not events:
        st.success("No fight intervals detected in this recording.")
        return

    st.markdown("##### Detected Fight Intervals")
    formatted = []
    for e in events:
        formatted.append({
            "Event ID": e["event_id"],
            "Start Time": f"{int(e['start_sec'] // 60):02d}:{int(e['start_sec'] % 60):02d} ({e['start_sec']}s)",
            "End Time": f"{int(e['end_sec'] // 60):02d}:{int(e['end_sec'] % 60):02d} ({e['end_sec']}s)",
            "Duration (s)": e["duration_sec"],
            "Peak Confidence": f"{e['peak_confidence'] * 100:.1f}%",
            "Avg Confidence": f"{e['avg_confidence'] * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(formatted), use_container_width=True)

