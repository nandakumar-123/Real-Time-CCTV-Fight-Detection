"""
Unit tests for alert generation, incident dispatching, and cooldown.
"""

from pathlib import Path
import pytest
from inference.alert_manager import AlertManager
from inference.event_aggregator import FightEvent

def test_alert_dispatch(tmp_path):
    mgr = AlertManager()
    mgr.events_dir = tmp_path

    events = [
        FightEvent(
            event_id=1,
            start_sec=1.5,
            end_sec=4.5,
            duration_sec=3.0,
            peak_fight_confidence=0.92,
            average_fight_confidence=0.88,
            num_positive_windows=3
        )
    ]

    packet = mgr.trigger_alerts("cctv_entrance_01.mp4", events, overall_fight_prob=0.92)
    assert packet["fight_detected"] is True
    assert packet["alert_status"] == "CRITICAL_ALERT"
    assert len(packet["events"]) == 1

    # Verify event file saved
    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1

