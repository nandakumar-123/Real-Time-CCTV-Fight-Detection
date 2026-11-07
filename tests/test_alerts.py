"""
Unit tests for alert generation, incident dispatching, and cooldown.
Supports both pytest and python -m unittest.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from inference.alert_manager import AlertManager
from inference.event_aggregator import FightEvent

class TestAlerts(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_alert_dispatch(self):
        mgr = AlertManager()
        mgr.events_dir = self.test_dir

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
        self.assertTrue(packet["fight_detected"])
        self.assertEqual(packet["alert_status"], "CRITICAL_ALERT")
        self.assertEqual(len(packet["events"]), 1)

        # Verify event file saved
        saved_files = list(self.test_dir.glob("*.json"))
        self.assertEqual(len(saved_files), 1)

if __name__ == "__main__":
    unittest.main()
