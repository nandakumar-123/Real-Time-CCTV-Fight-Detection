"""
Unit tests for temporal window slicing and event aggregation.
Supports both pytest and python -m unittest.
"""

import unittest
from inference.temporal_windows import generate_temporal_windows
from inference.event_aggregator import EventAggregator

class TestInference(unittest.TestCase):
    def test_temporal_windows_generation(self):
        windows = generate_temporal_windows(video_duration=5.0, window_duration=2.0, stride=1.0)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0].start_sec, 0.0)
        self.assertEqual(windows[0].end_sec, 2.0)
        self.assertEqual(windows[-1].end_sec, 5.0)

    def test_event_aggregator_clustering(self):
        aggregator = EventAggregator(threshold=0.50, min_consecutive_windows=2)
        sample_windows = [
            {"start_sec": 0.0, "end_sec": 2.0, "fight_prob": 0.1},
            {"start_sec": 1.0, "end_sec": 3.0, "fight_prob": 0.85},
            {"start_sec": 2.0, "end_sec": 4.0, "fight_prob": 0.90},
            {"start_sec": 3.0, "end_sec": 5.0, "fight_prob": 0.20}
        ]
        events = aggregator.aggregate_events(sample_windows)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_id, 1)
        self.assertEqual(event.start_sec, 1.0)
        self.assertEqual(event.end_sec, 5.0)
        self.assertGreaterEqual(event.peak_fight_confidence, 0.60)

if __name__ == "__main__":
    unittest.main()
