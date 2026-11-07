"""
Unit tests for temporal window slicing and event aggregation.
"""

import pytest
from inference.temporal_windows import generate_temporal_windows
from inference.event_aggregator import EventAggregator

def test_temporal_windows_generation():
    windows = generate_temporal_windows(video_duration=5.0, window_duration=2.0, stride=1.0)
    assert len(windows) == 4
    assert windows[0].start_sec == 0.0
    assert windows[0].end_sec == 2.0
    assert windows[-1].end_sec == 5.0

def test_event_aggregator_clustering():
    aggregator = EventAggregator(threshold=0.50, min_consecutive_windows=2)
    sample_windows = [
        {"start_sec": 0.0, "end_sec": 2.0, "fight_prob": 0.1},
        {"start_sec": 1.0, "end_sec": 3.0, "fight_prob": 0.85},
        {"start_sec": 2.0, "end_sec": 4.0, "fight_prob": 0.90},
        {"start_sec": 3.0, "end_sec": 5.0, "fight_prob": 0.20}
    ]
    events = aggregator.aggregate_events(sample_windows)
    assert len(events) == 1
    event = events[0]
    assert event.event_id == 1
    assert event.start_sec == 1.0
    assert event.end_sec == 4.0
    assert event.peak_fight_confidence >= 0.85

