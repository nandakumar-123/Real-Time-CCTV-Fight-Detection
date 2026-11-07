"""
Temporal Event Aggregator and Smoothing Engine.
Smooths sliding-window prediction probabilities, groups consecutive positive detections,
and estimates precise physical fight start and end timestamps.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class FightEvent:
    event_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_fight_confidence: float
    average_fight_confidence: float
    num_positive_windows: int

class EventAggregator:
    """
    Aggregates window-level prediction streams into discrete fight incidents.
    Filters out spurious single-window glitches using consecutive window thresholds.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        min_consecutive_windows: int = 2,
        smoothing_window: int = 3
    ):
        self.threshold = threshold
        self.min_consecutive_windows = min_consecutive_windows
        self.smoothing_window = smoothing_window

    def smooth_probabilities(self, probs: List[float]) -> List[float]:
        """Applies a moving average filter to smooth probability fluctuations."""
        if len(probs) < self.smoothing_window:
            return probs

        smoothed = []
        half = self.smoothing_window // 2
        for i in range(len(probs)):
            start_i = max(0, i - half)
            end_i = min(len(probs), i + half + 1)
            window_slice = probs[start_i:end_i]
            smoothed.append(sum(window_slice) / len(window_slice))
        return smoothed

    def aggregate_events(self, window_predictions: List[Dict[str, Any]]) -> List[FightEvent]:
        """
        Parses list of window dicts:
        [{'start_sec': float, 'end_sec': float, 'fight_prob': float}, ...]
        Returns consolidated FightEvent objects.
        """
        if not window_predictions:
            return []

        raw_probs = [w["fight_prob"] for w in window_predictions]
        smoothed_probs = self.smooth_probabilities(raw_probs)

        events: List[FightEvent] = []
        active_cluster: List[Dict[str, Any]] = []
        event_counter = 1

        for i, w in enumerate(window_predictions):
            p = smoothed_probs[i]
            is_positive = p >= self.threshold

            if is_positive:
                active_cluster.append({
                    "start_sec": w["start_sec"],
                    "end_sec": w["end_sec"],
                    "prob": p
                })
            else:
                if len(active_cluster) >= self.min_consecutive_windows:
                    events.append(self._build_event(event_counter, active_cluster))
                    event_counter += 1
                active_cluster = []

        # Flush trailing cluster
        if len(active_cluster) >= self.min_consecutive_windows:
            events.append(self._build_event(event_counter, active_cluster))

        return events

    def _build_event(self, event_id: int, cluster: List[Dict[str, Any]]) -> FightEvent:
        start_sec = cluster[0]["start_sec"]
        end_sec = cluster[-1]["end_sec"]
        probs = [item["prob"] for item in cluster]

        return FightEvent(
            event_id=event_id,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            duration_sec=round(end_sec - start_sec, 2),
            peak_fight_confidence=round(max(probs), 4),
            average_fight_confidence=round(sum(probs) / len(probs), 4),
            num_positive_windows=len(cluster)
        )

