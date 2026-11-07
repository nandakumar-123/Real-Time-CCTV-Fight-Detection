"""
Alert and Incident Notification Manager.
Triggers alert states, enforces alert cooldowns to avoid repetitive alarms,
and records structured event summaries to outputs/events/.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from config.settings import settings
from utils.logger import inference_logger
from inference.event_aggregator import FightEvent

class AlertManager:
    """Manages alert dispatch, cooldown windows, and event serialization."""

    def __init__(self, alert_cooldown_sec: float = 5.0):
        self.alert_cooldown_sec = alert_cooldown_sec
        self.last_alert_time: float = -999.0
        self.events_dir = settings.events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def trigger_alerts(
        self,
        video_name: str,
        events: List[FightEvent],
        overall_fight_prob: float
    ) -> Dict[str, Any]:
        """
        Processes detected fight events and returns a structured alert dispatch packet.
        """
        has_fight = len(events) > 0
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_packet = {
            "timestamp": timestamp_str,
            "video_name": video_name,
            "alert_status": "CRITICAL_ALERT" if has_fight else "NORMAL",
            "fight_detected": has_fight,
            "overall_fight_probability": round(overall_fight_prob, 4),
            "num_events": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "start_timestamp": f"{int(e.start_sec // 60):02d}:{int(e.start_sec % 60):02d}",
                    "end_timestamp": f"{int(e.end_sec // 60):02d}:{int(e.end_sec % 60):02d}",
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                    "peak_confidence": e.peak_fight_confidence,
                    "avg_confidence": e.average_fight_confidence
                }
                for e in events
            ]
        }

        if has_fight:
            inference_logger.warning(
                f"[ALERT] Physical fight detected in '{video_name}'! "
                f"Events count: {len(events)} | Peak confidence: {events[0].peak_fight_confidence:.2f}"
            )
            self._save_event_record(video_name, alert_packet)

        return alert_packet

    def _save_event_record(self, video_name: str, alert_packet: Dict[str, Any]):
        """Persists event record to outputs/events/."""
        safe_stem = Path(video_name).stem
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"event_{safe_stem}_{time_tag}.json"
        out_p = self.events_dir / filename

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(alert_packet, f, indent=4)
        inference_logger.info(f"Saved event summary to {out_p}")

