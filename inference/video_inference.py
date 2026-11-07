"""
End-to-End Video Inference Engine.
Performs sliding-window feature analysis, VideoMAE V2 classification,
temporal fight event aggregation, and alert dispatching.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from config.settings import settings
from config.labels import ID_TO_LABEL, LABEL_FIGHT, LABEL_NON_FIGHT
from utils.logger import inference_logger
from utils.video_utils import validate_video_path, get_video_metadata
from data.video_preprocessor import VideoPreprocessor
from inference.temporal_windows import generate_temporal_windows
from inference.event_aggregator import EventAggregator, FightEvent
from inference.alert_manager import AlertManager

class VideoInferenceEngine:
    """Runs temporal window classification across complete uploaded video files."""

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        window_duration: float = 2.0,
        stride: float = 1.0,
        threshold: float = 0.50,
        min_consecutive_windows: int = 2
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.window_duration = window_duration
        self.stride = stride
        self.threshold = threshold

        self.preprocessor = VideoPreprocessor(
            num_frames=settings.num_frames,
            image_size=settings.image_size,
            is_training=False
        )
        self.aggregator = EventAggregator(
            threshold=threshold,
            min_consecutive_windows=min_consecutive_windows
        )
        self.alert_manager = AlertManager()

    def process_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on video.
        Returns complete analysis dictionary including window predictions and aggregated events.
        """
        t_start = time.time()
        v_path = Path(video_path)

        # 1. Validation
        is_valid, msg = validate_video_path(v_path)
        if not is_valid:
            raise ValueError(msg)

        # 2. Extract Metadata
        meta = get_video_metadata(v_path)
        duration = meta["duration_seconds"]

        if progress_callback:
            progress_callback(0.1, "Generating temporal analysis windows...")

        # 3. Generate Windows
        windows = generate_temporal_windows(
            video_duration=duration,
            window_duration=self.window_duration,
            stride=self.stride
        )

        window_results = []
        total_windows = len(windows)
        fight_probs = []

        # 4. Inference on each window
        for i, win in enumerate(windows):
            if progress_callback:
                frac = 0.1 + 0.8 * (i / max(1, total_windows))
                progress_callback(frac, f"Analyzing window {i+1} of {total_windows}...")

            tensor = self.preprocessor.process(
                video_path=v_path,
                start_sec=win.start_sec,
                end_sec=win.end_sec
            )

            # Ensure batch dimension [1, T, C, H, W]
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0) # [2]
                prob_non_fight = float(probs[LABEL_NON_FIGHT].item())
                prob_fight = float(probs[LABEL_FIGHT].item())

            pred_class = LABEL_FIGHT if prob_fight >= self.threshold else LABEL_NON_FIGHT
            fight_probs.append(prob_fight)

            window_results.append({
                "window_index": win.window_index,
                "start_sec": win.start_sec,
                "end_sec": win.end_sec,
                "start_timestamp": f"{int(win.start_sec // 60):02d}:{int(win.start_sec % 60):02d}",
                "end_timestamp": f"{int(win.end_sec // 60):02d}:{int(win.end_sec % 60):02d}",
                "predicted_label": ID_TO_LABEL[pred_class],
                "non_fight_prob": round(prob_non_fight, 4),
                "fight_prob": round(prob_fight, 4)
            })

        if progress_callback:
            progress_callback(0.95, "Aggregating temporal events & evaluating alerts...")

        # 5. Event Aggregation
        events = self.aggregator.aggregate_events(window_results)

        # Overall probabilities: weighted or max/mean of positive windows
        overall_fight_prob = float(np.mean(fight_probs)) if fight_probs else 0.0
        peak_fight_prob = float(np.max(fight_probs)) if fight_probs else 0.0

        is_fight = len(events) > 0
        overall_label = "Fight" if is_fight else "Non-Fight"

        # 6. Dispatch Alerts
        alert_info = self.alert_manager.trigger_alerts(
            video_name=v_path.name,
            events=events,
            overall_fight_prob=peak_fight_prob if is_fight else overall_fight_prob
        )

        elapsed = round(time.time() - t_start, 2)

        final_result = {
            "file_name": v_path.name,
            "metadata": meta,
            "overall_classification": overall_label,
            "fight_detected": is_fight,
            "peak_fight_probability": round(peak_fight_prob, 4),
            "mean_fight_probability": round(overall_fight_prob, 4),
            "processing_time_sec": elapsed,
            "num_windows": total_windows,
            "num_fight_events": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "start_sec": e.start_sec,
                    "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                    "peak_confidence": e.peak_fight_confidence,
                    "avg_confidence": e.average_fight_confidence,
                    "num_windows": e.num_positive_windows
                }
                for e in events
            ],
            "window_predictions": window_results,
            "alert": alert_info
        }

        # 7. Save prediction record to outputs/predictions/
        self._save_prediction_results(v_path.name, final_result)

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return final_result

    def _save_prediction_results(self, video_name: str, result: Dict[str, Any]):
        out_dir = settings.predictions_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(video_name).stem
        json_p = out_dir / f"pred_{safe_stem}.json"
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        
        # Save window breakdown as CSV
        csv_p = out_dir / f"windows_{safe_stem}.csv"
        pd.DataFrame(result["window_predictions"]).to_csv(csv_p, index=False)

