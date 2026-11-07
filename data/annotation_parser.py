"""
Annotation Parser for NTU CCTV-Fights Dataset.
Parses temporal fight boundaries (start_time, end_time, start_frame, end_frame)
to create accurate positive (Fight) and negative (Non-Fight) temporal segments.
Preserves source video identifiers to ensure source-independent data splitting.
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from config.labels import LABEL_FIGHT, LABEL_NON_FIGHT
from utils.logger import app_logger

@dataclass
class VideoInterval:
    video_id: str
    video_path: str
    source_type: str # 'cctv' or 'other'
    start_sec: float
    end_sec: float
    label: int # 0: Non-Fight, 1: Fight
    label_name: str # "Non-Fight" or "Fight"

class NTUAnnotationParser:
    """
    Parses NTU CCTV-Fights temporal annotation files.
    Standard annotation formats in NTU CCTV-Fights include:
    - Text files with lines: video_name, start_frame, end_frame, [fps] or start_sec, end_sec
    - CSV files with headers: VideoID, Start, End
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.annotation_dir = self.dataset_dir / "annotations"
        if not self.annotation_dir.exists():
            # If not in subfolder, search root
            self.annotation_dir = self.dataset_dir

    def parse_annotations(self, video_durations: Dict[str, float]) -> List[VideoInterval]:
        """
        Parses all annotation files found in the dataset directory.
        Pairs annotated fight spans with video durations to extract:
        1. Positive fight intervals [start_fight, end_fight]
        2. Negative non-fight intervals [0, start_fight] and [end_fight, duration]
        """
        intervals: List[VideoInterval] = []
        if not self.dataset_dir.exists():
            app_logger.warning(f"NTU dataset directory does not exist: {self.dataset_dir}")
            return intervals

        # Search for .txt or .csv annotation files
        annot_files = list(self.annotation_dir.glob("*.txt")) + list(self.annotation_dir.glob("*.csv"))

        fight_spans_by_video: Dict[str, List[Tuple[float, float]]] = {}

        for af in annot_files:
            try:
                with open(af, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("Video"):
                            continue
                        
                        parts = re.split(r'[\s,;]+', line)
                        if len(parts) >= 3:
                            vid_name = parts[0]
                            try:
                                # Start and End could be seconds or frames
                                val1 = float(parts[1])
                                val2 = float(parts[2])
                                start_sec = min(val1, val2)
                                end_sec = max(val1, val2)

                                if vid_name not in fight_spans_by_video:
                                    fight_spans_by_video[vid_name] = []
                                fight_spans_by_video[vid_name].append((start_sec, end_sec))
                            except ValueError:
                                continue
            except Exception as e:
                app_logger.warning(f"Error reading annotation file {af}: {e}")

        # Now construct positive and negative intervals for each video
        for vid_name, duration in video_durations.items():
            vid_stem = Path(vid_name).stem
            # Identify if CCTV based on naming conventions (CCTV videos often prefixed or numbered in 0-280)
            is_cctv = "cctv" in vid_name.lower() or "surveillance" in vid_name.lower()
            source_type = "cctv" if is_cctv else "other"

            spans = fight_spans_by_video.get(vid_name, fight_spans_by_video.get(vid_stem, []))

            if not spans:
                # Video has no fight instances -> Full video is Non-Fight
                intervals.append(VideoInterval(
                    video_id=vid_stem,
                    video_path=vid_name,
                    source_type=source_type,
                    start_sec=0.0,
                    end_sec=duration,
                    label=LABEL_NON_FIGHT,
                    label_name="Non-Fight"
                ))
            else:
                # Merge overlapping fight intervals
                spans = sorted(spans, key=lambda x: x[0])
                merged: List[Tuple[float, float]] = []
                for s, e in spans:
                    if not merged:
                        merged.append((s, e))
                    else:
                        prev_s, prev_e = merged[-1]
                        if s <= prev_e:
                            merged[-1] = (prev_s, max(prev_e, e))
                        else:
                            merged.append((s, e))

                # Create positive intervals
                last_end = 0.0
                for s, e in merged:
                    s_bounded = max(0.0, min(s, duration))
                    e_bounded = max(s_bounded, min(e, duration))

                    # Non-fight before the fight
                    if s_bounded - last_end >= 2.0: # At least 2 seconds for a valid negative interval
                        intervals.append(VideoInterval(
                            video_id=vid_stem,
                            video_path=vid_name,
                            source_type=source_type,
                            start_sec=last_end,
                            end_sec=s_bounded,
                            label=LABEL_NON_FIGHT,
                            label_name="Non-Fight"
                        ))

                    # Fight instance
                    if e_bounded - s_bounded >= 1.0:
                        intervals.append(VideoInterval(
                            video_id=vid_stem,
                            video_path=vid_name,
                            source_type=source_type,
                            start_sec=s_bounded,
                            end_sec=e_bounded,
                            label=LABEL_FIGHT,
                            label_name="Fight"
                        ))

                    last_end = e_bounded

                # Trailing non-fight interval after the last fight
                if duration - last_end >= 2.0:
                    intervals.append(VideoInterval(
                        video_id=vid_stem,
                        video_path=vid_name,
                        source_type=source_type,
                        start_sec=last_end,
                        end_sec=duration,
                        label=LABEL_NON_FIGHT,
                        label_name="Non-Fight"
                    ))

        return intervals

