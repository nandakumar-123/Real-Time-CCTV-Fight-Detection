"""
Source-Aware Dataset Splitting Module.
Partitions dataset instances into 70% Train, 15% Validation, and 15% Test.
Guarantees that all frames, clips, or windows from any single physical source video
reside exclusively in one partition to prevent data leakage.
"""

import os
import sys
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.settings import settings
from config.labels import LABEL_FIGHT, LABEL_NON_FIGHT
from utils.logger import app_logger
from utils.video_utils import SUPPORTED_EXTENSIONS
from data.annotation_parser import NTUAnnotationParser

def collect_rwf2000_samples(rwf_dir: Path) -> List[Dict[str, Any]]:
    """Discovers all clips in RWF-2000 and assigns source video identifiers."""
    samples = []
    if not rwf_dir.exists():
        return samples

    for root, _, files in os.walk(rwf_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                path_lower = str(p).lower()
                is_fight = "fight" in path_lower and "nonfight" not in path_lower and "non_fight" not in path_lower
                label = LABEL_FIGHT if is_fight else LABEL_NON_FIGHT

                # In RWF-2000, clips usually have unique names or share an underscore prefix
                # e.g., video12_1.mp4, video12_2.mp4. We derive source_id from base name prefix.
                stem = p.stem
                source_id = stem.split("_")[0] if "_" in stem else stem

                samples.append({
                    "dataset": "RWF-2000",
                    "source_id": f"rwf_{source_id}",
                    "video_path": str(p.resolve()),
                    "start_sec": 0.0,
                    "end_sec": -1.0, # full clip
                    "label": label,
                    "label_name": "Fight" if label == LABEL_FIGHT else "Non-Fight",
                    "is_cctv": True
                })
    return samples

def collect_ntu_samples(ntu_dir: Path) -> List[Dict[str, Any]]:
    """Parses NTU CCTV-Fights using NTUAnnotationParser for temporal window intervals."""
    samples = []
    if not ntu_dir.exists():
        return samples

    # Gather video files and estimated durations
    video_files = []
    for root, _, files in os.walk(ntu_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                video_files.append(p)

    video_durations = {}
    import cv2
    for p in video_files:
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            video_durations[str(p.resolve())] = count / fps if fps > 0 else 5.0
        else:
            video_durations[str(p.resolve())] = 5.0

    parser = NTUAnnotationParser(ntu_dir)
    intervals = parser.parse_annotations(video_durations)

    for inv in intervals:
        samples.append({
            "dataset": "NTU-CCTV-Fights",
            "source_id": f"ntu_{inv.video_id}",
            "video_path": inv.video_path,
            "start_sec": inv.start_sec,
            "end_sec": inv.end_sec,
            "label": inv.label,
            "label_name": inv.label_name,
            "is_cctv": (inv.source_type == "cctv")
        })

    return samples

def create_splits(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs source-aware train/val/test splitting.
    Train: 70%, Val: 15%, Test: 15%.
    """
    random.seed(seed)
    all_samples = []
    all_samples.extend(collect_rwf2000_samples(settings.rwf2000_data_dir))
    all_samples.extend(collect_ntu_samples(settings.ntu_cctv_fights_data_dir))

    if not all_samples:
        app_logger.warning("No samples discovered in configured dataset folders. Creating empty split tables.")
        cols = ["dataset", "source_id", "video_path", "start_sec", "end_sec", "label", "label_name", "is_cctv"]
        empty_df = pd.DataFrame(columns=cols)
        return empty_df, empty_df, empty_df

    df = pd.DataFrame(all_samples)

    # First split: 70% Train, 30% Holdout (Val + Test) grouped by source_id
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, holdout_idx = next(gss1.split(df, groups=df["source_id"]))

    train_df = df.iloc[train_idx].copy()
    holdout_df = df.iloc[holdout_idx].copy()

    # Second split: Holdout into 50% Val, 50% Test (each 15% of total dataset)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss2.split(holdout_df, groups=holdout_df["source_id"]))

    val_df = holdout_df.iloc[val_rel_idx].copy()
    test_df = holdout_df.iloc[test_rel_idx].copy()

    # Ensure no leakage of source_id across splits
    train_sources = set(train_df["source_id"])
    val_sources = set(val_df["source_id"])
    test_sources = set(test_df["source_id"])

    assert len(train_sources.intersection(val_sources)) == 0, "Leakage detected between Train and Val!"
    assert len(train_sources.intersection(test_sources)) == 0, "Leakage detected between Train and Test!"
    assert len(val_sources.intersection(test_sources)) == 0, "Leakage detected between Val and Test!"

    return train_df, val_df, test_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Saves generated splits to outputs/splits/ as CSV files."""
    out_dir = settings.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_p = out_dir / "train.csv"
    val_p = out_dir / "validation.csv"
    test_p = out_dir / "test.csv"

    train_df.to_csv(train_p, index=False)
    val_df.to_csv(val_p, index=False)
    test_df.to_csv(test_p, index=False)

    app_logger.info(f"Saved source-aware dataset splits:")
    app_logger.info(f"  - Train:      {len(train_df)} samples -> {train_p}")
    app_logger.info(f"  - Validation: {len(val_df)} samples -> {val_p}")
    app_logger.info(f"  - Test:       {len(test_df)} samples -> {test_p}")

    print(f"Dataset splits generated successfully:")
    print(f"  - Train:      {len(train_df)} samples ({train_p})")
    print(f"  - Validation: {len(val_df)} samples ({val_p})")
    print(f"  - Test:       {len(test_df)} samples ({test_p})")

if __name__ == "__main__":
    t_df, v_df, te_df = create_splits(seed=settings.random_seed)
    save_splits(t_df, v_df, te_df)

