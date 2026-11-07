"""
Dataset Inspection and Integrity Verification Module.
Analyzes RWF-2000 and NTU CCTV-Fights datasets, auditing file formats,
video resolutions, durations, corruptions, duplicate hashes, and label distributions.
Generates comprehensive reports in reports/ (JSON, CSV, and TXT).
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2

from config.settings import settings
from config.labels import LABEL_FIGHT, LABEL_NON_FIGHT
from utils.logger import app_logger
from utils.video_utils import SUPPORTED_EXTENSIONS

def calculate_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """Calculates SHA-256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def inspect_video_file(path: Path) -> Dict[str, Any]:
    """Inspects a single video file for integrity, duration, fps, and decode readiness."""
    file_size_bytes = path.stat().st_size
    info = {
        "file_name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": file_size_bytes,
        "is_corrupt": False,
        "error_message": "",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_sec": 0.0
    }

    if file_size_bytes == 0:
        info["is_corrupt"] = True
        info["error_message"] = "Empty file (0 bytes)"
        return info

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info["is_corrupt"] = True
        info["error_message"] = "Cannot open with OpenCV VideoCapture"
        return info

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Test reading the first frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        info["is_corrupt"] = True
        info["error_message"] = "Failed to decode the first video frame"
        return info

    duration = frame_count / fps if fps > 0 else 0.0

    info.update({
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2)
    })
    return info

def inspect_dataset_directory(dataset_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Walks dataset directory, discovering all videos, labels, and anomalies."""
    report = {
        "dataset_name": dataset_name,
        "directory": str(dataset_dir.resolve()),
        "exists": dataset_dir.exists(),
        "total_files_scanned": 0,
        "valid_videos_count": 0,
        "corrupt_videos_count": 0,
        "skipped_files_count": 0,
        "fight_count": 0,
        "non_fight_count": 0,
        "duplicate_hash_count": 0,
        "corrupt_details": [],
        "skipped_details": [],
        "video_records": []
    }

    if not dataset_dir.exists():
        app_logger.warning(f"Dataset directory does not exist: {dataset_dir}")
        return report

    seen_hashes = {}

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            report["total_files_scanned"] += 1
            file_p = Path(root) / f
            suffix = file_p.suffix.lower()

            if suffix not in SUPPORTED_EXTENSIONS:
                report["skipped_files_count"] += 1
                report["skipped_details"].append({
                    "file": str(file_p),
                    "reason": f"Unsupported file extension: {suffix}"
                })
                continue

            # Determine label based on parent directory or filename conventions
            # (RWF-2000 structure: train/Fight, train/NonFight, val/Fight, val/NonFight)
            path_str_lower = str(file_p).lower()
            if "nonfight" in path_str_lower or "non_fight" in path_str_lower or "normal" in path_str_lower:
                label = LABEL_NON_FIGHT
                label_name = "Non-Fight"
            elif "fight" in path_str_lower:
                label = LABEL_FIGHT
                label_name = "Fight"
            else:
                label = LABEL_NON_FIGHT # fallback default
                label_name = "Non-Fight (Unspecified)"

            # Check video integrity
            v_info = inspect_video_file(file_p)
            v_info["label"] = label
            v_info["label_name"] = label_name

            if v_info["is_corrupt"]:
                report["corrupt_videos_count"] += 1
                report["corrupt_details"].append({
                    "file": str(file_p),
                    "reason": v_info["error_message"]
                })
            else:
                # Check for duplicates via hash
                try:
                    f_hash = calculate_file_hash(file_p)
                    if f_hash in seen_hashes:
                        report["duplicate_hash_count"] += 1
                        v_info["duplicate_of"] = seen_hashes[f_hash]
                    else:
                        seen_hashes[f_hash] = str(file_p)
                except Exception:
                    pass

                report["valid_videos_count"] += 1
                if label == LABEL_FIGHT:
                    report["fight_count"] += 1
                else:
                    report["non_fight_count"] += 1

                report["video_records"].append(v_info)

    return report

def run_inspection():
    """Runs end-to-end dataset inspection for both RWF-2000 and NTU CCTV-Fights."""
    app_logger.info("Starting Dataset Inspection...")

    rwf_dir = settings.rwf2000_data_dir
    ntu_dir = settings.ntu_cctv_fights_data_dir

    rwf_report = inspect_dataset_directory(rwf_dir, "RWF-2000")
    ntu_report = inspect_dataset_directory(ntu_dir, "NTU-CCTV-Fights")

    summary = {
        "RWF2000": {k: v for k, v in rwf_report.items() if k != "video_records"},
        "NTU_CCTV_Fights": {k: v for k, v in ntu_report.items() if k != "video_records"},
        "total_valid_videos": rwf_report["valid_videos_count"] + ntu_report["valid_videos_count"],
        "total_corrupt_videos": rwf_report["corrupt_videos_count"] + ntu_report["corrupt_videos_count"]
    }

    # Save summary JSON
    rep_dir = settings.report_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    json_path = rep_dir / "dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Save summary CSV
    csv_path = rep_dir / "dataset_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Total Scanned", "Valid Videos", "Corrupt Videos", "Fight Count", "Non-Fight Count", "Duplicates"])
        for name, r in [("RWF-2000", rwf_report), ("NTU-CCTV-Fights", ntu_report)]:
            writer.writerow([
                name,
                r["total_files_scanned"],
                r["valid_videos_count"],
                r["corrupt_videos_count"],
                r["fight_count"],
                r["non_fight_count"],
                r["duplicate_hash_count"]
            ])

    # Save human-readable inspection TXT
    txt_path = rep_dir / "dataset_inspection.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("CCTV AI FIGHT DETECTION — DATASET INSPECTION AUDIT\n")
        f.write("================================================================================\n\n")
        for name, r in [("RWF-2000", rwf_report), ("NTU-CCTV-Fights", ntu_report)]:
            f.write(f"Dataset: {name}\n")
            f.write(f"  Directory:           {r['directory']}\n")
            f.write(f"  Directory Exists:    {r['exists']}\n")
            f.write(f"  Total Files Scanned: {r['total_files_scanned']}\n")
            f.write(f"  Valid Videos:        {r['valid_videos_count']}\n")
            f.write(f"  Corrupt Videos:      {r['corrupt_videos_count']}\n")
            f.write(f"  Skipped Files:       {r['skipped_files_count']}\n")
            f.write(f"  Fight Labels:        {r['fight_count']}\n")
            f.write(f"  Non-Fight Labels:    {r['non_fight_count']}\n")
            f.write(f"  Duplicate Hashes:    {r['duplicate_hash_count']}\n\n")

            if r["corrupt_details"]:
                f.write(f"  Corrupted Files in {name}:\n")
                for c in r["corrupt_details"]:
                    f.write(f"    - {c['file']}: {c['reason']}\n")
                f.write("\n")

    app_logger.info(f"Dataset inspection completed. Summary saved to {json_path}")
    print(f"Inspection completed successfully.")
    print(f"  - Summary JSON: {json_path}")
    print(f"  - Summary CSV:  {csv_path}")
    print(f"  - Inspection Details: {txt_path}")

if __name__ == "__main__":
    run_inspection()

