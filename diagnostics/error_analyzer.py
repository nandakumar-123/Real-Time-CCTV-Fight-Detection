"""
AI-Assisted and Rule-Based Error Diagnostic Analyzer.
Parses application, training, and inference log files to diagnose runtime faults,
CUDA out-of-memory errors, tensor mismatches, and dataset anomalies.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import settings

DIAGNOSTIC_RULES = [
    {
        "category": "CUDA Out Of Memory",
        "pattern": re.compile(r"(CUDA out of memory|torch\.cuda\.OutOfMemoryError)", re.IGNORECASE),
        "cause": "GPU VRAM capacity exceeded by batch size, video resolution, or unreleased tensor allocations.",
        "suggestion": "Reduce BATCH_SIZE to 1 in .env, increase gradient accumulation steps, or enable gradient checkpointing.",
        "patch_hint": "In .env: Set BATCH_SIZE=1 and rerun training."
    },
    {
        "category": "Missing Dataset Directory",
        "pattern": re.compile(r"(Dataset directory does not exist|No such file or directory.*Datasets)", re.IGNORECASE),
        "cause": "Configured dataset path in .env does not match physical storage location on Windows.",
        "suggestion": "Update RWF2000_DATA_DIR and NTU_CCTV_FIGHTS_DATA_DIR in .env to the correct absolute paths.",
        "patch_hint": "Edit .env file and set valid directory paths."
    },
    {
        "category": "Video Codec / Decode Failure",
        "pattern": re.compile(r"(Could not open video file|Failed to decode the first video frame|moov atom not found)", re.IGNORECASE),
        "cause": "Corrupted container header or missing codec for the video stream.",
        "suggestion": "Re-encode problematic videos with ffmpeg (h264/yuv420p) or skip corrupt files identified in reports/dataset_summary.csv.",
        "patch_hint": "ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4"
    },
    {
        "category": "Module / Dependency Missing",
        "pattern": re.compile(r"ModuleNotFoundError: No module named '([^']+)'", re.IGNORECASE),
        "cause": "Required Python package is not installed in the active environment.",
        "suggestion": "Install the missing package manually using pip.",
        "patch_hint": "pip install <package_name>"
    },
    {
        "category": "PyTorch CPU Multiprocessing Deadlock",
        "pattern": re.compile(r"(DataLoader worker.*killed|freeze|multiprocessing)", re.IGNORECASE),
        "cause": "Windows spawn multiprocessing issues when num_workers > 0 on CPU.",
        "suggestion": "Set NUM_WORKERS=0 in .env when running on Windows with CPU execution.",
        "patch_hint": "In .env: Set NUM_WORKERS=0"
    }
]

class ErrorAnalyzer:
    """Scans system log files to extract and categorize recent error occurrences."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir or settings.log_dir)

    def scan_recent_errors(self, max_lines: int = 500) -> List[Dict[str, Any]]:
        """Reads error and application logs, matching against diagnostic patterns."""
        errors_found = []
        log_files = [self.log_dir / "errors.log", self.log_dir / "application.log", self.log_dir / "training.log"]

        for log_f in log_files:
            if not log_f.exists():
                continue

            try:
                with open(log_f, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-max_lines:]

                for line in lines:
                    if "ERROR" in line or "CRITICAL" in line or "Exception" in line or "Traceback" in line:
                        diagnosis = self._diagnose_line(line)
                        errors_found.append({
                            "log_file": log_f.name,
                            "raw_message": line.strip(),
                            "category": diagnosis["category"],
                            "cause": diagnosis["cause"],
                            "suggestion": diagnosis["suggestion"],
                            "patch_hint": diagnosis["patch_hint"]
                        })
            except Exception:
                pass

        return errors_found

    def _diagnose_line(self, line: str) -> Dict[str, str]:
        for rule in DIAGNOSTIC_RULES:
            if rule["pattern"].search(line):
                return {
                    "category": rule["category"],
                    "cause": rule["cause"],
                    "suggestion": rule["suggestion"],
                    "patch_hint": rule["patch_hint"]
                }
        return {
            "category": "General Runtime Error",
            "cause": "Unclassified application or system exception occurred during execution.",
            "suggestion": "Inspect full traceback in logs/errors.log for detailed stack trace.",
            "patch_hint": "Check parameter configuration and file access permissions."
        }

