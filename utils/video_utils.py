"""
Video Utility Module.
Handles video decoding, metadata extraction, validation, and synthetic video creation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from utils.logger import app_logger

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
Union_Path = Union[Path, str]

def validate_video_path(video_path: Union_Path) -> Tuple[bool, str]:
    """Validates if file exists, is non-empty, and has a supported video extension."""
    p = Path(video_path)
    if not p.exists():
        return False, f"Video file does not exist: {p}"
    if not p.is_file():
        return False, f"Path is not a regular file: {p}"
    if p.stat().st_size == 0:
        return False, f"Video file is empty (0 bytes): {p}"
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported video extension: {p.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
    return True, "Valid video file."

def get_video_metadata(video_path: Union_Path) -> Dict[str, Any]:
    """
    Extracts video metadata using OpenCV.
    Returns dictionary with width, height, fps, total_frames, duration_seconds.
    """
    p = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(p)

    if not cap.isOpened():
        raise IOError(f"Could not open video file: {p}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "file_name": Path(p).name,
        "file_path": p,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "file_size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)
    }

def read_video_frames_cv2(
    video_path: Union_Path,
    target_frame_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Decodes specific frame indices from a video, or all frames if target_frame_indices is None.
    Returns RGB frame numpy arrays [H, W, 3].
    """
    p = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(p)

    if not cap.isOpened():
        raise IOError(f"Failed to open video file for reading: {p}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_frame_indices is not None:
        sorted_indices = sorted(set(target_frame_indices))
        idx_map = {idx: None for idx in sorted_indices}

        curr_frame_idx = 0
        while cap.isOpened() and curr_frame_idx <= max(sorted_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame_idx in idx_map:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx_map[curr_frame_idx] = rgb_frame
            curr_frame_idx += 1

        cap.release()

        # Reconstruct list maintaining requested order and handling missing indices via fallback
        last_valid = None
        for idx in target_frame_indices:
            f = idx_map.get(idx)
            if f is not None:
                last_valid = f
                frames.append(f)
            elif last_valid is not None:
                frames.append(last_valid.copy())
            else:
                # Fallback blank frame
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return frames

def create_synthetic_video(
    output_path: Union_Path,
    duration_sec: float = 3.0,
    fps: int = 25,
    width: int = 320,
    height: int = 240,
    action_type: str = "normal"
) -> Path:
    """
    Creates a synthetic video clip for testing and pipeline validation.
    Action type 'fight' creates high-motion colored blobs; 'normal' creates slow translation.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = [i % 50, (i * 2) % 60, (i * 3) % 70]

        if action_type == "fight":
            # Erratic rapid movements simulating dynamic interaction
            cx1 = int(width / 2 + np.sin(i * 1.5) * (width * 0.3))
            cy1 = int(height / 2 + np.cos(i * 1.8) * (height * 0.3))
            cx2 = int(width / 2 - np.sin(i * 1.7) * (width * 0.3))
            cy2 = int(height / 2 - np.cos(i * 1.4) * (height * 0.3))

            cv2.circle(frame, (cx1, cy1), 20, (0, 0, 255), -1) # Red
            cv2.circle(frame, (cx2, cy2), 20, (255, 0, 0), -1) # Blue
            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 3)
        else:
            # Steady slow movement
            cx = int(width * 0.2 + (i / total_frames) * (width * 0.6))
            cy = int(height * 0.5)
            cv2.circle(frame, (cx, cy), 25, (0, 255, 0), -1) # Green

        out.write(frame)

    out.release()
    return out_p

