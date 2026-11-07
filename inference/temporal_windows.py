"""
Temporal Sliding Window Generator.
Partitions video duration into uniform overlapping time intervals
for fine-grained temporal fight detection and localization.
"""

from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class WindowSpan:
    window_index: int
    start_sec: float
    end_sec: float
    duration_sec: float

def generate_temporal_windows(
    video_duration: float,
    window_duration: float = 2.0,
    stride: float = 1.0
) -> List[WindowSpan]:
    """
    Generates overlapping temporal window spans spanning the video duration.
    Args:
        video_duration: Total video length in seconds
        window_duration: Length of each window in seconds
        stride: Temporal stride between consecutive windows
    """
    windows = []
    if video_duration <= 0:
        return windows

    # If video is shorter than window duration, generate single window
    if video_duration <= window_duration:
        windows.append(WindowSpan(
            window_index=0,
            start_sec=0.0,
            end_sec=video_duration,
            duration_sec=video_duration
        ))
        return windows

    curr_start = 0.0
    idx = 0
    while curr_start < video_duration:
        curr_end = min(curr_start + window_duration, video_duration)
        windows.append(WindowSpan(
            window_index=idx,
            start_sec=round(curr_start, 2),
            end_sec=round(curr_end, 2),
            duration_sec=round(curr_end - curr_start, 2)
        ))
        idx += 1
        curr_start += stride

        if curr_end >= video_duration:
            break

    return windows

