"""
Unit tests for video decoding, uniform 16-frame sampling, and synthetic video creation.
"""

from pathlib import Path
import pytest
import numpy as np
from utils.video_utils import create_synthetic_video, get_video_metadata, validate_video_path
from data.video_preprocessor import VideoPreprocessor

def test_synthetic_video_and_preprocessing(tmp_path):
    video_p = tmp_path / "test_clip.mp4"
    create_synthetic_video(video_p, duration_sec=2.0, fps=25, width=160, height=120)

    is_valid, msg = validate_video_path(video_p)
    assert is_valid, msg

    meta = get_video_metadata(video_p)
    assert meta["frame_count"] == 50
    assert meta["duration_seconds"] == 2.0

    preprocessor = VideoPreprocessor(num_frames=16, image_size=224, is_training=False)
    frames = preprocessor.decode_frames(video_p)
    assert frames.shape == (16, 224, 224, 3)

    tensor = preprocessor.process(video_p)
    # Check shape [16, 3, 224, 224]
    assert tensor.shape == (16, 3, 224, 224)

