"""
Unit tests for video decoding, uniform 16-frame sampling, and synthetic video creation.
Supports both pytest and python -m unittest.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from utils.video_utils import create_synthetic_video, get_video_metadata, validate_video_path
from data.video_preprocessor import VideoPreprocessor

class TestVideoProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_synthetic_video_and_preprocessing(self):
        video_p = self.test_dir / "test_clip.mp4"
        create_synthetic_video(video_p, duration_sec=2.0, fps=25, width=160, height=120)

        is_valid, msg = validate_video_path(video_p)
        self.assertTrue(is_valid, msg)

        meta = get_video_metadata(video_p)
        self.assertEqual(meta["frame_count"], 50)
        self.assertEqual(meta["duration_seconds"], 2.0)

        preprocessor = VideoPreprocessor(num_frames=16, image_size=224, is_training=False)
        frames = preprocessor.decode_frames(video_p)
        self.assertEqual(frames.shape, (16, 224, 224, 3))

        tensor = preprocessor.process(video_p)
        self.assertEqual(tensor.shape, (16, 3, 224, 224))

if __name__ == "__main__":
    unittest.main()
