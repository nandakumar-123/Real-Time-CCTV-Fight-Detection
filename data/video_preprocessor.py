"""
Video Preprocessing and Temporal Sampling Pipeline.
Handles lazy video decoding, 16-frame uniform temporal sampling,
VideoMAE V2 normalization, and motion-preserving spatial augmentations.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any
import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:
    torch = None

from config.settings import settings
from utils.logger import app_logger

# VideoMAE V2 / ImageNet Standard Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class VideoPreprocessor:
    """
    Decodes and transforms raw video files or specific temporal windows
    into standardized PyTorch tensors for VideoMAE V2 Base.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 224,
        is_training: bool = False
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_training = is_training

    def sample_frame_indices(
        self,
        total_frames: int,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[int]:
        """
        Calculates 16 uniformly distributed frame indices within [start_frame, end_frame].
        If training, applies slight temporal jitter; during eval, sampling is purely deterministic.
        """
        if end_frame is None or end_frame <= start_frame:
            end_frame = total_frames

        span = max(1, end_frame - start_frame)

        if span < self.num_frames:
            # If video segment has fewer frames than required, tile/repeat indices smoothly
            base = np.linspace(start_frame, end_frame - 1, num=span, dtype=int)
            indices = np.resize(base, self.num_frames).tolist()
            return sorted(indices)

        # Uniform segment sampling
        seg_size = span / float(self.num_frames)
        indices = []
        for i in range(self.num_frames):
            seg_start = start_frame + i * seg_size
            seg_end = start_frame + (i + 1) * seg_size
            if self.is_training:
                # Random frame within segment
                idx = int(random.uniform(seg_start, min(seg_end, end_frame - 1)))
            else:
                # Midpoint frame within segment (deterministic)
                idx = int((seg_start + seg_end) / 2.0)
            indices.append(min(int(idx), total_frames - 1))

        return sorted(indices)

    def decode_frames(
        self,
        video_path: Union[str, Path],
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None
    ) -> np.ndarray:
        """
        Lazy-reads only the required 16 frames from disk via OpenCV VideoCapture.
        Returns uint8 array of shape [num_frames, H, W, 3] in RGB format.
        """
        path_str = str(Path(video_path).resolve())
        cap = cv2.VideoCapture(path_str)

        if not cap.isOpened():
            raise IOError(f"Could not open video file: {path_str}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has 0 or invalid frame count: {path_str}")

        start_frame = int(start_sec * fps) if start_sec is not None else 0
        end_frame = int(end_sec * fps) if end_sec is not None else total_frames

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        target_indices = self.sample_frame_indices(total_frames, start_frame, end_frame)

        frames = []
        curr_idx = 0
        target_set = set(target_indices)
        captured_map = {}

        max_target = max(target_indices)
        while cap.isOpened() and curr_idx <= max_target:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(
                    rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
                )
                captured_map[curr_idx] = rgb_resized
            curr_idx += 1

        cap.release()

        # Build final 16 frames list
        last_good = None
        for idx in target_indices:
            frame = captured_map.get(idx)
            if frame is not None:
                last_good = frame
                frames.append(frame)
            elif last_good is not None:
                frames.append(last_good.copy())
            else:
                frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return np.stack(frames, axis=0) # [T, H, W, C]

    def apply_spatial_augmentations(self, frames: np.ndarray) -> np.ndarray:
        """
        Applies identical spatial transformations across all 16 frames in the clip
        to preserve temporal consistency and motion features.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Mild brightness & contrast jitter (consistent across frames)
        if random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15) # contrast
            beta = random.uniform(-15, 15)     # brightness
            frames = np.clip(frames.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return frames

    def process(
        self,
        video_path: Union[str, Path],
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None
    ) -> Any:
        """
        Full pipeline: Decode -> Sample -> Augment -> Normalize -> PyTorch Tensor.
        Returns:
            torch.Tensor: Shape [T, C, H, W] or [C, T, H, W] normalized float tensor.
        """
        frames = self.decode_frames(video_path, start_sec, end_sec)

        if self.is_training:
            frames = self.apply_spatial_augmentations(frames)

        # Convert to float and scale to [0, 1]
        frames_float = frames.astype(np.float32) / 255.0

        # Normalize with ImageNet mean and standard deviation
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
        normalized = (frames_float - mean) / std

        if torch is not None:
            # VideoMAE HF format expects shape [T, C, H, W] (Frames, Channels, Height, Width)
            # Input frames shape is [T, H, W, C]
            tensor = torch.from_numpy(normalized).permute(0, 3, 1, 2).contiguous().float()
            return tensor

        return normalized

