"""
PyTorch Dataset and DataLoader Factory for CCTV Fight Detection.
Implements lazy video window loading, caching checks, and robust error resilience.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

from config.settings import settings
from data.video_preprocessor import VideoPreprocessor
from utils.logger import app_logger

class FightDetectionDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for video clips and temporal windows.
    Returns:
        tensor: [T, C, H, W] normalized video tensor (16 frames)
        label: torch.long (0 = Non-Fight, 1 = Fight)
        meta: dict containing video_path, source_id, start_sec, end_sec
    """

    def __init__(
        self,
        split_csv_path: Optional[Path] = None,
        dataframe: Optional[pd.DataFrame] = None,
        is_training: bool = False,
        num_frames: int = 16,
        image_size: int = 224
    ):
        if dataframe is not None:
            self.df = dataframe.copy().reset_index(drop=True)
        elif split_csv_path is not None and Path(split_csv_path).exists():
            self.df = pd.read_csv(split_csv_path)
        else:
            self.df = pd.DataFrame(columns=["video_path", "start_sec", "end_sec", "label", "source_id"])

        self.preprocessor = VideoPreprocessor(
            num_frames=num_frames,
            image_size=image_size,
            is_training=is_training
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", -1.0))
        label = int(row.get("label", 0))
        source_id = str(row.get("source_id", "unknown"))

        # If end_sec is -1.0, it denotes the whole video
        actual_end = end_sec if end_sec > 0 else None

        try:
            tensor = self.preprocessor.process(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=actual_end
            )
        except Exception as e:
            app_logger.warning(f"Error decoding {video_path} at index {idx}: {e}. Returning zeros fallback.")
            if torch is not None:
                tensor = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
            else:
                tensor = np.zeros((16, 224, 224, 3), dtype=np.float32)

        meta = {
            "video_path": video_path,
            "source_id": source_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label
        }

        if torch is not None and not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return tensor, label, meta

def create_dataloader(
    dataset: FightDetectionDataset,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Creates a configured PyTorch DataLoader."""
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is not installed. DataLoader cannot be created.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

