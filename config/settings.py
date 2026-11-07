"""
Application Settings and Configuration Management.
Loads environment variables safely with defaults and directory validation.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Try loading dotenv if installed, otherwise read from os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Settings:
    # Dataset Directories
    rwf2000_data_dir: Path = Path(os.getenv("RWF2000_DATA_DIR", "C:/Datasets/RWF-2000"))
    ntu_cctv_fights_data_dir: Path = Path(os.getenv("NTU_CCTV_FIGHTS_DATA_DIR", "C:/Datasets/NTU-CCTV-Fights"))

    # Model Configuration
    model_id: str = os.getenv("MODEL_ID", "OpenGVLab/VideoMAEv2-Base")
    num_frames: int = int(os.getenv("NUM_FRAMES", "16"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))

    # Project Directories
    project_root: Path = Path(os.getenv("PROJECT_ROOT", "./")).resolve()
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
    checkpoint_dir: Path = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints")).resolve()
    log_dir: Path = Path(os.getenv("LOG_DIR", "./logs")).resolve()
    report_dir: Path = Path(os.getenv("REPORT_DIR", "./reports")).resolve()
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()

    # Derived Paths
    splits_dir: Path = output_dir / "splits"
    predictions_dir: Path = output_dir / "predictions"
    events_dir: Path = output_dir / "events"

    # Training Parameters
    epochs: int = int(os.getenv("EPOCHS", "30"))
    target_accuracy: float = float(os.getenv("TARGET_ACCURACY", "0.85"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    # Execution & Hardware Settings
    device: str = os.getenv("DEVICE", "auto").lower()
    batch_size: str = os.getenv("BATCH_SIZE", "auto")
    num_workers: str = os.getenv("NUM_WORKERS", "auto")
    learning_rate: str = os.getenv("LEARNING_RATE", "auto")

    # Safety limits
    max_ram_percent: float = float(os.getenv("MAX_RAM_PERCENT", "85.0"))
    max_vram_percent: float = float(os.getenv("MAX_VRAM_PERCENT", "90.0"))
    max_cpu_threads: int = int(os.getenv("MAX_CPU_THREADS", "8"))

    def __post_init__(self):
        """Ensure all required output directories exist."""
        for d in [
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.report_dir,
            self.upload_dir,
            self.splits_dir,
            self.predictions_dir,
            self.events_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def validate_datasets(self) -> dict:
        """
        Check if configured dataset directories exist on disk.
        Returns a dict indicating presence and path status.
        """
        rwf_exists = self.rwf2000_data_dir.exists() and self.rwf2000_data_dir.is_dir()
        ntu_exists = self.ntu_cctv_fights_data_dir.exists() and self.ntu_cctv_fights_data_dir.is_dir()

        return {
            "RWF2000": {
                "configured_path": str(self.rwf2000_data_dir),
                "exists": rwf_exists
            },
            "NTU_CCTV_FIGHTS": {
                "configured_path": str(self.ntu_cctv_fights_data_dir),
                "exists": ntu_exists
            }
        }

# Global instance
settings = Settings()

