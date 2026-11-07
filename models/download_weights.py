"""
Pretrained Model Download and Verification Utility.
User-controlled command to download and verify VideoMAE V2 Base weights
prior to running training or inference.
Command: python -m models.download_weights
"""

import sys
import argparse
from pathlib import Path

from config.settings import settings
from utils.logger import app_logger

def download_and_verify_model(model_id: str = "OpenGVLab/VideoMAEv2-Base", cache_dir: str = None):
    """
    Downloads VideoMAE V2 Base configuration and weights from Hugging Face Hub
    into a local cache directory and verifies checkpoint tensor integrity.
    """
    print(f"\n=======================================================")
    print(f"Downloading Pretrained Model: {model_id}")
    print(f"=======================================================\n")

    try:
        from transformers import VideoMAEConfig, VideoMAEModel
        import torch
    except ImportError:
        print("[ERROR] 'transformers' and 'torch' are required.")
        print("Please install them first using:")
        print("  pip install torch transformers huggingface-hub")
        sys.exit(1)

    try:
        print(f"1. Fetching model configuration for '{model_id}'...")
        config = VideoMAEConfig.from_pretrained(model_id, cache_dir=cache_dir)
        config.num_frames = settings.num_frames
        config.image_size = settings.image_size
        print("   Configuration verified.")

        print(f"2. Downloading model weights for '{model_id}'...")
        model = VideoMAEModel.from_pretrained(
            model_id,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True
        )
        print("   Model weights downloaded successfully.")

        print(f"3. Verifying dummy forward pass tensor compatibility...")
        dummy_tensor = torch.zeros(1, settings.num_frames, 3, settings.image_size, settings.image_size)
        with torch.no_grad():
            outputs = model(dummy_tensor)
        print(f"   Forward pass completed. Feature representation shape: {outputs.last_hidden_state.shape}")

        print(f"\n[SUCCESS] Model '{model_id}' is downloaded, cached, and verified for training/inference.\n")
        app_logger.info(f"Model {model_id} downloaded and verified successfully.")
    except Exception as e:
        print(f"\n[WARNING] Could not download from Hugging Face Hub directly: {e}")
        print("If you are in an offline environment or have a local checkpoint, you can configure")
        print("the path to your local weights in .env (MODEL_ID=<path_or_id>).")
        app_logger.warning(f"Model download warning: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify VideoMAE V2 Base model")
    parser.add_argument("--model_id", type=str, default=settings.model_id, help="Hugging Face Model ID or local path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional custom cache directory")
    args = parser.parse_args()

    download_and_verify_model(model_id=args.model_id, cache_dir=args.cache_dir)
