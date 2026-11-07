"""
VideoMAE V2 Base Pretrained Model Loader.
Loads official VideoMAE V2 Base transformer backbone (OpenGVLab/VideoMAEv2-Base)
or compatible VideoMAE architectures from Hugging Face transformers.
"""

from typing import Tuple, Any, Optional
from pathlib import Path

from config.settings import settings
from utils.logger import app_logger

def load_videomae_backbone(model_id: str = "OpenGVLab/VideoMAEv2-Base") -> Tuple[Any, int]:
    """
    Loads pretrained VideoMAE V2 Base architecture and extracts the hidden dimension.
    Returns:
        backbone: Pretrained PyTorch model / backbone module
        hidden_dim: Dimension of feature representations (typically 768 for Base)
    """
    try:
        import torch
        import torch.nn as nn
        from transformers import VideoMAEModel, VideoMAEConfig
    except ImportError:
        raise ImportError(
            "transformers and torch are required. Install via pip install torch transformers timm"
        )

    app_logger.info(f"Loading VideoMAE backbone: {model_id}")

    try:
        # First attempt: Official Hugging Face load
        config = VideoMAEConfig.from_pretrained(model_id)
        config.num_frames = settings.num_frames
        config.image_size = settings.image_size

        backbone = VideoMAEModel.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        hidden_dim = config.hidden_size
        app_logger.info(f"Loaded pretrained backbone '{model_id}' successfully. Hidden dimension: {hidden_dim}")
        return backbone, hidden_dim
    except Exception as e:
        app_logger.warning(
            f"Could not load '{model_id}' directly from HuggingFace ({e}). "
            f"Attempting standard VideoMAE-base fallback architecture..."
        )
        try:
            # Fallback to standard base configuration (768 hidden size, 12 layers, 12 heads)
            config = VideoMAEConfig(
                image_size=settings.image_size,
                patch_size=16,
                num_channels=3,
                num_frames=settings.num_frames,
                tubelet_size=2,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                qkv_bias=True,
                use_mean_pooling=True,
                decoder_num_attention_heads=6,
                decoder_hidden_size=384,
                decoder_num_hidden_layers=4,
                decoder_intermediate_size=1536
            )
            backbone = VideoMAEModel(config)
            hidden_dim = 768
            app_logger.info("Initialized standard VideoMAE Base architecture successfully.")
            return backbone, hidden_dim
        except Exception as e2:
            app_logger.error(f"Critical failure initializing VideoMAE backbone: {e2}")
            raise RuntimeError(f"Failed to initialize VideoMAE architecture: {e2}")

