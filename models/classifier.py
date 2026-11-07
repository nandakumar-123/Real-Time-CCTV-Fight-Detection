"""
CCTV Fight Detection Classification Model.
Attaches custom binary classification head onto VideoMAE V2 Base backbone,
supports staged freezing/unfreezing, and provides probability calculations.
"""

from typing import Tuple, Dict, Any, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from config.labels import NUM_CLASSES
from models.videomae_loader import load_videomae_backbone
from utils.logger import app_logger

class CCTVFightClassifier(nn.Module):
    """
    Two-class Fight / Non-Fight Video Classifier based on VideoMAE V2.
    Inputs: [B, T, C, H, W] Video Tensor (16 frames, 3 channels, 224x224)
    Outputs: [B, 2] Raw logits
    """

    def __init__(self, model_id: str = "OpenGVLab/VideoMAEv2-Base", dropout_prob: float = 0.5):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch is required to initialize CCTVFightClassifier.")

        self.backbone, self.hidden_dim = load_videomae_backbone(model_id)

        # Classification Head
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim, NUM_CLASSES)

        # Initialize head weights with small standard deviation
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.
        Args:
            x: [B, T, C, H, W] video frames tensor
        Returns:
            logits: [B, 2] class logits
        """
        # VideoMAE model in Hugging Face expects [batch_size, num_frames, num_channels, height, width]
        outputs = self.backbone(x)

        # Extract pooled feature representation
        if hasattr(outputs, "last_hidden_state"):
            # Global average pool over sequence/spatiotemporal tokens
            features = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs[0].mean(dim=1)

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def predict_probabilities(self, x: "torch.Tensor") -> "torch.Tensor":
        """Computes softmax probabilities: shape [B, 2] (Class 0: Non-Fight, Class 1: Fight)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    # =========================================================================
    # Staged Fine-Tuning Control Methods (Stage 1, Stage 2, Stage 3)
    # =========================================================================

    def freeze_backbone(self):
        """Stage 1: Freezes all backbone parameters; only trains classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        app_logger.info("Stage 1 configured: Backbone fully frozen, classification head trainable.")

    def unfreeze_final_blocks(self, num_blocks: int = 2):
        """
        Stage 2: Unfreezes the last N transformer layers and layer norm.
        """
        # First ensure all backbone layers are frozen
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Attempt locating encoder layers
        encoder_layers = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layers"):
            encoder_layers = self.backbone.layers

        if encoder_layers is not None and len(encoder_layers) > 0:
            total_blocks = len(encoder_layers)
            start_unfreeze = max(0, total_blocks - num_blocks)
            for i in range(start_unfreeze, total_blocks):
                for p in encoder_layers[i].parameters():
                    p.requires_grad = True
            app_logger.info(f"Stage 2 configured: Unfroze final {num_blocks} blocks (layers {start_unfreeze} to {total_blocks - 1}).")
        else:
            app_logger.warning("Could not automatically locate encoder layers; unfreezing top 20% of backbone parameters.")
            params = list(self.backbone.parameters())
            for p in params[int(len(params) * 0.8):]:
                p.requires_grad = True

        # Head remains trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Stage 3: Full fine-tuning of all backbone and head parameters."""
        for param in self.parameters():
            param.requires_grad = True
        app_logger.info("Stage 3 configured: Full network unfrozen for end-to-end fine-tuning.")

    def get_parameter_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-4, weight_decay: float = 0.05) -> List[Dict[str, Any]]:
        """
        Constructs differential parameter groups with distinct learning rates
        for backbone vs. classification head.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "name": "backbone"
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head"
            })
        return groups

