"""
Unit tests for VideoMAE V2 architecture, binary head, and checkpoint management.
"""

import pytest
import torch
from models.classifier import CCTVFightClassifier
from models.checkpoint_manager import CheckpointManager

def test_model_instantiation_and_forward():
    model = CCTVFightClassifier()
    model.eval()

    # Create dummy video input batch [B=1, T=16, C=3, H=224, W=224]
    dummy_input = torch.randn(1, 16, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected shape [1, 2], got {logits.shape}"

    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (1, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-4)

def test_staged_freezing():
    model = CCTVFightClassifier()

    # Stage 1: Freeze backbone
    model.freeze_backbone()
    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    head_trainable = [p for p in model.classifier.parameters() if p.requires_grad]
    assert len(backbone_trainable) == 0
    assert len(head_trainable) > 0

    # Stage 3: Unfreeze all
    model.unfreeze_all()
    backbone_trainable_all = [p for p in model.backbone.parameters() if p.requires_grad]
    assert len(backbone_trainable_all) > 0

def test_checkpoint_roundtrip(tmp_path):
    model = CCTVFightClassifier()
    mgr = CheckpointManager(checkpoint_dir=tmp_path)

    saved_path = mgr.save_checkpoint(
        model=model,
        epoch=1,
        metrics={"accuracy": 0.88, "f1_score": 0.87},
        is_best=True
    )
    assert saved_path.exists()

    new_model = CCTVFightClassifier()
    new_model, meta = mgr.load_checkpoint(new_model, saved_path)
    assert meta["metrics"]["accuracy"] == 0.88

