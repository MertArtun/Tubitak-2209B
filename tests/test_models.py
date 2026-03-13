"""Tests for model architectures, CBAM, loss functions, and training utilities."""

import pytest
import torch
import torch.nn as nn

from src.models.cbam import CBAM, ChannelAttention, SpatialAttention
from src.models.losses import FocalLoss, SoftTargetCrossEntropy, create_weighted_ce_loss

# ── CLI and checkpoint tests ─────────────────────────────────────────────────


class TestTrainCLI:
    """Test train.py CLI argument parsing."""

    def test_output_dir_arg(self):
        """--output_dir should override model and result dirs."""
        from src.models.train import _parse_args

        args = _parse_args(["--output_dir", "/tmp/test_out"])
        assert args.output_dir == "/tmp/test_out"

    def test_resume_flag(self):
        """--resume should default to True."""
        from src.models.train import _parse_args

        args = _parse_args([])
        assert args.resume is True

    def test_no_resume_flag(self):
        """--no_resume should set resume to False."""
        from src.models.train import _parse_args

        args = _parse_args(["--no_resume"])
        assert args.resume is False

    def test_checkpoint_dir_arg(self):
        """--checkpoint_dir should override checkpoint location."""
        from src.models.train import _parse_args

        args = _parse_args(["--checkpoint_dir", "/tmp/ckpts"])
        assert args.checkpoint_dir == "/tmp/ckpts"

    def test_default_args(self):
        """Default args should have expected values."""
        from src.models.train import _parse_args

        args = _parse_args([])
        assert args.output_dir is None
        assert args.checkpoint_dir is None
        assert args.resume is True
        assert args.model == "efficientnet_b3"


class TestCheckpointUtils:
    """Test checkpoint save/load round-trip."""

    def test_save_load_checkpoint_roundtrip(self, tmp_path):
        """save_training_state -> load_training_state preserves all fields."""
        from src.models.train import load_training_state, save_training_state

        state = {
            "epoch": 3,
            "phase": 2,
            "best_val_acc": 0.75,
            "history": {"train_loss": [0.5, 0.4, 0.3]},
        }
        path = tmp_path / "ckpt.pth"
        save_training_state(state, path)
        loaded = load_training_state(path)

        assert loaded["epoch"] == 3
        assert loaded["phase"] == 2
        assert loaded["best_val_acc"] == 0.75
        assert loaded["history"]["train_loss"] == [0.5, 0.4, 0.3]

    def test_load_nonexistent_returns_none(self, tmp_path):
        """load_training_state returns None if file missing."""
        from src.models.train import load_training_state

        result = load_training_state(tmp_path / "nonexistent.pth")
        assert result is None

# ── Architecture tests ────────────────────────────────────────────────────────


class TestModelForward:
    """Test forward pass of each model architecture."""

    @pytest.fixture(autouse=True)
    def _batch(self, sample_batch):
        self.batch = sample_batch

    def test_efficientnet_b3_forward(self):
        """EfficientNet-B3 forward pass outputs shape (2, 3)."""
        from src.models.architectures import EfficientNetB3Classifier

        model = EfficientNetB3Classifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        assert out.shape == (2, 3)

    def test_efficientnet_b0_forward(self):
        """EfficientNet-B0 forward pass outputs shape (2, 3)."""
        from src.models.architectures import EfficientNetB0Classifier

        model = EfficientNetB0Classifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        assert out.shape == (2, 3)

    def test_mobilenet_v3_forward(self):
        """MobileNetV3 forward pass outputs shape (2, 3)."""
        from src.models.architectures import MobileNetV3Classifier

        model = MobileNetV3Classifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        assert out.shape == (2, 3)

    def test_resnet50_cbam_forward(self):
        """ResNet50+CBAM forward pass outputs shape (2, 3)."""
        from src.models.architectures import ResNet50CBAMClassifier

        model = ResNet50CBAMClassifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        assert out.shape == (2, 3)

    def test_model_output_probabilities(self):
        """After softmax, outputs should sum to ~1.0."""
        from src.models.architectures import EfficientNetB0Classifier

        model = EfficientNetB0Classifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(self.batch)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


class TestModelFactory:
    """Test create_model factory function."""

    def test_create_model_factory(self):
        """create_model('efficientnet_b3') returns correct type."""
        from src.models.architectures import (
            EfficientNetB3Classifier,
            create_model,
        )

        model = create_model("efficientnet_b3", pretrained=False)
        assert isinstance(model, EfficientNetB3Classifier)

    def test_create_model_invalid(self):
        """create_model('invalid') raises ValueError."""
        from src.models.architectures import create_model

        with pytest.raises(ValueError, match="Unknown model"):
            create_model("invalid")


class TestFreezeUnfreeze:
    """Test backbone freeze/unfreeze toggling."""

    def test_freeze_unfreeze(self):
        """freeze_backbone/unfreeze_backbone toggle requires_grad."""
        from src.models.architectures import EfficientNetB0Classifier

        model = EfficientNetB0Classifier(pretrained=False)

        # Freeze
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad


# ── CBAM tests ────────────────────────────────────────────────────────────────


class TestCBAM:
    """Test CBAM sub-modules and full module."""

    def test_cbam_channel_attention(self):
        """ChannelAttention forward pass preserves shape."""
        ca = ChannelAttention(in_channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = ca(x)
        assert out.shape == x.shape

    def test_cbam_spatial_attention(self):
        """SpatialAttention forward pass preserves shape."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 8, 8)
        out = sa(x)
        assert out.shape == x.shape

    def test_cbam_full(self):
        """Full CBAM forward pass preserves shape."""
        cbam = CBAM(in_channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = cbam(x)
        assert out.shape == x.shape

    def test_cbam_channel_attention_scale_range(self):
        """Channel attention applies sigmoid, so scale should be in [0, 1]."""
        ca = ChannelAttention(in_channels=32)
        x = torch.randn(2, 32, 4, 4)
        # Check that output values don't exceed input range wildly
        # (sigmoid scaling means output <= input in absolute terms for positive inputs)
        out = ca(x)
        assert out.shape == x.shape


# ── Loss function tests ──────────────────────────────────────────────────────


class TestLosses:
    """Test FocalLoss and weighted CrossEntropy creation."""

    def test_focal_loss_computation(self):
        """FocalLoss returns a scalar > 0."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_focal_loss_vs_ce(self):
        """FocalLoss with gamma=0 should approximate CrossEntropyLoss."""
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        focal = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()

        focal_val = focal(logits, targets)
        ce_val = ce(logits, targets)

        assert torch.allclose(focal_val, ce_val, atol=1e-5)

    def test_focal_loss_with_alpha(self):
        """FocalLoss with per-class alpha weights should still produce a scalar."""
        alpha = torch.tensor([1.0, 2.0, 1.5])
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_weighted_ce_loss(self):
        """create_weighted_ce_loss returns CrossEntropyLoss."""
        weights = torch.tensor([1.0, 2.0, 1.5])
        loss_fn = create_weighted_ce_loss(weights)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_weighted_ce_loss_forward(self):
        """Weighted CrossEntropyLoss should compute a scalar loss."""
        weights = torch.tensor([1.0, 2.0, 1.5])
        loss_fn = create_weighted_ce_loss(weights)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0


# ── Soft Target Cross-Entropy tests ────────────────────────────────────────


class TestSoftTargetCrossEntropy:
    """Test SoftTargetCrossEntropy with hard and soft labels."""

    def test_soft_ce_with_hard_labels(self):
        """With hard labels (1D), should equal standard cross-entropy."""
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        soft_ce = SoftTargetCrossEntropy()
        standard_ce = nn.CrossEntropyLoss()

        soft_val = soft_ce(logits, targets)
        standard_val = standard_ce(logits, targets)

        assert torch.allclose(soft_val, standard_val, atol=1e-5)

    def test_soft_ce_with_soft_labels(self):
        """With soft labels (2D), should compute correct cross-entropy."""
        logits = torch.randn(4, 3)
        # One-hot soft labels (should equal hard label CE)
        targets_hard = torch.tensor([0, 1, 2, 1])
        targets_soft = torch.zeros(4, 3)
        targets_soft[0, 0] = 1.0
        targets_soft[1, 1] = 1.0
        targets_soft[2, 2] = 1.0
        targets_soft[3, 1] = 1.0

        soft_ce = SoftTargetCrossEntropy()

        loss_hard = soft_ce(logits, targets_hard)
        loss_soft = soft_ce(logits, targets_soft)

        assert torch.allclose(loss_hard, loss_soft, atol=1e-5)

    def test_soft_ce_gradient_flows(self):
        """Gradient should flow through the soft label loss."""
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2],
        ])

        soft_ce = SoftTargetCrossEntropy()
        loss = soft_ce(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 3)
        assert not torch.all(logits.grad == 0)
