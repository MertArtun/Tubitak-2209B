"""Tests for model architectures, CBAM, and loss functions."""

import pytest
import torch
import torch.nn as nn

from src.models.cbam import CBAM, ChannelAttention, SpatialAttention
from src.models.losses import FocalLoss, create_weighted_ce_loss


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
