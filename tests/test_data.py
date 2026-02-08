"""Tests for the data pipeline: config mappings, transforms, and sampling."""

import numpy as np
import pytest
import torch

from configs.config import CLASS_NAMES, CLASS_TO_IDX, FER_TO_3CLASS_IDX
from src.data.transforms import get_train_transforms, get_val_transforms


# ── Config mapping tests ──────────────────────────────────────────────────────


class TestFERMapping:
    """Verify FER2013 7-class to 3-class mapping."""

    def test_fer_3class_mapping(self):
        """angry->negative(0), happy->positive(2), neutral->neutral(1), surprise->None."""
        assert FER_TO_3CLASS_IDX[0] == CLASS_TO_IDX["negative"]  # angry
        assert FER_TO_3CLASS_IDX[1] == CLASS_TO_IDX["negative"]  # disgust
        assert FER_TO_3CLASS_IDX[2] == CLASS_TO_IDX["negative"]  # fear
        assert FER_TO_3CLASS_IDX[3] == CLASS_TO_IDX["positive"]  # happy
        assert FER_TO_3CLASS_IDX[4] == CLASS_TO_IDX["negative"]  # sad
        assert FER_TO_3CLASS_IDX[5] is None                       # surprise excluded
        assert FER_TO_3CLASS_IDX[6] == CLASS_TO_IDX["neutral"]   # neutral

    def test_class_names_order(self):
        """CLASS_NAMES should be ['negative', 'neutral', 'positive']."""
        assert CLASS_NAMES == ["negative", "neutral", "positive"]

    def test_class_to_idx_consistency(self):
        """CLASS_TO_IDX should be the inverse of CLASS_NAMES ordering."""
        for idx, name in enumerate(CLASS_NAMES):
            assert CLASS_TO_IDX[name] == idx


# ── Transform tests ──────────────────────────────────────────────────────────


class TestTransforms:
    """Verify train and val transform pipelines."""

    def test_train_transforms_output_shape(self, sample_image):
        """Train transforms produce a (3, 224, 224) tensor."""
        transform = get_train_transforms()
        result = transform(image=sample_image)["image"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_val_transforms_output_shape(self, sample_image):
        """Val transforms produce a (3, 224, 224) tensor."""
        transform = get_val_transforms()
        result = transform(image=sample_image)["image"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_transforms_normalize_range(self, sample_image):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        transform = get_val_transforms()
        result = transform(image=sample_image)["image"]
        assert result.min() >= -4.0
        assert result.max() <= 4.0

    def test_train_transforms_dtype(self, sample_image):
        """Output tensor should be float32."""
        transform = get_train_transforms()
        result = transform(image=sample_image)["image"]
        assert result.dtype == torch.float32

    def test_val_transforms_deterministic(self, sample_image):
        """Val transforms should produce identical output for same input."""
        transform = get_val_transforms()
        result1 = transform(image=sample_image.copy())["image"]
        result2 = transform(image=sample_image.copy())["image"]
        assert torch.allclose(result1, result2)


# ── Sampler tests ────────────────────────────────────────────────────────────


class TestSampler:
    """Verify weighted sampler and class weight computation."""

    def _make_mock_dataset(self, labels):
        """Create a minimal object with a .labels attribute."""

        class _FakeDataset:
            pass

        ds = _FakeDataset()
        ds.labels = labels
        return ds

    def test_weighted_sampler_creation(self):
        """Sampler should be created from a dataset with known class distribution."""
        from src.data.sampler import create_weighted_sampler

        # 10 negative, 5 neutral, 5 positive -> negative is majority
        labels = [0] * 10 + [1] * 5 + [2] * 5
        ds = self._make_mock_dataset(labels)

        sampler = create_weighted_sampler(ds)
        assert sampler is not None
        assert len(list(sampler)) == len(labels)

    def test_class_weights(self):
        """get_class_weights returns inverse proportional weights."""
        from src.data.sampler import get_class_weights

        # 10 negative, 5 neutral, 5 positive
        labels = [0] * 10 + [1] * 5 + [2] * 5
        ds = self._make_mock_dataset(labels)

        weights = get_class_weights(ds)
        assert weights.shape == (3,)
        # Minority classes (1, 2) should have higher weight than majority (0)
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]

    def test_class_weights_dtype(self):
        """Weights should be float32."""
        from src.data.sampler import get_class_weights

        labels = [0] * 6 + [1] * 3 + [2] * 3
        ds = self._make_mock_dataset(labels)

        weights = get_class_weights(ds)
        assert weights.dtype == torch.float32
