"""Class-balancing utilities for imbalanced datasets."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from src.data.dataset import FERDataset


def create_weighted_sampler(dataset: FERDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes.

    Args:
        dataset: A FERDataset instance with a `labels` attribute.

    Returns:
        WeightedRandomSampler configured for balanced sampling.
    """
    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)

    # Weight per class: inversely proportional to frequency
    class_weights = {cls: total / count for cls, count in label_counts.items()}

    # Per-sample weight
    sample_weights = [class_weights[label] for label in dataset.labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True,
    )


def get_class_weights(dataset: FERDataset) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Useful for passing to loss functions like CrossEntropyLoss(weight=...).

    Args:
        dataset: A FERDataset instance with a `labels` attribute.

    Returns:
        Float tensor of shape (num_classes,) with per-class weights.
    """
    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)
    num_classes = max(label_counts.keys()) + 1

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = label_counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)

    return weights
