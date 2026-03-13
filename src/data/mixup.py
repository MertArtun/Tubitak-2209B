"""Batch-level MixUp and CutMix augmentation for training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MixUpCutMix:
    """Apply MixUp or CutMix augmentation at the batch level.

    Randomly selects MixUp or CutMix (50/50) and applies with given probability.
    Always returns soft labels (N x C, float).

    Args:
        num_classes: Number of target classes.
        mixup_alpha: Beta distribution alpha for MixUp.
        cutmix_alpha: Beta distribution alpha for CutMix.
        prob: Probability of applying any augmentation.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp or CutMix to a batch.

        Args:
            images: Batch of images (N, C, H, W).
            targets: Hard labels (N,) with class indices.

        Returns:
            Tuple of (augmented_images, soft_labels) where soft_labels is (N, num_classes).
        """
        # Convert hard labels to one-hot soft labels
        soft_targets = F.one_hot(targets, self.num_classes).float()

        if torch.rand(1).item() > self.prob:
            return images, soft_targets

        # Randomly choose MixUp or CutMix
        if torch.rand(1).item() < 0.5:
            return self._mixup(images, soft_targets)
        return self._cutmix(images, soft_targets)

    def _mixup(
        self, images: torch.Tensor, soft_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp: linear interpolation of two samples."""
        lam = _sample_lambda(self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * soft_targets + (1 - lam) * soft_targets[index]
        return mixed_images, mixed_targets

    def _cutmix(
        self, images: torch.Tensor, soft_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix: paste a random patch from another sample."""
        lam = _sample_lambda(self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        _, _, h, w = images.shape
        bbx1, bby1, bbx2, bby2 = _rand_bbox(h, w, lam)

        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to the actual patch area ratio
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (w * h)
        mixed_targets = actual_lam * soft_targets + (1 - actual_lam) * soft_targets[index]
        return mixed_images, mixed_targets


def _sample_lambda(alpha: float) -> float:
    """Sample lambda from Beta(alpha, alpha) distribution."""
    if alpha <= 0:
        return 1.0
    dist = torch.distributions.Beta(alpha, alpha)
    return dist.sample().item()


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    """Generate random bounding box for CutMix.

    Returns:
        (x1, y1, x2, y2) coordinates of the patch.
    """
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()

    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(w, cx + cut_w // 2)
    bby2 = min(h, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2
