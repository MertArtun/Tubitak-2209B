"""Custom loss functions for emotion classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import FOCAL_LOSS_GAMMA


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Ref: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Operates on raw logits (applies log-softmax internally).

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Optional per-class weight tensor of shape (num_classes,).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = FOCAL_LOSS_GAMMA,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw predictions of shape (N, C).
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Gather the probabilities for the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_weighted_ce_loss(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """Create a weighted CrossEntropyLoss.

    Args:
        class_weights: Per-class weight tensor of shape (num_classes,).

    Returns:
        Configured CrossEntropyLoss instance.
    """
    return nn.CrossEntropyLoss(weight=class_weights.float())
