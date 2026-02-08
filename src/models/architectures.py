"""Model architectures for 3-class emotion classification.

All models use timm pretrained backbones and share a common interface:
    - forward(x) -> logits
    - get_features(x) -> feature vector (before classifier head)
    - freeze_backbone() / unfreeze_backbone()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import timm
import torch
import torch.nn as nn

from configs.config import MODEL_CONFIGS, NUM_CLASSES
from src.models.cbam import CBAM


class EmotionClassifier(nn.Module, ABC):
    """Abstract base for all emotion classifiers."""

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before the classification head."""

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (train head only)."""

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""


# ---------------------------------------------------------------------------
# Shared classifier head builder
# ---------------------------------------------------------------------------

def _build_head(feature_dim: int, hidden_dim: int, dropout: float, dropout2: float) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout2),
        nn.Linear(hidden_dim, NUM_CLASSES),
    )


# ---------------------------------------------------------------------------
# EfficientNet-B3
# ---------------------------------------------------------------------------

class EfficientNetB3Classifier(EmotionClassifier):
    """EfficientNet-B3 backbone with custom classifier head."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        cfg = MODEL_CONFIGS["efficientnet_b3"]
        self.backbone = timm.create_model(
            cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="",
        )
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Pool and flatten only (skip dropout and linear layers)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        return pooled.flatten(1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------------------------

class EfficientNetB0Classifier(EmotionClassifier):
    """EfficientNet-B0 backbone with custom classifier head."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        cfg = MODEL_CONFIGS["efficientnet_b0"]
        self.backbone = timm.create_model(
            cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="",
        )
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        return pooled.flatten(1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# MobileNetV3-Large
# ---------------------------------------------------------------------------

class MobileNetV3Classifier(EmotionClassifier):
    """MobileNetV3-Large backbone with custom classifier head."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        cfg = MODEL_CONFIGS["mobilenet_v3"]
        self.backbone = timm.create_model(
            cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="",
        )
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        return pooled.flatten(1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# ResNet50 + CBAM
# ---------------------------------------------------------------------------

class ResNet50CBAMClassifier(EmotionClassifier):
    """ResNet50 backbone with CBAM inserted after each residual stage."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        cfg = MODEL_CONFIGS["resnet50_cbam"]
        self.backbone = timm.create_model(
            cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="",
        )

        # Insert CBAM modules after each residual layer group
        # ResNet50 stages: layer1(256), layer2(512), layer3(1024), layer4(2048)
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone with CBAM after each stage."""
        # timm resnet50 exposes layer1..layer4 via forward_features internals
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.cbam1(x)

        x = self.backbone.layer2(x)
        x = self.cbam2(x)

        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_backbone(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, 1)
        return pooled.flatten(1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[EmotionClassifier]] = {
    "efficientnet_b3": EfficientNetB3Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "mobilenet_v3": MobileNetV3Classifier,
    "resnet50_cbam": ResNet50CBAMClassifier,
}


def create_model(model_name: str, pretrained: bool = True) -> EmotionClassifier:
    """Factory function to create a model by name.

    Args:
        model_name: One of 'efficientnet_b3', 'efficientnet_b0',
                    'mobilenet_v3', 'resnet50_cbam'.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        An EmotionClassifier instance.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return _MODEL_REGISTRY[model_name](pretrained=pretrained)
