"""Data augmentation pipelines using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> A.Compose:
    """Return augmentation pipeline for training data."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(
            max_holes=1,
            max_height=int(IMAGE_SIZE * 0.2),
            max_width=int(IMAGE_SIZE * 0.2),
            min_height=int(IMAGE_SIZE * 0.05),
            min_width=int(IMAGE_SIZE * 0.05),
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Return augmentation pipeline for validation / test data."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
