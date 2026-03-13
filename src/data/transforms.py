"""Data augmentation pipelines using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(phase: int = 1) -> A.Compose:
    """Return augmentation pipeline for training data.

    Args:
        phase: Training phase (1 = light for head-only, 2 = strong for fine-tuning).

    Returns:
        Albumentations Compose pipeline.
    """
    if phase == 1:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    # Phase 2: strong augmentation for full fine-tuning
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.1)),
        ], p=0.3),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1),
            A.OpticalDistortion(distort_limit=0.05),
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(int(IMAGE_SIZE * 0.05), int(IMAGE_SIZE * 0.25)),
            hole_width_range=(int(IMAGE_SIZE * 0.05), int(IMAGE_SIZE * 0.25)),
            fill=0,
            p=0.3,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
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
