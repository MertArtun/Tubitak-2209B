"""PyTorch Dataset for FER2013 emotion data (3-class mapping)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.config import FER_TO_3CLASS_IDX


class FERDataset(Dataset):
    """Dataset that supports both FER2013 CSV pixel format and image folder format.

    The 7 original FER2013 emotions are mapped to 3 classes:
        - negative (0): angry, disgust, fear, sad
        - neutral  (1): neutral
        - positive (2): happy
    Surprise samples are excluded.

    Folder format expected:
        root/
            positive/
                img1.png
                ...
            negative/
                ...
            neutral/
                ...
    """

    def __init__(
        self,
        root: str | Path | None = None,
        csv_path: str | Path | None = None,
        transform: Callable | None = None,
        usage: str | None = None,
    ) -> None:
        """Initialize FERDataset.

        Provide *either* ``root`` (folder format) or ``csv_path`` (CSV format).

        Args:
            root: Path to image folder (positive/negative/neutral sub-dirs).
            csv_path: Path to FER2013 CSV file.
            transform: Albumentations transform pipeline.
            usage: Filter CSV rows by the 'Usage' column
                   ('Training', 'PublicTest', 'PrivateTest').
        """
        self.transform = transform
        self.images: list[np.ndarray] | list[Path] = []
        self.labels: list[int] = []
        self._from_csv = csv_path is not None

        if csv_path is not None:
            self._load_csv(Path(csv_path), usage)
        elif root is not None:
            self._load_folder(Path(root))
        else:
            raise ValueError("Either 'root' or 'csv_path' must be provided.")

    # ── loaders ──────────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: Path, usage: str | None) -> None:
        """Load data from FER2013 CSV with pixel strings."""
        df = pd.read_csv(csv_path)
        if usage is not None:
            df = df[df["Usage"] == usage]

        for _, row in df.iterrows():
            original_label = int(row["emotion"])
            mapped = FER_TO_3CLASS_IDX.get(original_label)
            if mapped is None:
                continue  # skip surprise

            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            img = pixels.reshape(48, 48)
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            self.images.append(img)
            self.labels.append(mapped)

    def _load_folder(self, root: Path) -> None:
        """Load data from image folder structure."""
        class_map = {"negative": 0, "neutral": 1, "positive": 2}
        for class_name, label in class_map.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.images.append(img_path)
                    self.labels.append(label)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        label = self.labels[idx]

        if self._from_csv:
            image = self.images[idx]  # already numpy RGB
        else:
            path = self.images[idx]
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label
