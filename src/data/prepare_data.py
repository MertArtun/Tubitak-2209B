"""Prepare FER2013 CSV data into a 3-class image folder structure.

Usage:
    python -m src.data.prepare_data --input data/raw/fer2013.csv --output data/processed
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import (
    CLASS_NAMES,
    FER_TO_3CLASS_IDX,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare FER2013 data for training.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/fer2013.csv",
        help="Path to the FER2013 CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed images.",
    )
    return parser.parse_args()


def load_and_map(csv_path: str) -> tuple[list[np.ndarray], list[int]]:
    """Load FER2013 CSV and map to 3 classes, excluding surprise.

    Args:
        csv_path: Path to fer2013.csv.

    Returns:
        Tuple of (images, labels) where images are 48x48 RGB numpy arrays.
    """
    df = pd.read_csv(csv_path)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        original_label = int(row["emotion"])
        mapped = FER_TO_3CLASS_IDX.get(original_label)
        if mapped is None:
            continue  # exclude surprise

        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        images.append(img)
        labels.append(mapped)

    return images, labels


def save_images(
    images: list[np.ndarray],
    labels: list[int],
    output_dir: Path,
    split_name: str,
) -> None:
    """Save images to disk in class-based folder structure.

    Args:
        images: List of RGB numpy arrays.
        labels: Corresponding class indices.
        output_dir: Base output directory.
        split_name: One of 'train', 'val', 'test'.
    """
    for class_name in CLASS_NAMES:
        (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    counters: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}
    for img, label in zip(images, labels):
        class_name = CLASS_NAMES[label]
        idx = counters[label]
        counters[label] += 1
        filename = output_dir / split_name / class_name / f"{class_name}_{idx:05d}.png"
        # Convert RGB back to BGR for cv2.imwrite
        cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def print_statistics(
    train_labels: list[int],
    val_labels: list[int],
    test_labels: list[int],
) -> None:
    """Print class distribution for each split."""
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = Counter(labels)
        total = len(labels)
        print(f"\n{split_name} set: {total} samples")
        for idx, name in enumerate(CLASS_NAMES):
            count = counts.get(idx, 0)
            pct = 100.0 * count / total if total > 0 else 0.0
            print(f"  {name}: {count} ({pct:.1f}%)")


def main() -> None:
    """Run the data preparation pipeline."""
    args = parse_args()

    print(f"Loading FER2013 data from {args.input} ...")
    images, labels = load_and_map(args.input)
    print(f"Total samples after mapping (surprise excluded): {len(images)}")

    # Stratified train / (val+test) split
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_imgs, valtest_imgs, train_labels, valtest_labels = train_test_split(
        images,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Split val+test into val and test
    relative_test_ratio = TEST_RATIO / val_test_ratio
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        valtest_imgs,
        valtest_labels,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=RANDOM_SEED,
    )

    output_dir = Path(args.output)
    print(f"\nSaving images to {output_dir} ...")

    save_images(train_imgs, train_labels, output_dir, "train")
    save_images(val_imgs, val_labels, output_dir, "val")
    save_images(test_imgs, test_labels, output_dir, "test")

    print_statistics(train_labels, val_labels, test_labels)
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
