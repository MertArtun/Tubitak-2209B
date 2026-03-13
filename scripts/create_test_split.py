"""Create stratified train/val/test splits from processed_merged data.

Pools all images from train/ and val/ in processed_merged,
then splits into clean train/val/test with natural class distribution.

Usage:
    python scripts/create_test_split.py [--data_dir ...] [--output_dir ...]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from configs.config import (
    CLASS_NAMES,
    CLEANED_DATA_DIR,
    MERGED_TEST_RATIO,
    MERGED_VAL_RATIO,
    PROCESSED_MERGED_DIR,
    RANDOM_SEED,
)


def create_test_split(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Create stratified train/val/test splits.

    Args:
        data_dir: Source directory with train/ and val/ subdirs.
        output_dir: Destination for cleaned splits.

    Returns:
        Summary dict with split counts.
    """
    if data_dir is None:
        data_dir = PROCESSED_MERGED_DIR
    if output_dir is None:
        output_dir = CLEANED_DATA_DIR

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Collect all images from train/ and val/
    all_files: list[Path] = []
    all_labels: list[str] = []

    for split_name in ["train", "val"]:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping.")
            continue
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    all_files.append(img_path)
                    all_labels.append(class_name)

    if not all_files:
        print(f"ERROR: No images found in {data_dir}")
        return {"error": "No images found"}

    print(f"Total images pooled: {len(all_files)}")
    for cls in CLASS_NAMES:
        count = all_labels.count(cls)
        print(f"  {cls}: {count}")

    # Stratified split: first separate test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=MERGED_TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=all_labels,
    )

    # Then separate val from remaining
    val_ratio_adjusted = MERGED_VAL_RATIO / (1.0 - MERGED_TEST_RATIO)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val_labels,
    )

    # Copy files to output directory
    summary: dict[str, dict[str, int]] = {}
    for split_name, files, labels in [
        ("train", train_files, train_labels),
        ("val", val_files, val_labels),
        ("test", test_files, test_labels),
    ]:
        split_counts: dict[str, int] = {}
        for file_path, label in zip(files, labels):
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / file_path.name

            # Handle name collisions
            if dest_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(str(file_path), str(dest_path))
            split_counts[label] = split_counts.get(label, 0) + 1

        summary[split_name] = split_counts
        total = sum(split_counts.values())
        print(f"\n{split_name}: {total} images")
        for cls in CLASS_NAMES:
            print(f"  {cls}: {split_counts.get(cls, 0)}")

    print(f"\nOutput directory: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified test split.")
    parser.add_argument(
        "--data_dir", type=str, default=str(PROCESSED_MERGED_DIR),
        help="Source data directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(CLEANED_DATA_DIR),
        help="Output directory for cleaned splits.",
    )
    args = parser.parse_args()

    create_test_split(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
