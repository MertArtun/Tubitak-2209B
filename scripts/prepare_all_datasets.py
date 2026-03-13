"""Prepare and merge all downloaded datasets into a unified 3-class structure.

Reads datasets from data/raw/, maps emotions to 3 classes (negative/neutral/positive),
and writes the results to data/processed_merged/.

Each dataset has its own format and label mapping, handled by dedicated processors.

Usage:
    python scripts/prepare_all_datasets.py
    python scripts/prepare_all_datasets.py --dataset fer2013-images  # Single dataset
    python scripts/prepare_all_datasets.py --list                     # List datasets
    python scripts/prepare_all_datasets.py --stats-only               # Show stats only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASS_NAMES  # noqa: E402

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MERGED_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_merged"

# Image extensions to include
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ── Emotion label mappings per dataset ───────────────────────────────────────

# Standard 7-emotion → 3-class mapping (used by most datasets)
EMOTION_7_TO_3: dict[str, str | None] = {
    "angry": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "fearful": "negative",
    "sad": "negative",
    "sadness": "negative",
    "happy": "positive",
    "happiness": "positive",
    "neutral": "neutral",
    "surprise": None,  # excluded
    "surprised": None,
    "contempt": "negative",
}

# FER2013 numeric label → 3-class
FER_LABEL_TO_3: dict[int, str | None] = {
    0: "negative",   # angry
    1: "negative",   # disgust
    2: "negative",   # fear
    3: "positive",   # happy
    4: "negative",   # sad
    5: None,         # surprise
    6: "neutral",    # neutral
}

# AffectNet standard numeric label → 3-class
AFFECTNET_LABEL_TO_3: dict[int, str | None] = {
    0: "neutral",    # neutral
    1: "positive",   # happy
    2: "negative",   # sad
    3: None,         # surprise
    4: "negative",   # fear
    5: "negative",   # disgust
    6: "negative",   # anger
    7: "negative",   # contempt
}

# AffectNet YOLO format label → 3-class (from data.yaml names order)
# 0=Anger, 1=Contempt, 2=Disgust, 3=Fear, 4=Happy, 5=Neutral, 6=Sad, 7=Surprise
AFFECTNET_YOLO_LABEL_TO_3: dict[int, str | None] = {
    0: "negative",   # anger
    1: "negative",   # contempt
    2: "negative",   # disgust
    3: "negative",   # fear
    4: "positive",   # happy
    5: "neutral",    # neutral
    6: "negative",   # sad
    7: None,         # surprise
}

# RAF-DB numeric label → 3-class (1-indexed)
RAFDB_LABEL_TO_3: dict[int, str | None] = {
    1: None,         # surprise
    2: "negative",   # fear
    3: "negative",   # disgust
    4: "positive",   # happiness
    5: "negative",   # sadness
    6: "negative",   # anger
    7: "neutral",    # neutral
}


def map_emotion_name(name: str) -> str | None:
    """Map an emotion name (case-insensitive) to 3-class label.

    Args:
        name: Emotion name string.

    Returns:
        One of 'negative', 'neutral', 'positive', or None if excluded.
    """
    return EMOTION_7_TO_3.get(name.strip().lower())


def is_image(path: Path) -> bool:
    """Check if a path is an image file."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def copy_image(src: Path, dst_dir: Path, prefix: str, counter: int) -> Path:
    """Copy an image to the destination with a standardized name.

    Args:
        src: Source image path.
        dst_dir: Destination directory.
        prefix: Filename prefix (dataset name).
        counter: Incrementing counter for unique names.

    Returns:
        Destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    dst = dst_dir / f"{prefix}_{counter:06d}{ext}"
    shutil.copy2(src, dst)
    return dst


# ── Dataset processors ───────────────────────────────────────────────────────


def process_fer2013_images(raw_dir: Path, output_dir: Path) -> Counter:
    """Process FER2013 folder format (msambare/fer2013).

    Expected structure: {raw_dir}/train/{emotion}/, {raw_dir}/test/{emotion}/

    Args:
        raw_dir: Path to data/raw/fer2013-images/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    for split_dir in sorted(raw_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            mapped = map_emotion_name(emotion_dir.name)
            if mapped is None:
                continue
            for img_path in sorted(emotion_dir.iterdir()):
                if is_image(img_path):
                    dst_split = "train" if split_dir.name == "train" else "val"
                    copy_image(
                        img_path,
                        output_dir / dst_split / mapped,
                        "fer2013img",
                        counter,
                    )
                    counter += 1
                    stats[mapped] += 1

    return stats


def process_face_expression(raw_dir: Path, output_dir: Path) -> Counter:
    """Process Oheix face expression dataset.

    Expected structure: {raw_dir}/images/images/train/{emotion}/, .../validation/{emotion}/

    Args:
        raw_dir: Path to data/raw/face-expression/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    # Try multiple possible structures
    candidates = [
        raw_dir / "images" / "images",
        raw_dir / "images",
        raw_dir,
    ]

    base = None
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            # Check if it has train/validation subdirs
            if (candidate / "train").is_dir() or (candidate / "validation").is_dir():
                base = candidate
                break

    if base is None:
        print(f"    WARNING: Could not find expected structure in {raw_dir}")
        return stats

    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        out_split = "train" if "train" in split_dir.name.lower() else "val"
        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            mapped = map_emotion_name(emotion_dir.name)
            if mapped is None:
                continue
            for img_path in sorted(emotion_dir.iterdir()):
                if is_image(img_path):
                    copy_image(
                        img_path,
                        output_dir / out_split / mapped,
                        "faceexpr",
                        counter,
                    )
                    counter += 1
                    stats[mapped] += 1

    return stats


def process_rafdb(raw_dir: Path, output_dir: Path) -> Counter:
    """Process RAF-DB dataset.

    Actual structure:
      - DATASET/train/{1-7}/ and DATASET/test/{1-7}/ (numeric emotion folders)
      - train_labels.csv and test_labels.csv with columns: image,label
      - Labels (1-indexed): 1=surprise,2=fear,3=disgust,4=happiness,5=sadness,6=anger,7=neutral

    Args:
        raw_dir: Path to data/raw/raf-db/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    # Use CSV label files + DATASET folder with images
    import csv as csv_mod

    for split_csv, split_name in [
        ("train_labels.csv", "train"),
        ("test_labels.csv", "val"),
    ]:
        csv_path = raw_dir / split_csv
        if not csv_path.exists():
            continue

        # Build image lookup from DATASET/{split}/ folders
        dataset_split_dir = raw_dir / "DATASET" / split_name
        if split_name == "val":
            dataset_split_dir = raw_dir / "DATASET" / "test"

        img_lookup: dict[str, Path] = {}
        if dataset_split_dir.is_dir():
            for img_path in dataset_split_dir.rglob("*"):
                if is_image(img_path):
                    img_lookup[img_path.name] = img_path

        with open(csv_path, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                img_name = row.get("image", "").strip()
                try:
                    label_id = int(row.get("label", ""))
                except (ValueError, TypeError):
                    continue

                mapped = RAFDB_LABEL_TO_3.get(label_id)
                if mapped is None:
                    continue

                img_path = img_lookup.get(img_name)
                if img_path:
                    out_split = "train" if split_name == "train" else "val"
                    copy_image(
                        img_path, output_dir / out_split / mapped, "rafdb", counter
                    )
                    counter += 1
                    stats[mapped] += 1

    return stats


def process_affectnet_yolo(raw_dir: Path, output_dir: Path) -> Counter:
    """Process AffectNet YOLO format dataset.

    Actual structure: YOLO_format/{train,valid,test}/{images,labels}/
    Labels: 0=Anger,1=Contempt,2=Disgust,3=Fear,4=Happy,5=Neutral,6=Sad,7=Surprise
    Each label .txt has: class_id cx cy w h

    Args:
        raw_dir: Path to data/raw/affectnet-yolo/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    # Find the YOLO_format base directory
    yolo_base = raw_dir / "YOLO_format"
    if not yolo_base.is_dir():
        yolo_base = raw_dir  # fallback

    for split_name in ["train", "valid", "test"]:
        labels_dir = yolo_base / split_name / "labels"
        images_dir = yolo_base / split_name / "images"

        if not labels_dir.is_dir() or not images_dir.is_dir():
            continue

        out_split = "train" if split_name == "train" else "val"

        for label_file in sorted(labels_dir.glob("*.txt")):
            with open(label_file) as f:
                first_line = f.readline().strip()
                if not first_line:
                    continue
                try:
                    class_id = int(first_line.split()[0])
                except (ValueError, IndexError):
                    continue

            mapped = AFFECTNET_YOLO_LABEL_TO_3.get(class_id)
            if mapped is None:
                continue

            # Find corresponding image
            stem = label_file.stem
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path:
                copy_image(
                    img_path, output_dir / out_split / mapped, "affyolo", counter
                )
                counter += 1
                stats[mapped] += 1

    return stats


def process_affectnet_train(raw_dir: Path, output_dir: Path) -> Counter:
    """Process AffectNet training data (folder structure with emotion names or IDs).

    Args:
        raw_dir: Path to data/raw/affectnet-train/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    for subdir in sorted(raw_dir.rglob("*")):
        if not subdir.is_dir():
            continue

        # Try by emotion name
        mapped = map_emotion_name(subdir.name)

        # Try by numeric ID
        if mapped is None:
            try:
                label_id = int(subdir.name)
                mapped = AFFECTNET_LABEL_TO_3.get(label_id)
            except ValueError:
                pass

        if mapped is None:
            continue

        for img_path in sorted(subdir.iterdir()):
            if is_image(img_path):
                copy_image(
                    img_path, output_dir / "train" / mapped, "afftrain", counter
                )
                counter += 1
                stats[mapped] += 1

    return stats


def process_expw(raw_dir: Path, output_dir: Path) -> Counter:
    """Process ExpW (Expression in the Wild) dataset.

    Actual structure:
      - label.lst: image_name face_id x y w h confidence expression_label
      - origin.7z.* : split 7z archives containing the images (need 7z to extract)
      - Labels: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral

    Args:
        raw_dir: Path to data/raw/expw/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    import subprocess

    stats: Counter = Counter()
    counter = 0

    label_file = raw_dir / "label.lst"
    if not label_file.exists():
        print("    WARNING: label.lst not found")
        return stats

    # Check if images are already extracted
    origin_dir = raw_dir / "origin"
    if not origin_dir.is_dir():
        # Need to extract 7z split archives
        first_archive = raw_dir / "origin.7z.001"
        if not first_archive.exists():
            first_archive = raw_dir / "origin.7z.001.001"
        if not first_archive.exists():
            # Try to find any 7z file
            archives = sorted(raw_dir.glob("origin.7z*"))
            if archives:
                first_archive = archives[0]
            else:
                print("    WARNING: No 7z archives found. Cannot extract images.")
                print("    Install 7z and extract manually: 7z x origin.7z.001.001 -oorigin/")
                return stats

        print("    Extracting 7z archives (this may take a while)...")
        try:
            result = subprocess.run(
                ["7z", "x", str(first_archive), f"-o{raw_dir}"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"    7z extraction failed: {result.stderr[:200]}")
                print("    Install 7z: brew install p7zip")
                return stats
        except FileNotFoundError:
            print("    WARNING: 7z not found. Install with: brew install p7zip")
            print("    Then run: cd data/raw/expw && 7z x origin.7z.001.001")
            return stats

    # Build image lookup
    all_images: dict[str, Path] = {}
    search_dirs = [origin_dir, raw_dir]
    for search_dir in search_dirs:
        if search_dir.is_dir():
            for img in search_dir.rglob("*"):
                if is_image(img):
                    all_images[img.name] = img
        if all_images:
            break

    if not all_images:
        print("    WARNING: No images found after extraction.")
        return stats

    print(f"    Found {len(all_images)} images, processing labels...")

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_name = parts[0]
            try:
                expression = int(parts[7])
            except (ValueError, IndexError):
                continue

            mapped = FER_LABEL_TO_3.get(expression)
            if mapped is None:
                continue

            if img_name in all_images:
                copy_image(
                    all_images[img_name],
                    output_dir / "train" / mapped,
                    "expw",
                    counter,
                )
                counter += 1
                stats[mapped] += 1

    return stats


def process_ck_plus(raw_dir: Path, output_dir: Path) -> Counter:
    """Process CK+ dataset.

    Actual structure: ckextended.csv with FER2013-like format (emotion,pixels,Usage).
    Uses same FER label mapping: 0=angry,1=disgust,2=fear,3=happy,4=sad,5=surprise,6=neutral

    Args:
        raw_dir: Path to data/raw/ck-plus/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    import cv2
    import numpy as np

    stats: Counter = Counter()
    counter = 0

    csv_path = raw_dir / "ckextended.csv"
    if not csv_path.exists():
        print(f"    WARNING: {csv_path} not found")
        return stats

    import csv as csv_mod

    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                emotion = int(row["emotion"])
            except (ValueError, KeyError):
                continue

            mapped = FER_LABEL_TO_3.get(emotion)
            if mapped is None:
                continue

            pixels_str = row.get("pixels", "")
            if not pixels_str:
                continue

            pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
            if pixels.size != 48 * 48:
                continue

            img = pixels.reshape(48, 48)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Save as image file
            dst_dir = output_dir / "train" / mapped
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / f"ckplus_{counter:06d}.png"
            cv2.imwrite(str(dst_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            counter += 1
            stats[mapped] += 1

    return stats


def process_kdef(raw_dir: Path, output_dir: Path) -> Counter:
    """Process KDEF database.

    KDEF filenames encode emotion: e.g., AM01ANS.JPG
    Code positions: AN=angry, DI=disgust, AF=afraid, HA=happy,
                    SA=sad, SU=surprise, NE=neutral

    Args:
        raw_dir: Path to data/raw/kdef/.
        output_dir: Merged output directory.

    Returns:
        Counter of class distribution.
    """
    stats: Counter = Counter()
    counter = 0

    KDEF_CODE_TO_3: dict[str, str | None] = {
        "AN": "negative",   # angry
        "DI": "negative",   # disgust
        "AF": "negative",   # afraid
        "HA": "positive",   # happy
        "SA": "negative",   # sad
        "SU": None,         # surprise
        "NE": "neutral",    # neutral
    }

    # Try folder structure with emotion names first
    found_folders = False
    for subdir in sorted(raw_dir.rglob("*")):
        if not subdir.is_dir():
            continue
        mapped = map_emotion_name(subdir.name)
        if mapped is None and subdir.name.lower() not in EMOTION_7_TO_3:
            continue
        if mapped is None:
            continue
        found_folders = True
        for img_path in sorted(subdir.iterdir()):
            if is_image(img_path):
                copy_image(
                    img_path, output_dir / "train" / mapped, "kdef", counter
                )
                counter += 1
                stats[mapped] += 1

    if found_folders:
        return stats

    # Parse emotion from filename
    for img_path in sorted(raw_dir.rglob("*")):
        if not is_image(img_path):
            continue
        name = img_path.stem.upper()
        # KDEF format: XX##EESS where EE is emotion code (position 4-5)
        if len(name) >= 6:
            emotion_code = name[4:6]
            mapped = KDEF_CODE_TO_3.get(emotion_code)
            if mapped is None:
                continue
            copy_image(
                img_path, output_dir / "train" / mapped, "kdef", counter
            )
            counter += 1
            stats[mapped] += 1

    return stats


# ── Dataset registry ─────────────────────────────────────────────────────────

PROCESSORS: dict[str, tuple[str, callable]] = {
    "fer2013-images": ("FER2013 (folder format)", process_fer2013_images),
    "face-expression": ("Face Expression (Oheix)", process_face_expression),
    "raf-db": ("RAF-DB", process_rafdb),
    "affectnet-yolo": ("AffectNet YOLO", process_affectnet_yolo),
    "affectnet-train": ("AffectNet Training", process_affectnet_train),
    "expw": ("ExpW", process_expw),
    "ck-plus": ("CK+", process_ck_plus),
    "kdef": ("KDEF", process_kdef),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare and merge all datasets into 3-class structure."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process a single dataset by name.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all processable datasets and exit.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics of existing merged data.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Input directory with raw datasets (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MERGED_OUTPUT_DIR),
        help=f"Output directory for merged data (default: {MERGED_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def print_stats(output_dir: Path) -> None:
    """Print class distribution statistics for the merged dataset.

    Args:
        output_dir: Path to data/processed_merged/.
    """
    print(f"\n{'='*60}")
    print(f"Statistics for: {output_dir}")
    print(f"{'='*60}")

    total_stats: Counter = Counter()

    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if not split_dir.exists():
            continue

        split_stats: Counter = Counter()
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = sum(1 for f in class_dir.iterdir() if is_image(f))
                split_stats[class_name] = count
                total_stats[class_name] += count

        total_split = sum(split_stats.values())
        if total_split > 0:
            print(f"\n  {split_name}: {total_split} images")
            for class_name in CLASS_NAMES:
                count = split_stats[class_name]
                pct = 100.0 * count / total_split if total_split > 0 else 0.0
                print(f"    {class_name}: {count:>8} ({pct:.1f}%)")

    grand_total = sum(total_stats.values())
    print(f"\n  TOTAL: {grand_total} images")
    for class_name in CLASS_NAMES:
        count = total_stats[class_name]
        pct = 100.0 * count / grand_total if grand_total > 0 else 0.0
        print(f"    {class_name}: {count:>8} ({pct:.1f}%)")
    print()


def main() -> None:
    """Run the dataset preparation and merging pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)

    if args.list:
        print("\nProcessable datasets:\n")
        print(f"{'Name':<20} {'Description'}")
        print("-" * 50)
        for name, (desc, _) in PROCESSORS.items():
            exists = (raw_dir / name).is_dir()
            status = "READY" if exists else "NOT FOUND"
            print(f"  {name:<20} {desc:<30} [{status}]")
        print()
        return

    if args.stats_only:
        if output_dir.exists():
            print_stats(output_dir)
        else:
            print(f"Output directory not found: {output_dir}")
        return

    # Create output directories
    for split_name in ["train", "val"]:
        for class_name in CLASS_NAMES:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    # Select datasets to process
    if args.dataset:
        if args.dataset not in PROCESSORS:
            print(f"ERROR: Unknown dataset '{args.dataset}'. Use --list to see available.")
            sys.exit(1)
        processors = {args.dataset: PROCESSORS[args.dataset]}
    else:
        processors = PROCESSORS

    print(f"\nProcessing datasets from: {raw_dir}")
    print(f"Output directory: {output_dir}\n")

    all_stats: dict[str, Counter] = {}

    for name, (desc, processor) in processors.items():
        dataset_dir = raw_dir / name
        if not dataset_dir.is_dir():
            print(f"  [{name}] SKIP - not downloaded")
            continue

        print(f"  [{name}] Processing {desc}...")
        stats = processor(dataset_dir, output_dir)

        if sum(stats.values()) == 0:
            print("    WARNING: No images processed. Check dataset structure.")
        else:
            total = sum(stats.values())
            print(f"    Processed {total} images: ", end="")
            parts = [f"{cls}={stats.get(cls, 0)}" for cls in CLASS_NAMES]
            print(", ".join(parts))

        all_stats[name] = stats

    # Print overall summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    grand_total: Counter = Counter()
    for name, stats in all_stats.items():
        total = sum(stats.values())
        if total > 0:
            print(f"  {name:<20}: {total:>8} images")
            for cls in CLASS_NAMES:
                grand_total[cls] += stats.get(cls, 0)

    total_all = sum(grand_total.values())
    print(f"\n  {'TOTAL':<20}: {total_all:>8} images")
    if total_all > 0:
        for cls in CLASS_NAMES:
            count = grand_total[cls]
            pct = 100.0 * count / total_all
            print(f"    {cls}: {count:>8} ({pct:.1f}%)")

    print_stats(output_dir)
    print("Preparation complete.")


if __name__ == "__main__":
    main()
