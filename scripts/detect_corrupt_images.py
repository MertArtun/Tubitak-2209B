"""Detect and quarantine corrupt, too-small, or blank images.

Usage:
    python scripts/detect_corrupt_images.py [--remove]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from configs.config import (
    CLEANING_REPORTS_DIR,
    MIN_FILE_SIZE_BYTES,
    MIN_IMAGE_SIZE,
    PROCESSED_MERGED_DIR,
    QUARANTINE_DIR,
)


def detect_corrupt_images(
    data_dir: Path,
    quarantine_dir: Path,
    report_dir: Path,
    remove: bool = False,
    dry_run: bool = False,
) -> dict:
    """Scan images for corruption, size, and variance issues.

    Args:
        data_dir: Root directory with train/val/test subdirs.
        quarantine_dir: Where to move flagged files.
        report_dir: Where to save the JSON report.
        remove: If True, delete files instead of quarantining.
        dry_run: If True, only report issues and do not mutate files.

    Returns:
        Report dict with detected issues.
    """
    quarantine_corrupt = quarantine_dir / "corrupt"
    report_dir.mkdir(parents=True, exist_ok=True)

    issues: dict[str, list[str]] = {
        "unreadable": [],
        "too_small_file": [],
        "too_small_image": [],
        "low_variance": [],
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_images = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    print(f"Scanning {len(all_images)} images in {data_dir} ...")

    for img_path in all_images:
        rel_path = str(img_path.relative_to(data_dir))

        # File size check
        if img_path.stat().st_size < MIN_FILE_SIZE_BYTES:
            issues["too_small_file"].append(rel_path)
            _quarantine_or_remove(img_path, quarantine_corrupt, data_dir, remove, dry_run)
            continue

        # Read check
        img = cv2.imread(str(img_path))
        if img is None:
            issues["unreadable"].append(rel_path)
            _quarantine_or_remove(img_path, quarantine_corrupt, data_dir, remove, dry_run)
            continue

        h, w = img.shape[:2]

        # Dimension check
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            issues["too_small_image"].append(rel_path)
            _quarantine_or_remove(img_path, quarantine_corrupt, data_dir, remove, dry_run)
            continue

        # Variance check (flat/blank images)
        if np.var(img.astype(np.float32)) < 1.0:
            issues["low_variance"].append(rel_path)
            _quarantine_or_remove(img_path, quarantine_corrupt, data_dir, remove, dry_run)
            continue

    report = {
        "source_dir": str(data_dir),
        "total_scanned": len(all_images),
        "issues": issues,
        "total_flagged": sum(len(v) for v in issues.values()),
        "action": "reported_only" if dry_run else ("removed" if remove else "quarantined"),
        "dry_run": dry_run,
    }

    report_path = report_dir / "corrupt_images.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")
    _print_summary(report)
    return report


def _quarantine_or_remove(
    file_path: Path,
    quarantine_dir: Path,
    source_root: Path,
    remove: bool,
    dry_run: bool,
) -> None:
    """Move file to quarantine or delete it."""
    if dry_run:
        return
    if remove:
        file_path.unlink()
        return
    rel = file_path.relative_to(source_root)
    dest = quarantine_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), str(dest))


def _print_summary(report: dict) -> None:
    """Print a human-readable summary."""
    issues = report["issues"]
    print("\n--- Corrupt Image Report ---")
    print(f"Total scanned : {report['total_scanned']}")
    print(f"Unreadable    : {len(issues['unreadable'])}")
    print(f"Too small file: {len(issues['too_small_file'])}")
    print(f"Too small img : {len(issues['too_small_image'])}")
    print(f"Low variance  : {len(issues['low_variance'])}")
    print(f"Total flagged : {report['total_flagged']}")
    print(f"Action        : {report['action']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect corrupt images.")
    parser.add_argument(
        "--data_dir", type=str, default=str(PROCESSED_MERGED_DIR),
        help="Root data directory to scan.",
    )
    parser.add_argument(
        "--remove", action="store_true",
        help="Delete flagged files instead of quarantining.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only; do not move/delete files.",
    )
    args = parser.parse_args()
    if args.remove and args.dry_run:
        parser.error("--remove and --dry-run cannot be used together.")

    detect_corrupt_images(
        data_dir=Path(args.data_dir),
        quarantine_dir=QUARANTINE_DIR,
        report_dir=CLEANING_REPORTS_DIR,
        remove=args.remove,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
