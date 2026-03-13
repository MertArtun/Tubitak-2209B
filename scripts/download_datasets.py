"""Download all emotion recognition datasets for training.

Downloads datasets from Kaggle and GitHub into data/raw/ subdirectories.
Requires a valid Kaggle API token at ~/.kaggle/kaggle.json.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --tier 1        # Only Tier 1 datasets
    python scripts/download_datasets.py --dataset fer2013  # Single dataset
    python scripts/download_datasets.py --list           # List all datasets
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ── Dataset definitions ──────────────────────────────────────────────────────

DATASETS: list[dict[str, str | int]] = [
    # Tier 1 - Primary
    {
        "name": "fer2013",
        "slug": "deadskull7/fer2013",
        "tier": 1,
        "description": "FER2013 CSV format (~35,887 images as pixel strings)",
    },
    {
        "name": "fer2013-images",
        "slug": "msambare/fer2013",
        "tier": 1,
        "description": "FER2013 folder format (train/test with emotion subdirs)",
    },
    {
        "name": "ferplus",
        "slug": "",  # GitHub, not Kaggle
        "tier": 1,
        "source": "github",
        "description": "FER+ corrected labels from Microsoft",
    },
    # Tier 2 - Secondary
    {
        "name": "raf-db",
        "slug": "shuvoalok/raf-db-dataset",
        "tier": 2,
        "description": "RAF-DB dataset (~29,672 images)",
    },
    {
        "name": "affectnet-yolo",
        "slug": "fatihkgg/affectnet-yolo-format",
        "tier": 2,
        "description": "AffectNet subset in YOLO format",
    },
    {
        "name": "affectnet-train",
        "slug": "noamsegal/affectnet-training-data",
        "tier": 2,
        "description": "AffectNet training subset",
    },
    {
        "name": "expw",
        "slug": "shahzadabbas/expression-in-the-wild-expw-dataset",
        "tier": 2,
        "description": "Expression in the Wild (~91,793 images)",
    },
    # Tier 3 - Additional
    {
        "name": "face-expression",
        "slug": "jonathanoheix/face-expression-recognition-dataset",
        "tier": 3,
        "description": "Face Expression Recognition (~35K images)",
    },
    {
        "name": "ck-plus",
        "slug": "davilsena/ckdataset",
        "tier": 3,
        "description": "CK+ dataset (327 labeled sequences)",
    },
    {
        "name": "kdef",
        "slug": "chenrich/kdef-database",
        "tier": 3,
        "description": "KDEF database (4,900 images)",
    },
]

FERPLUS_LABEL_URL = (
    "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download emotion recognition datasets."
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Download only datasets of this tier (1=primary, 2=secondary, 3=additional).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Download a single dataset by name (e.g., 'fer2013', 'raf-db').",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help=f"Output directory for raw data (default: {RAW_DATA_DIR}).",
    )
    return parser.parse_args()


def check_kaggle_token() -> bool:
    """Check if Kaggle API token exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API token not found at ~/.kaggle/kaggle.json")
        print("Steps to set up:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def find_kaggle_cli() -> str:
    """Find the kaggle CLI executable path.

    Returns:
        Path to kaggle executable.
    """
    # Check in the same venv as this script
    venv_kaggle = Path(sys.executable).parent / "kaggle"
    if venv_kaggle.exists():
        return str(venv_kaggle)

    # Fallback to PATH
    kaggle_path = shutil.which("kaggle")
    if kaggle_path:
        return kaggle_path

    return "kaggle"


def download_kaggle_dataset(slug: str, output_dir: Path) -> bool:
    """Download and extract a Kaggle dataset.

    Args:
        slug: Kaggle dataset slug (e.g., 'deadskull7/fer2013').
        output_dir: Directory to extract files into.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle_cmd = find_kaggle_cli()

    try:
        result = subprocess.run(
            [
                kaggle_cmd,
                "datasets", "download",
                slug,
                "-p", str(output_dir),
                "--unzip",
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"  Kaggle CLI error: {result.stderr.strip()}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("  Download timed out (>10 min).")
        return False
    except FileNotFoundError:
        print("  Kaggle CLI not found. Install with: pip install kaggle")
        return False


def download_ferplus_labels(output_dir: Path) -> bool:
    """Download FER+ corrected labels from GitHub.

    Args:
        output_dir: Directory to save fer2013new.csv.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fer2013new.csv"

    try:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(output_file), FERPLUS_LABEL_URL],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"  curl error: {result.stderr.strip()}")
            return False

        if output_file.exists() and output_file.stat().st_size > 0:
            return True

        print("  Downloaded file is empty.")
        return False

    except subprocess.TimeoutExpired:
        print("  Download timed out.")
        return False


def is_dataset_downloaded(name: str, output_dir: Path) -> bool:
    """Check if a dataset directory exists and has content.

    Args:
        name: Dataset name (subdirectory name).
        output_dir: Base output directory.

    Returns:
        True if dataset directory exists and is non-empty.
    """
    dataset_dir = output_dir / name
    if not dataset_dir.exists():
        return False

    # Check if directory has any files (recursively)
    try:
        next(dataset_dir.rglob("*"))
        return True
    except StopIteration:
        return False


def list_datasets() -> None:
    """Print all available datasets."""
    print("\nAvailable datasets:\n")
    print(f"{'Name':<20} {'Tier':<6} {'Source':<8} {'Description'}")
    print("-" * 80)
    for ds in DATASETS:
        source = ds.get("source", "kaggle")
        print(f"{ds['name']:<20} {ds['tier']:<6} {source:<8} {ds['description']}")
    print()


def download_dataset(ds: dict, output_dir: Path) -> bool:
    """Download a single dataset.

    Args:
        ds: Dataset definition dict.
        output_dir: Base output directory.

    Returns:
        True if download succeeded or already exists, False otherwise.
    """
    name = ds["name"]
    dataset_dir = output_dir / name

    if is_dataset_downloaded(name, output_dir):
        print(f"  [SKIP] {name}: already downloaded")
        return True

    if ds.get("source") == "github":
        print(f"  [DOWNLOAD] {name}: FER+ labels from GitHub...")
        return download_ferplus_labels(dataset_dir)

    slug = ds["slug"]
    print(f"  [DOWNLOAD] {name}: {slug}...")
    return download_kaggle_dataset(slug, dataset_dir)


def main() -> None:
    """Run the dataset download pipeline."""
    args = parse_args()

    if args.list:
        list_datasets()
        return

    output_dir = Path(args.output_dir)

    # Filter datasets
    datasets = DATASETS
    if args.dataset:
        datasets = [ds for ds in DATASETS if ds["name"] == args.dataset]
        if not datasets:
            print(f"ERROR: Unknown dataset '{args.dataset}'. Use --list to see available.")
            sys.exit(1)
    elif args.tier:
        datasets = [ds for ds in DATASETS if ds["tier"] <= args.tier]

    # Check Kaggle token for Kaggle datasets
    needs_kaggle = any(ds.get("source") != "github" for ds in datasets)
    if needs_kaggle and not check_kaggle_token():
        sys.exit(1)

    print(f"\nDownloading {len(datasets)} dataset(s) to {output_dir}/\n")

    results: dict[str, bool] = {}
    for ds in datasets:
        success = download_dataset(ds, output_dir)
        results[ds["name"]] = success

    # Summary
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'='*60}")
    print(f"Download complete: {succeeded} succeeded, {failed} failed")

    if failed:
        print("\nFailed datasets:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")
        sys.exit(1)

    print(f"\nAll datasets saved to: {output_dir}/")


if __name__ == "__main__":
    main()
