"""Orchestrator for the full data cleaning pipeline.

Runs each step in sequence:
report -> corrupt -> duplicates -> FER+ -> split -> final report.

Usage:
    python scripts/run_data_cleaning.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from configs.config import (
    CLEANED_DATA_DIR,
    CLEANING_REPORTS_DIR,
    PROCESSED_MERGED_DIR,
    QUARANTINE_DIR,
)


def run_pipeline(dry_run: bool = False) -> bool:
    """Execute the full data cleaning pipeline.

    Returns:
        True when all executed steps succeed, False otherwise.
    """
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print(f"Mode: {'DRY RUN (report only)' if dry_run else 'FULL (quarantine enabled)'}")
    print(f"Source: {PROCESSED_MERGED_DIR}")
    print(f"Output: {CLEANED_DATA_DIR}")
    print("=" * 60)

    if not PROCESSED_MERGED_DIR.exists():
        print(f"\nERROR: Source directory not found: {PROCESSED_MERGED_DIR}")
        print("Please run prepare_all_datasets.py first.")
        return False

    CLEANING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    steps: list[tuple[str, Callable[[], None]]] = [
        ("1/6 - Initial Dataset Report", _step_initial_report),
        ("2/6 - Detect Corrupt Images", lambda: _step_corrupt(dry_run=dry_run)),
        ("3/6 - Detect Duplicates", lambda: _step_duplicates(dry_run=dry_run)),
        ("4/6 - FER+ Label Analysis", _step_ferplus),
    ]
    if not dry_run:
        steps.extend([
            ("5/6 - Create Stratified Splits", _step_split),
            ("6/6 - Final Dataset Report", _step_final_report),
        ])

    failed_steps: list[str] = []
    for title, func in steps:
        ok = _run_step(title, func)
        if not ok:
            failed_steps.append(title)
            break

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline finished in {elapsed:.1f}s")
    print(f"Reports: {CLEANING_REPORTS_DIR}")
    if not dry_run:
        print(f"Cleaned data: {CLEANED_DATA_DIR}")
        print(f"Quarantine: {QUARANTINE_DIR}")

    if failed_steps:
        print(f"FAILED steps: {failed_steps}")
        print("=" * 60)
        return False

    if dry_run:
        print("Skipped in dry-run: split creation, final cleaned-data report")
    print("All executed steps succeeded.")
    print("=" * 60)
    return True


def _run_step(title: str, func: Callable[[], None]) -> bool:
    """Run a pipeline step with timing and fail-fast error handling."""
    print(f"\n{'-' * 60}")
    print(f"[{title}]")
    print(f"{'-' * 60}")
    t0 = time.time()
    try:
        func()
    except Exception as exc:
        print(f"ERROR in {title}: {exc}")
        return False
    elapsed = time.time() - t0
    print(f"[{title}] completed in {elapsed:.1f}s")
    return True


def _step_initial_report() -> None:
    from scripts.generate_dataset_report import generate_report

    generate_report(data_dir=PROCESSED_MERGED_DIR, report_dir=CLEANING_REPORTS_DIR)


def _step_corrupt(dry_run: bool) -> None:
    from scripts.detect_corrupt_images import detect_corrupt_images

    detect_corrupt_images(
        data_dir=PROCESSED_MERGED_DIR,
        quarantine_dir=QUARANTINE_DIR,
        report_dir=CLEANING_REPORTS_DIR,
        dry_run=dry_run,
    )


def _step_duplicates(dry_run: bool) -> None:
    from scripts.detect_duplicates import detect_duplicates

    detect_duplicates(
        data_dir=PROCESSED_MERGED_DIR,
        quarantine_dir=QUARANTINE_DIR,
        report_dir=CLEANING_REPORTS_DIR,
        dry_run=dry_run,
    )


def _step_ferplus() -> None:
    from scripts.fix_fer_labels import analyze_ferplus

    analyze_ferplus(apply=False)


def _step_split() -> None:
    from scripts.create_test_split import create_test_split

    create_test_split()


def _step_final_report() -> None:
    from scripts.generate_dataset_report import generate_report

    generate_report(data_dir=CLEANED_DATA_DIR, report_dir=CLEANING_REPORTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data cleaning pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate reports only; do not move/delete files or create cleaned split.",
    )
    args = parser.parse_args()

    success = run_pipeline(dry_run=args.dry_run)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
