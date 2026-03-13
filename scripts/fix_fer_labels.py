"""Analyze FER+ relabeling potential; apply only via deterministic manifest.

Usage:
    python scripts/fix_fer_labels.py
    python scripts/fix_fer_labels.py --apply --manifest results/data_cleaning/fer_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from configs.config import (
    CLASS_NAMES,
    CLEANING_REPORTS_DIR,
    FERPLUS_CONFIDENCE_THRESHOLD,
    PROCESSED_MERGED_DIR,
    RAW_DATA_DIR,
)

_FERPLUS_TO_3CLASS = {
    "neutral": "neutral",
    "happiness": "positive",
    "surprise": None,
    "sadness": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "contempt": "negative",
}

_FERPLUS_EMOTION_COLS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]


def analyze_ferplus(
    fer2013_csv: Path | None = None,
    ferplus_csv: Path | None = None,
    report_dir: Path | None = None,
    apply: bool = False,
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
) -> dict:
    """Analyze FER+ labels and optionally apply manifest-based relabeling."""
    if fer2013_csv is None:
        fer2013_csv = RAW_DATA_DIR / "fer2013" / "fer2013.csv"
    if ferplus_csv is None:
        ferplus_csv = RAW_DATA_DIR / "ferplus" / "fer2013new.csv"
    if report_dir is None:
        report_dir = CLEANING_REPORTS_DIR
    if data_dir is None:
        data_dir = PROCESSED_MERGED_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "fer2013_csv": str(fer2013_csv),
        "ferplus_csv": str(ferplus_csv),
        "alignment_verified": False,
        "total_rows": 0,
        "mappable_rows": 0,
        "low_confidence_rows": 0,
        "excluded_surprise_rows": 0,
        "class_distribution": {},
        "applied": False,
        "apply_mode": "manifest_only",
        "manifest_path": str(manifest_path) if manifest_path else None,
    }

    if not fer2013_csv.exists():
        report["error"] = f"FER2013 CSV not found: {fer2013_csv}"
        _save_report(report, report_dir)
        return report

    if not ferplus_csv.exists():
        report["error"] = f"FER+ CSV not found: {ferplus_csv}"
        _save_report(report, report_dir)
        return report

    try:
        fer2013_df = pd.read_csv(fer2013_csv)
        ferplus_df = pd.read_csv(ferplus_csv)
    except Exception as exc:
        report["error"] = f"Failed to read CSVs: {exc}"
        _save_report(report, report_dir)
        return report

    if len(fer2013_df) != len(ferplus_df):
        report["error"] = (
            f"Row count mismatch: fer2013={len(fer2013_df)}, ferplus={len(ferplus_df)}. "
            "Alignment cannot be verified."
        )
        _save_report(report, report_dir)
        return report

    report["alignment_verified"] = True
    report["total_rows"] = len(ferplus_df)

    available_cols = ferplus_df.columns.tolist()
    emotion_cols = [c for c in _FERPLUS_EMOTION_COLS if c in available_cols]
    if not emotion_cols:
        report["error"] = "Could not find emotion vote columns in FER+ CSV."
        _save_report(report, report_dir)
        return report

    class_dist = {c: 0 for c in CLASS_NAMES}
    low_confidence = 0
    excluded = 0

    for _, row in ferplus_df.iterrows():
        votes = {col: row.get(col, 0) for col in emotion_cols}
        total_votes = sum(v for v in votes.values() if pd.notna(v))
        if total_votes == 0:
            low_confidence += 1
            continue

        class_votes = {c: 0.0 for c in CLASS_NAMES}
        for emo, count in votes.items():
            if pd.isna(count):
                continue
            mapped_class = _FERPLUS_TO_3CLASS.get(emo)
            if mapped_class is None:
                excluded += float(count)
                continue
            class_votes[mapped_class] += float(count)

        total_mapped = sum(class_votes.values())
        if total_mapped == 0:
            excluded += 1
            continue

        winner = max(class_votes, key=lambda k: class_votes[k])
        confidence = class_votes[winner] / total_mapped
        if confidence < FERPLUS_CONFIDENCE_THRESHOLD:
            low_confidence += 1
            continue

        class_dist[winner] += 1

    report["mappable_rows"] = sum(class_dist.values())
    report["low_confidence_rows"] = low_confidence
    report["excluded_surprise_rows"] = int(excluded)
    report["class_distribution"] = class_dist

    if apply:
        if manifest_path is None:
            report["error"] = "Apply requested but no manifest was provided."
        elif not manifest_path.exists():
            report["error"] = f"Manifest not found: {manifest_path}"
        else:
            apply_summary = _apply_manifest(data_dir=data_dir, manifest_path=manifest_path)
            report["apply_summary"] = apply_summary
            report["applied"] = apply_summary["applied_moves"] > 0
    else:
        print("Dry run analysis complete; no relabeling applied.")

    _save_report(report, report_dir)
    return report


def _apply_manifest(data_dir: Path, manifest_path: Path) -> dict:
    """Apply relabel moves defined in a deterministic manifest JSON file."""
    payload = json.loads(manifest_path.read_text())
    raw_changes = payload["changes"] if isinstance(payload, dict) else payload

    applied_moves = 0
    skipped_invalid = 0
    skipped_missing = 0
    skipped_noop = 0

    for item in raw_changes:
        rel_path = item.get("path")
        new_class = item.get("new_class")
        if not rel_path or new_class not in CLASS_NAMES:
            skipped_invalid += 1
            continue

        rel = Path(rel_path)
        if len(rel.parts) < 3:
            skipped_invalid += 1
            continue

        src = data_dir / rel
        if not src.exists():
            skipped_missing += 1
            continue

        split_name = rel.parts[0]
        old_class = rel.parts[1]
        if old_class == new_class:
            skipped_noop += 1
            continue

        dst_dir = data_dir / split_name / new_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(src), str(dst))
        applied_moves += 1

    return {
        "data_dir": str(data_dir),
        "manifest_path": str(manifest_path),
        "requested_changes": len(raw_changes),
        "applied_moves": applied_moves,
        "skipped_invalid": skipped_invalid,
        "skipped_missing": skipped_missing,
        "skipped_noop": skipped_noop,
    }


def _save_report(report: dict, report_dir: Path) -> None:
    """Save analysis report to JSON."""
    report_path = report_dir / "fer_label_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")

    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print("\n--- FER+ Label Analysis ---")
    print(f"Total rows         : {report['total_rows']}")
    print(f"Alignment verified : {report['alignment_verified']}")
    print(f"Mappable rows      : {report['mappable_rows']}")
    print(f"Low confidence     : {report['low_confidence_rows']}")
    print(f"Class distribution : {report['class_distribution']}")
    if "apply_summary" in report:
        print(f"Apply summary      : {report['apply_summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze/apply FER+ label updates.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply relabeling moves from a deterministic manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to JSON manifest with label moves.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROCESSED_MERGED_DIR),
        help="Target dataset root for manifest apply mode.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    if args.apply and manifest_path is None:
        parser.error("--apply requires --manifest.")

    analyze_ferplus(
        apply=args.apply,
        manifest_path=manifest_path,
        data_dir=Path(args.data_dir),
    )


if __name__ == "__main__":
    main()
