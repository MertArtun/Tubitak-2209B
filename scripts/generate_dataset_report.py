"""Generate dataset statistics report with visualizations.

Usage:
    python scripts/generate_dataset_report.py [--data_dir ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.config import CLASS_NAMES, CLEANED_DATA_DIR, CLEANING_REPORTS_DIR

# Known dataset prefixes for source attribution
_DATASET_PREFIXES = {
    "fer2013img_": "FER2013",
    "faceexpr_": "Face-Expression",
    "rafdb_": "RAF-DB",
    "affyolo_": "AffectNet-YOLO",
    "afftrain_": "AffectNet-Train",
    "expw_": "ExpW",
    "ckplus_": "CK+",
    "kdef_": "KDEF",
}


def generate_report(
    data_dir: Path | None = None,
    report_dir: Path | None = None,
) -> dict:
    """Generate comprehensive dataset statistics.

    Args:
        data_dir: Root directory with train/val/test subdirs.
        report_dir: Where to save reports and plots.

    Returns:
        Report dict.
    """
    if data_dir is None:
        data_dir = CLEANED_DATA_DIR
    if report_dir is None:
        report_dir = CLEANING_REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    splits_data: dict[str, dict[str, list[Path]]] = {}
    size_stats: list[tuple[int, int]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for split_name in ["train", "val", "test"]:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        splits_data[split_name] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                splits_data[split_name][class_name] = []
                continue
            files = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            splits_data[split_name][class_name] = files

            # Collect size stats and source info
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    h, w = img.shape[:2]
                    size_stats.append((w, h))

                # Source attribution
                fname = f.name.lower()
                matched = False
                for prefix, source in _DATASET_PREFIXES.items():
                    if fname.startswith(prefix):
                        source_counts[source] += 1
                        matched = True
                        break
                if not matched:
                    source_counts["Unknown"] += 1

    # Build report
    report: dict = {
        "data_dir": str(data_dir),
        "splits": {},
        "total_images": 0,
        "size_stats": {},
        "source_contribution": dict(source_counts),
    }

    # Split / class counts
    for split_name, classes in splits_data.items():
        split_info: dict[str, int] = {}
        for cls, files in classes.items():
            split_info[cls] = len(files)
        split_info["total"] = sum(split_info.values())
        report["splits"][split_name] = split_info
        report["total_images"] += split_info["total"]

    # Size statistics
    if size_stats:
        widths = [s[0] for s in size_stats]
        heights = [s[1] for s in size_stats]
        report["size_stats"] = {
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "mean_height": float(np.mean(heights)),
            "median_height": float(np.median(heights)),
        }

    # Save JSON report
    json_path = report_dir / "dataset_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Save Markdown report
    md_path = report_dir / "dataset_report.md"
    md_path.write_text(_generate_markdown(report))

    # Generate plots
    _plot_class_distribution(report, report_dir)
    _plot_source_contribution(report, report_dir)

    print(f"Reports saved to {report_dir}")
    _print_summary(report)
    return report


def _generate_markdown(report: dict) -> str:
    """Generate markdown report."""
    lines = ["# Dataset Report\n"]
    lines.append(f"**Source:** `{report['data_dir']}`\n")
    lines.append(f"**Total images:** {report['total_images']}\n")

    # Split table
    lines.append("## Split Distribution\n")
    lines.append("| Split | " + " | ".join(CLASS_NAMES) + " | Total |")
    lines.append("|-------|" + "|".join(["-------"] * (len(CLASS_NAMES) + 1)) + "|")

    for split_name, counts in report["splits"].items():
        row = f"| {split_name} |"
        for cls in CLASS_NAMES:
            row += f" {counts.get(cls, 0)} |"
        row += f" {counts.get('total', 0)} |"
        lines.append(row)

    # Size stats
    if report["size_stats"]:
        lines.append("\n## Image Size Statistics\n")
        stats = report["size_stats"]
        lines.append(f"- Width: min={stats['min_width']}, max={stats['max_width']}, "
                      f"mean={stats['mean_width']:.1f}, median={stats['median_width']:.1f}")
        lines.append(f"- Height: min={stats['min_height']}, max={stats['max_height']}, "
                      f"mean={stats['mean_height']:.1f}, median={stats['median_height']:.1f}")

    # Source contribution
    if report["source_contribution"]:
        lines.append("\n## Source Dataset Contribution\n")
        lines.append("| Source | Count |")
        lines.append("|--------|-------|")
        for source, count in sorted(report["source_contribution"].items(), key=lambda x: -x[1]):
            lines.append(f"| {source} | {count} |")

    return "\n".join(lines) + "\n"


def _plot_class_distribution(report: dict, report_dir: Path) -> None:
    """Bar chart of class distribution per split."""
    splits = report["splits"]
    if not splits:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    for i, (split_name, counts) in enumerate(splits.items()):
        values = [counts.get(cls, 0) for cls in CLASS_NAMES]
        ax.bar(x + i * width, values, width, label=split_name)

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution per Split")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_source_contribution(report: dict, report_dir: Path) -> None:
    """Pie chart of source dataset contribution."""
    sources = report["source_contribution"]
    if not sources:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(sources.keys())
    sizes = list(sources.values())

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Dataset Source Contribution")

    plt.tight_layout()
    plt.savefig(report_dir / "source_contribution.png", dpi=150)
    plt.close(fig)


def _print_summary(report: dict) -> None:
    """Print human-readable summary."""
    print("\n--- Dataset Report ---")
    print(f"Total images: {report['total_images']}")
    for split_name, counts in report["splits"].items():
        print(f"  {split_name}: {counts.get('total', 0)}")
        for cls in CLASS_NAMES:
            print(f"    {cls}: {counts.get(cls, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset report.")
    parser.add_argument(
        "--data_dir", type=str, default=str(CLEANED_DATA_DIR),
        help="Data directory to report on.",
    )
    args = parser.parse_args()

    generate_report(data_dir=Path(args.data_dir))


if __name__ == "__main__":
    main()
