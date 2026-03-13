"""Detect exact and near-duplicate images using perceptual hashing.

Usage:
    python scripts/detect_duplicates.py [--remove] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

from configs.config import (
    CLEANING_REPORTS_DIR,
    ENABLE_NEAR_DUP_SCAN,
    MAX_NEAR_DUP_CANDIDATES_PER_IMAGE,
    PHASH_HAMMING_THRESHOLD,
    PHASH_HASH_SIZE,
    PROCESSED_MERGED_DIR,
    QUARANTINE_DIR,
)


def detect_duplicates(
    data_dir: Path,
    quarantine_dir: Path,
    report_dir: Path,
    remove: bool = False,
    dry_run: bool = False,
    near_scan: bool | None = None,
) -> dict:
    """Find exact and near-duplicate images across all splits.

    Args:
        data_dir: Root directory with train/val/test subdirs.
        quarantine_dir: Where to move duplicate files.
        report_dir: Where to save the JSON report.
        remove: If True, delete duplicates instead of quarantining.
        dry_run: If True, only report issues and do not mutate files.
        near_scan: Override config for near-duplicate scan.

    Returns:
        Report dict with duplicate groups and cross-split leaks.
    """
    if near_scan is None:
        near_scan = ENABLE_NEAR_DUP_SCAN

    quarantine_dups = quarantine_dir / "duplicates"
    report_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_images = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    print(f"Hashing {len(all_images)} images (hash_size={PHASH_HASH_SIZE}) ...")

    hash_map: dict[str, list[str]] = defaultdict(list)
    file_hashes: dict[str, imagehash.ImageHash] = {}

    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img.convert("RGB"), hash_size=PHASH_HASH_SIZE)
            rel = str(img_path.relative_to(data_dir))
            hash_map[str(h)].append(rel)
            file_hashes[rel] = h
        except Exception:
            continue

    exact_groups = {k: v for k, v in hash_map.items() if len(v) > 1}

    near_groups: list[list[str]] = []
    candidate_checks = 0
    if near_scan:
        near_groups, candidate_checks = _find_near_duplicate_groups(file_hashes)

    cross_split_leaks = _find_cross_split_leaks(exact_groups, near_groups)

    quarantined_count = 0
    for group in list(exact_groups.values()) + near_groups:
        for dup_rel in group[1:]:
            dup_path = data_dir / dup_rel
            if dup_path.exists():
                _quarantine_or_remove(
                    dup_path,
                    quarantine_dups,
                    data_dir,
                    remove=remove,
                    dry_run=dry_run,
                )
                if not dry_run:
                    quarantined_count += 1

    report = {
        "source_dir": str(data_dir),
        "total_hashed": len(file_hashes),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_files": sum(len(v) - 1 for v in exact_groups.values()),
        "near_scan_enabled": near_scan,
        "candidate_checks": candidate_checks,
        "near_duplicate_groups": len(near_groups),
        "near_duplicate_files": sum(len(g) - 1 for g in near_groups),
        "cross_split_leaks": cross_split_leaks,
        "total_quarantined": quarantined_count,
        "action": "reported_only" if dry_run else ("removed" if remove else "quarantined"),
        "dry_run": dry_run,
        "exact_groups": exact_groups,
        "near_groups": near_groups,
    }

    report_path = report_dir / "duplicates_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {report_path}")
    _print_summary(report)
    return report


def _find_near_duplicate_groups(
    file_hashes: dict[str, imagehash.ImageHash],
) -> tuple[list[list[str]], int]:
    """Find near-duplicates using hash block indexing to avoid O(N^2) scans."""
    rels = sorted(file_hashes.keys())
    n = len(rels)
    if n < 2:
        return [], 0

    hashes = [file_hashes[r] for r in rels]
    bit_strings = [_hash_to_bits(h) for h in hashes]

    block_count = PHASH_HAMMING_THRESHOLD + 1
    block_index: dict[tuple[int, str], list[int]] = defaultdict(list)

    for idx, bits in enumerate(bit_strings):
        for block_idx, chunk in _iter_blocks(bits, block_count):
            block_index[(block_idx, chunk)].append(idx)

    parent = list(range(n))
    rank = [0] * n
    candidate_checks = 0

    for i, bits in enumerate(bit_strings):
        candidates: set[int] = set()
        for block_idx, chunk in _iter_blocks(bits, block_count):
            candidates.update(block_index[(block_idx, chunk)])
        candidates.discard(i)

        ordered = [j for j in sorted(candidates) if j > i]
        if MAX_NEAR_DUP_CANDIDATES_PER_IMAGE > 0:
            ordered = ordered[:MAX_NEAR_DUP_CANDIDATES_PER_IMAGE]

        for j in ordered:
            candidate_checks += 1
            dist = hashes[i] - hashes[j]
            if 0 < dist <= PHASH_HAMMING_THRESHOLD:
                _uf_union(parent, rank, i, j)

    groups: dict[int, list[str]] = defaultdict(list)
    for idx, rel in enumerate(rels):
        root = _uf_find(parent, idx)
        groups[root].append(rel)

    near_groups = [sorted(group) for group in groups.values() if len(group) > 1]
    near_groups.sort(key=lambda g: g[0])
    return near_groups, candidate_checks


def _hash_to_bits(h: imagehash.ImageHash) -> str:
    """Convert an ImageHash into a deterministic bitstring."""
    return "".join("1" if b else "0" for b in h.hash.flatten())


def _iter_blocks(bit_string: str, block_count: int) -> list[tuple[int, str]]:
    """Split bitstring into almost-equal contiguous blocks."""
    total = len(bit_string)
    block_len = total // block_count
    remainder = total % block_count
    blocks: list[tuple[int, str]] = []

    start = 0
    for idx in range(block_count):
        extra = 1 if idx < remainder else 0
        end = start + block_len + extra
        blocks.append((idx, bit_string[start:end]))
        start = end
    return blocks


def _uf_find(parent: list[int], x: int) -> int:
    """Find representative with path compression."""
    if parent[x] != x:
        parent[x] = _uf_find(parent, parent[x])
    return parent[x]


def _uf_union(parent: list[int], rank: list[int], a: int, b: int) -> None:
    """Union by rank."""
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def _find_cross_split_leaks(
    exact_groups: dict[str, list[str]],
    near_groups: list[list[str]],
) -> list[dict]:
    """Find images that appear in multiple splits (train/val/test)."""
    leaks = []
    all_groups = list(exact_groups.values()) + near_groups

    for group in all_groups:
        splits_found: dict[str, list[str]] = defaultdict(list)
        for rel_path in group:
            parts = Path(rel_path).parts
            if not parts:
                continue
            split_name = parts[0]
            splits_found[split_name].append(rel_path)

        if len(splits_found) > 1:
            leaks.append(dict(splits_found))

    return leaks


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
    print("\n--- Duplicate Report ---")
    print(f"Total hashed        : {report['total_hashed']}")
    print(f"Exact dup groups    : {report['exact_duplicate_groups']}")
    print(f"Exact dup files     : {report['exact_duplicate_files']}")
    print(f"Near scan enabled   : {report['near_scan_enabled']}")
    print(f"Candidate checks    : {report['candidate_checks']}")
    print(f"Near dup groups     : {report['near_duplicate_groups']}")
    print(f"Near dup files      : {report['near_duplicate_files']}")
    print(f"Cross-split leaks   : {len(report['cross_split_leaks'])}")
    print(f"Total quarantined   : {report['total_quarantined']}")
    print(f"Action              : {report['action']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate images.")
    parser.add_argument(
        "--data_dir", type=str, default=str(PROCESSED_MERGED_DIR),
        help="Root data directory to scan.",
    )
    parser.add_argument(
        "--remove", action="store_true",
        help="Delete duplicates instead of quarantining.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only; do not move/delete files.",
    )
    parser.add_argument(
        "--no-near", action="store_true",
        help="Disable near-duplicate scan (exact duplicates only).",
    )
    args = parser.parse_args()
    if args.remove and args.dry_run:
        parser.error("--remove and --dry-run cannot be used together.")

    detect_duplicates(
        data_dir=Path(args.data_dir),
        quarantine_dir=QUARANTINE_DIR,
        report_dir=CLEANING_REPORTS_DIR,
        remove=args.remove,
        dry_run=args.dry_run,
        near_scan=not args.no_near,
    )


if __name__ == "__main__":
    main()
