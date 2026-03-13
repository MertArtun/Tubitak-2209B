"""Tests for the data cleaning pipeline: corrupt detection, duplicates, splits."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from configs.config import MIN_FILE_SIZE_BYTES

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def data_tree(tmp_path: Path) -> Path:
    """Create a minimal train/val image tree for testing."""
    for split in ("train", "val"):
        for cls in ("negative", "neutral", "positive"):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            # Write one valid image per class per split
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{split}_{cls}_001.png"), img)
    return tmp_path


@pytest.fixture
def quarantine_dir(tmp_path: Path) -> Path:
    return tmp_path / "quarantine"


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "reports"
    d.mkdir()
    return d


# ── Corrupt image detection ──────────────────────────────────────────────────


class TestCorruptDetection:
    """Verify corrupt / too-small / low-variance detection."""

    def test_detects_unreadable_file(self, data_tree, quarantine_dir, report_dir):
        """A file with garbage bytes should be flagged as unreadable."""
        corrupt_path = data_tree / "train" / "negative" / "broken.png"
        # Must be >= MIN_FILE_SIZE_BYTES to reach the cv2.imread check
        corrupt_path.write_bytes(b"this is not an image at all " * 10)

        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(data_tree, quarantine_dir, report_dir)
        assert "train/negative/broken.png" in report["issues"]["unreadable"]

    def test_detects_too_small_file(self, data_tree, quarantine_dir, report_dir):
        """A file smaller than MIN_FILE_SIZE_BYTES should be flagged."""
        tiny_path = data_tree / "train" / "neutral" / "tiny.png"
        tiny_path.write_bytes(b"x" * (MIN_FILE_SIZE_BYTES - 1))

        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(data_tree, quarantine_dir, report_dir)
        assert "train/neutral/tiny.png" in report["issues"]["too_small_file"]

    def test_detects_too_small_image(self, data_tree, quarantine_dir, report_dir):
        """An image smaller than MIN_IMAGE_SIZE pixels should be flagged."""
        small_img = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
        small_path = data_tree / "train" / "positive" / "small.png"
        cv2.imwrite(str(small_path), small_img)

        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(data_tree, quarantine_dir, report_dir)
        assert "train/positive/small.png" in report["issues"]["too_small_image"]

    def test_detects_low_variance(self, data_tree, quarantine_dir, report_dir):
        """A flat (single color) image should be flagged."""
        flat_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        flat_path = data_tree / "train" / "negative" / "flat.png"
        cv2.imwrite(str(flat_path), flat_img)

        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(data_tree, quarantine_dir, report_dir)
        assert "train/negative/flat.png" in report["issues"]["low_variance"]

    def test_valid_image_not_flagged(self, data_tree, quarantine_dir, report_dir):
        """Valid images should not appear in any issue list."""
        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(data_tree, quarantine_dir, report_dir)
        # The fixture creates valid images; none should be flagged
        assert report["total_flagged"] == 0


# ── Duplicate detection ──────────────────────────────────────────────────────


class TestDuplicateDetection:
    """Verify exact and near-duplicate detection."""

    def test_detects_exact_duplicate(self, data_tree, quarantine_dir, report_dir):
        """Identical images in different locations should be flagged."""
        src = data_tree / "train" / "negative" / "train_negative_001.png"
        dst = data_tree / "train" / "neutral" / "dup_of_neg.png"
        shutil.copy2(str(src), str(dst))

        from scripts.detect_duplicates import detect_duplicates

        report = detect_duplicates(data_tree, quarantine_dir, report_dir)
        assert report["exact_duplicate_groups"] >= 1

    def test_different_images_not_flagged(self, data_tree, quarantine_dir, report_dir):
        """Different random images should not be exact duplicates."""
        from scripts.detect_duplicates import detect_duplicates

        report = detect_duplicates(data_tree, quarantine_dir, report_dir)
        # The fixture generates random images — should have 0 exact dups
        assert report["exact_duplicate_files"] == 0

    def test_cross_split_leak_detected(self, data_tree, quarantine_dir, report_dir):
        """Same image in train and val should appear in cross_split_leaks."""
        src = data_tree / "train" / "negative" / "train_negative_001.png"
        dst = data_tree / "val" / "negative" / "leaked_from_train.png"
        shutil.copy2(str(src), str(dst))

        from scripts.detect_duplicates import detect_duplicates

        report = detect_duplicates(data_tree, quarantine_dir, report_dir)
        assert len(report["cross_split_leaks"]) >= 1


# ── Quarantine action ────────────────────────────────────────────────────────


class TestQuarantineAction:
    """Verify quarantine moves files instead of deleting."""

    def test_quarantine_moves_not_deletes(self, data_tree, quarantine_dir, report_dir):
        """Flagged files should be moved to quarantine, not deleted."""
        corrupt_path = data_tree / "train" / "negative" / "broken.png"
        corrupt_path.write_bytes(b"corrupt data here")

        from scripts.detect_corrupt_images import detect_corrupt_images

        detect_corrupt_images(data_tree, quarantine_dir, report_dir, remove=False)

        # Original should be gone
        assert not corrupt_path.exists()
        # Should be in quarantine
        quarantine_files = list(quarantine_dir.rglob("broken.png"))
        assert len(quarantine_files) == 1

    def test_corrupt_dry_run_does_not_move(self, data_tree, quarantine_dir, report_dir):
        """Dry run should only report, not move files."""
        corrupt_path = data_tree / "train" / "negative" / "broken_dry.png"
        corrupt_path.write_bytes(b"not an image but large enough" * 20)

        from scripts.detect_corrupt_images import detect_corrupt_images

        report = detect_corrupt_images(
            data_tree,
            quarantine_dir,
            report_dir,
            remove=False,
            dry_run=True,
        )
        assert report["dry_run"] is True
        assert corrupt_path.exists()
        assert not any(p.name == "broken_dry.png" for p in quarantine_dir.rglob("*"))

    def test_duplicates_dry_run_does_not_move(self, data_tree, quarantine_dir, report_dir):
        """Dry run duplicate scan should not move files to quarantine."""
        src = data_tree / "train" / "negative" / "train_negative_001.png"
        dst = data_tree / "train" / "neutral" / "dup_dry.png"
        shutil.copy2(str(src), str(dst))

        from scripts.detect_duplicates import detect_duplicates

        report = detect_duplicates(
            data_tree,
            quarantine_dir,
            report_dir,
            remove=False,
            dry_run=True,
        )
        assert report["dry_run"] is True
        assert dst.exists()
        assert not any(p.name == "dup_dry.png" for p in quarantine_dir.rglob("*"))


class TestFERManifestApply:
    """Verify FER+ apply mode uses manifest and performs real moves."""

    def _write_minimal_fer_csvs(self, root: Path) -> tuple[Path, Path]:
        fer2013 = root / "fer2013.csv"
        ferplus = root / "fer2013new.csv"

        fer2013.write_text("emotion,pixels,Usage\n0,\"0 0 0 0\",Training\n")
        ferplus.write_text(
            "Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF\n"
            "Training,fer0000000.png,10,0,0,0,0,0,0,0,0,0\n"
        )
        return fer2013, ferplus

    def test_apply_requires_manifest(self, tmp_path):
        """Apply mode without manifest should return explicit error."""
        from scripts.fix_fer_labels import analyze_ferplus

        fer2013, ferplus = self._write_minimal_fer_csvs(tmp_path)
        report = analyze_ferplus(
            fer2013_csv=fer2013,
            ferplus_csv=ferplus,
            report_dir=tmp_path,
            apply=True,
            manifest_path=None,
            data_dir=tmp_path / "data",
        )
        assert "error" in report
        assert report["applied"] is False

    def test_apply_manifest_moves_file(self, tmp_path):
        """Apply mode should move files according to manifest rules."""
        from scripts.fix_fer_labels import analyze_ferplus

        fer2013, ferplus = self._write_minimal_fer_csvs(tmp_path)
        data_dir = tmp_path / "dataset"
        src = data_dir / "train" / "negative" / "sample.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"x")

        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "changes": [
                        {
                            "path": "train/negative/sample.png",
                            "new_class": "neutral",
                        }
                    ]
                }
            )
        )

        report = analyze_ferplus(
            fer2013_csv=fer2013,
            ferplus_csv=ferplus,
            report_dir=tmp_path,
            apply=True,
            manifest_path=manifest,
            data_dir=data_dir,
        )

        assert report["applied"] is True
        assert not src.exists()
        assert (data_dir / "train" / "neutral" / "sample.png").exists()


class TestOrchestrator:
    """Verify orchestrator step runner surfaces failures."""

    def test_run_step_returns_false_on_exception(self):
        from scripts.run_data_cleaning import _run_step

        def _boom() -> None:
            raise RuntimeError("boom")

        assert _run_step("failing-step", _boom) is False


# ── Stratified split ─────────────────────────────────────────────────────────


class TestStratifiedSplit:
    """Verify stratified split preserves proportions."""

    def _make_large_tree(self, tmp_path: Path) -> Path:
        """Create a tree with enough images for meaningful splits."""
        data_dir = tmp_path / "source"
        counts = {"negative": 100, "neutral": 40, "positive": 60}
        for split in ("train", "val"):
            for cls, count in counts.items():
                d = data_dir / split / cls
                d.mkdir(parents=True)
                for i in range(count):
                    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    cv2.imwrite(str(d / f"{split}_{cls}_{i:04d}.png"), img)
        return data_dir

    def test_all_splits_created(self, tmp_path):
        """train, val, and test directories should all exist after splitting."""
        from scripts.create_test_split import create_test_split

        source = self._make_large_tree(tmp_path)
        output = tmp_path / "output"
        create_test_split(data_dir=source, output_dir=output)

        assert (output / "train").exists()
        assert (output / "val").exists()
        assert (output / "test").exists()

    def test_no_images_lost(self, tmp_path):
        """Total image count should be preserved after split."""
        from scripts.create_test_split import create_test_split

        source = self._make_large_tree(tmp_path)
        output = tmp_path / "output"
        create_test_split(data_dir=source, output_dir=output)

        # Count source images
        source_count = sum(
            1 for _ in source.rglob("*.png")
        )
        # Count output images
        output_count = sum(
            1 for _ in output.rglob("*.png")
        )
        assert output_count == source_count

    def test_classes_preserved_in_all_splits(self, tmp_path):
        """All three classes should be present in each split."""
        from configs.config import CLASS_NAMES
        from scripts.create_test_split import create_test_split

        source = self._make_large_tree(tmp_path)
        output = tmp_path / "output"
        create_test_split(data_dir=source, output_dir=output)

        for split in ("train", "val", "test"):
            for cls in CLASS_NAMES:
                cls_dir = output / split / cls
                assert cls_dir.exists(), f"{split}/{cls} missing"
                assert len(list(cls_dir.iterdir())) > 0, f"{split}/{cls} empty"
