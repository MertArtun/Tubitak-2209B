"""Two-phase training pipeline for emotion classification.

Phase 1: Freeze backbone, train classifier head only (warm-up).
Phase 2: Unfreeze all layers, fine-tune with cosine annealing LR.

Usage:
    python -m src.models.train --model efficientnet_b3 --data_dir data/processed \
        --batch_size 32 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from configs.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    MODELS_DIR,
    NUM_WORKERS,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from src.data.sampler import get_class_weights
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from src.models.losses import FocalLoss


# ---------------------------------------------------------------------------
# Dataset wrapper (albumentations compat with ImageFolder)
# ---------------------------------------------------------------------------

class AlbumentationsImageFolder(ImageFolder):
    """ImageFolder subclass that applies albumentations transforms."""

    def __init__(self, root: str, transform: object = None, **kwargs) -> None:  # noqa: ANN401
        super().__init__(root, **kwargs)
        self.albu_transform = transform
        self.labels: list[int] = [s[1] for s in self.samples]

    def __getitem__(self, index: int):  # noqa: ANN204
        path, target = self.samples[index]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        if self.albu_transform is not None:
            augmented = self.albu_transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        return image_tensor, target


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """Run one epoch of training or validation.

    Returns:
        (average_loss, accuracy)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _save_curves(
    history: dict[str, list[float]],
    save_dir: Path,
    model_name: str,
) -> None:
    """Save loss and accuracy curves as PNG."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_acc"], label="Train")
    axes[1].plot(epochs_range, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_curves.png", dpi=150)
    plt.close(fig)


def _save_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    model_name: str,
) -> None:
    """Generate and save a confusion matrix plot."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Classification report
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    report_path = save_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(report)
    print(f"\nClassification Report:\n{report}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    data_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    device_str: str = "cuda",
    focal_loss: bool = True,
) -> Path:
    """Run the full two-phase training pipeline.

    Args:
        model_name: Model key from MODEL_CONFIGS.
        data_dir: Root of the processed dataset (must contain train/val/test dirs).
        batch_size: Mini-batch size.
        device_str: 'cuda' or 'cpu'.
        focal_loss: If True, use FocalLoss; otherwise weighted CrossEntropy.

    Returns:
        Path to the best model checkpoint.
    """
    data_dir = Path(data_dir)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Using device: {device}")

    # ── Data loaders ──────────────────────────────────────────────────────
    train_ds = AlbumentationsImageFolder(str(data_dir / "train"), transform=get_train_transforms())
    val_ds = AlbumentationsImageFolder(str(data_dir / "val"), transform=get_val_transforms())
    test_ds = AlbumentationsImageFolder(str(data_dir / "test"), transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = create_model(model_name, pretrained=True).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights = get_class_weights(train_ds).to(device)
    if focal_loss:
        criterion = FocalLoss(alpha=class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # ── Output dirs ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{model_name}_best.pth"

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    epochs_no_improve = 0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    # ── Phase 1: Head-only ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Training head only for {PHASE1_EPOCHS} epochs (lr={PHASE1_LR})")
    print(f"{'='*60}")
    model.freeze_backbone()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device, is_train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Phase1] Epoch {epoch}/{PHASE1_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "model_name": model_name, "epoch": epoch}, checkpoint_path)
            print(f"  -> New best val acc: {best_val_acc:.4f}, checkpoint saved.")

    # ── Phase 2: Full fine-tuning ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tuning all layers for {PHASE2_EPOCHS} epochs (lr={PHASE2_LR})")
    print(f"{'='*60}")
    model.unfreeze_backbone()
    optimizer = AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        global_epoch = PHASE1_EPOCHS + epoch
        print(
            f"[Phase2] Epoch {epoch}/{PHASE2_EPOCHS} (global {global_epoch}/{total_epochs}) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": model_name, "epoch": global_epoch}, checkpoint_path)
            print(f"  -> New best val acc: {best_val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    # ── Post-training ─────────────────────────────────────────────────────
    # Load best checkpoint for final evaluation
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    _save_curves(history, RESULTS_DIR, model_name)
    _save_confusion_matrix(model, test_loader, device, RESULTS_DIR, model_name)

    # Save history as JSON
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Curves:     {RESULTS_DIR / f'{model_name}_curves.png'}")
    return checkpoint_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion classifier.")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model name from MODEL_CONFIGS.")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR), help="Path to processed dataset.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_focal", action="store_true", help="Use weighted CE instead of Focal Loss.")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        focal_loss=not args.no_focal,
    )


if __name__ == "__main__":
    main()
