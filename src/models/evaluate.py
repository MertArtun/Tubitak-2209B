"""Evaluation script for trained emotion classifiers.

Computes accuracy, F1, precision, recall, confusion matrix, ROC-AUC curves,
and inference time.

Usage:
    python -m src.models.evaluate --model efficientnet_b3 \
        --checkpoint models/efficientnet_b3_best.pth --data_dir data/processed
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.data.transforms import get_val_transforms
from src.models.architectures import create_model
from src.models.train import AlbumentationsImageFolder


def evaluate(
    model_name: str,
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    device_str: str = "cuda",
) -> dict:
    """Evaluate a trained model on the test set.

    Args:
        model_name: Model key from MODEL_CONFIGS.
        checkpoint_path: Path to the saved .pth checkpoint.
        data_dir: Root of the processed dataset.
        batch_size: Mini-batch size.
        device_str: 'cuda' or 'cpu'.

    Returns:
        Dictionary with all computed metrics.
    """
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Using device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    test_ds = AlbumentationsImageFolder(str(data_dir / "test"), transform=get_val_transforms())
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = create_model(model_name, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Collect predictions ───────────────────────────────────────────────
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()

            all_targets.extend(targets.tolist())
            all_preds.extend(preds)
            all_probs.append(probs)

    all_probs_np = np.concatenate(all_probs, axis=0)
    all_targets_np = np.array(all_targets)

    # ── Metrics ───────────────────────────────────────────────────────────
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_weighted = f1_score(all_targets, all_preds, average="weighted")
    precision_macro = precision_score(all_targets, all_preds, average="macro")
    recall_macro = recall_score(all_targets, all_preds, average="macro")

    print(f"\n{'='*50}")
    print(f"Evaluation Results - {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f}")
    print(f"F1 (weighted):     {f1_weighted:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")

    # ── Classification Report ─────────────────────────────────────────────
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    report_path = RESULTS_DIR / f"{model_name}_eval_report.txt"
    report_path.write_text(report)
    print(f"\nClassification Report:\n{report}")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} - Confusion Matrix (Test)")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"{model_name}_eval_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {cm_path}")

    # ── ROC-AUC Curves (One-vs-Rest) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        binary_targets = (all_targets_np == i).astype(int)
        if binary_targets.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_targets, all_probs_np[:, i])
        auc_val = roc_auc_score(binary_targets, all_probs_np[:, i])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = RESULTS_DIR / f"{model_name}_roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"ROC curves saved: {roc_path}")

    # ── Inference Time ────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"\nInference time: {avg_ms:.2f} +/- {std_ms:.2f} ms/frame (over 100 samples)")

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "inference_ms_mean": avg_ms,
        "inference_ms_std": std_ms,
    }

    import json
    metrics_path = RESULTS_DIR / f"{model_name}_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved: {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained emotion classifier.")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model name from MODEL_CONFIGS.")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "efficientnet_b3_best.pth"))
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
