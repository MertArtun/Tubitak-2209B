# Model Training & Comparison Design

**Date:** 2026-03-05
**Status:** Approved

## Context

TÜBİTAK 2209-B Student Attention Detection System. Need to train emotion classification models (3-class: positive/neutral/negative) for online attention tracking. Deadline: April 2026.

## Decisions

| Decision | Choice |
|----------|--------|
| Dataset | Merged all 10 datasets (~187K images) |
| GPU | Google Colab Pro (2 parallel sessions) |
| Priority | Online (Zoom) scenario first |
| Models | Train all 4, compare, pick best |

## Architecture

### Training Setup

Two parallel Colab Pro notebooks, each training 2 models sequentially:

- **Notebook A (Session 1):** EfficientNet-B3 → EfficientNet-B0
- **Notebook B (Session 2):** MobileNetV3-Large → ResNet50+CBAM

Single parametric notebook design — model selected via parameter.

### Shared Storage (Google Drive)

```
/content/drive/MyDrive/tubitak/
├── checkpoints/          # Per-model best + latest checkpoints
├── onnx_models/          # Exported ONNX models
├── results/              # Evaluation artifacts
└── data/processed_merged/ # Train/val/test splits
```

## Training Pipeline

### Per-Model Flow

```
1. Setup
   ├── Drive mount
   ├── Code + dataset load
   └── Check for existing checkpoint (resume support)

2. Phase 1 — Head Only (5 epochs)
   ├── Backbone frozen
   ├── Light augmentation (HFlip, Rotate10, light ColorJitter)
   ├── FocalLoss (class-weighted, gamma=2.0)
   ├── AdamW, lr=1e-3
   └── Checkpoint after each epoch

3. Phase 2 — Full Fine-tune (25 epochs)
   ├── All layers unfrozen
   ├── Strong augmentation + MixUp/CutMix
   ├── SoftTargetCrossEntropy
   ├── AdamW, lr=1e-4, CosineAnnealingLR
   ├── Early stopping (patience=7, val accuracy)
   └── Checkpoint after each epoch

4. Evaluation (on test set)
   ├── Accuracy, F1-macro, F1-weighted, Precision, Recall
   ├── Confusion matrix PNG
   ├── ROC-AUC curve PNG
   └── Inference time (100 samples average)

5. Export
   ├── PyTorch → ONNX (opset 17, dynamic batch)
   ├── Validation: PyTorch vs ONNX output diff < 1e-5
   └── Results → JSON
```

### Checkpoint/Resume Mechanism

```python
checkpoint = {
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": int,
    "phase": int,  # 1 or 2
    "best_accuracy": float,
    "training_history": dict
}
```

On notebook start: check for checkpoint → resume if exists, else start fresh.

## Model Comparison

### Selection Criteria (Weighted)

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| F1-macro | 40% | Most reliable for imbalanced classes |
| Inference time | 25% | Online use requires fast CPU inference |
| Accuracy | 20% | General correctness |
| Model size | 15% | Deployment ease |

### Output Table

```
┌────────────────────┬──────────┬──────────┬───────────┬──────────────┐
│ Model              │ Accuracy │ F1-macro │ Param (M) │ Inference ms │
├────────────────────┼──────────┼──────────┼───────────┼──────────────┤
│ EfficientNet-B3    │          │          │   ~12M    │              │
│ EfficientNet-B0    │          │          │    ~5M    │              │
│ MobileNetV3-Large  │          │          │    ~5M    │              │
│ ResNet50+CBAM      │          │          │   ~25M    │              │
└────────────────────┴──────────┴──────────┴───────────┴──────────────┘
```

## Online Attention Tracking (Post-Training)

```
Webcam (2 FPS)
  → Face Detection (InsightFace)
  → Emotion Classification (ONNX, best model)
  → Attention Scoring (positive=0.8, neutral=0.5, negative=0.2)
  → Sliding window trend + anomaly detection
  → Dashboard (real-time display)
```

## Pre-requisites

1. **Create test split** for `processed_merged/` (currently only train/val exist)
2. Upload dataset to Google Drive

## Estimated Timeline

| Step | Task | Duration |
|------|------|----------|
| 1 | Create test split | 5 min |
| 2 | Write parametric Colab notebook | ~1 hour (we write this) |
| 3 | Notebook A: B3 → B0 | ~4-6 hours |
| 4 | Notebook B: MobileNetV3 → ResNet50+CBAM | ~4-6 hours (parallel) |
| 5 | Comparison report | Automatic |
| 6 | Download best ONNX model locally | 5 min |
| 7 | Test with existing API + dashboard | 30 min |

**Total:** Notebook prep + ~5-6 hours training + testing
