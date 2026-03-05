# Model Training & Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a parametric Colab notebook with checkpoint/resume that trains 4 emotion models on the merged ~187K dataset, compares them, and exports the best to ONNX.

**Architecture:** Single parametric notebook designed for Google Colab Pro. Two instances run in parallel (Notebook A: B3→B0, Notebook B: MobileNetV3→ResNet50+CBAM). Google Drive stores checkpoints, results, and ONNX models. Checkpoint/resume mechanism handles Colab disconnections.

**Tech Stack:** PyTorch, timm, albumentations, ONNX, onnxruntime, Google Colab Pro, Google Drive

---

### Task 1: Create test split for processed_merged dataset

**Files:**
- Modify: `configs/config.py` (add PROCESSED_MERGED_DIR to config if not already there)
- Run: `scripts/create_test_split.py`

**Step 1: Verify processed_merged has no test split**

Run: `ls data/processed_merged/`
Expected: Only `train/` and `val/` directories

**Step 2: Run the existing create_test_split.py script**

Run: `python scripts/create_test_split.py --data_dir data/processed_merged --output_dir data/processed_merged_split`

Expected: Creates `data/processed_merged_split/` with train/val/test subdirectories. ~187K images split into ~80% train / ~10% val / ~10% test.

**Step 3: Verify the split**

Run: `ls data/processed_merged_split/` and check each split has negative/neutral/positive subdirs.

Expected:
- train: ~150K images
- val: ~18K images
- test: ~18K images

**Step 4: Commit**

```bash
git add scripts/create_test_split.py
git commit -m "chore: create stratified test split for merged dataset"
```

---

### Task 2: Write the parametric Colab training notebook

**Files:**
- Create: `notebooks/train_merged_parametric.ipynb`

This is the main deliverable. The notebook must:
1. Be parametric — model name set in a single cell
2. Support checkpoint/resume from Google Drive
3. Use folder-based dataset (not CSV)
4. Use 2-phase training (head-only → full fine-tune) with MixUp/CutMix in phase 2
5. Auto-export to ONNX after training
6. Save all results to Drive

**Step 1: Create the notebook with the following cells**

The notebook structure (18 cells):

**Cell 1 (markdown): Title & instructions**
```markdown
# Student Attention Detection - Parametric Model Training

## Usage
1. Set `MODEL_NAME` in Cell 3 to one of: `efficientnet_b3`, `efficientnet_b0`, `mobilenet_v3`, `resnet50_cbam`
2. Upload your `processed_merged_split/` dataset to Google Drive at `Tubitak-2209B/data/`
3. Run all cells
4. If disconnected, just re-run — training resumes from last checkpoint

## Parallel Training (Colab Pro)
- **Notebook A:** `MODEL_NAME = "efficientnet_b3"` → then change to `"efficientnet_b0"`
- **Notebook B:** `MODEL_NAME = "mobilenet_v3"` → then change to `"resnet50_cbam"`
```

**Cell 2 (code): Setup & Install**
```python
# Install required packages
!pip install timm albumentations onnx onnxruntime-gpu tqdm -q

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Cell 3 (code): Configuration — THE ONLY CELL TO EDIT**
```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  CHANGE THIS to train a different model                     ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = "efficientnet_b3"  # efficientnet_b3 | efficientnet_b0 | mobilenet_v3 | resnet50_cbam

# ── Paths (Google Drive) ──────────────────────────────────────
DRIVE_ROOT = "/content/drive/MyDrive/Tubitak-2209B"
DATA_DIR = f"{DRIVE_ROOT}/data/processed_merged_split"
CHECKPOINT_DIR = f"{DRIVE_ROOT}/checkpoints"
RESULTS_DIR = f"{DRIVE_ROOT}/results"
ONNX_DIR = f"{DRIVE_ROOT}/onnx_models"

# ── Training Hyperparameters ──────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 25
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
FOCAL_LOSS_GAMMA = 2.0

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_CUTMIX_PROB = 0.5

NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42

# ── Model Configs ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "efficientnet_b3": {"timm_name": "efficientnet_b3", "feature_dim": 1536, "hidden_dim": 512, "dropout": 0.3, "dropout2": 0.2},
    "efficientnet_b0": {"timm_name": "efficientnet_b0", "feature_dim": 1280, "hidden_dim": 256, "dropout": 0.3, "dropout2": 0.2},
    "mobilenet_v3": {"timm_name": "mobilenetv3_large_100", "feature_dim": 1280, "hidden_dim": 256, "dropout": 0.3, "dropout2": 0.2},
    "resnet50_cbam": {"timm_name": "resnet50", "feature_dim": 2048, "hidden_dim": 512, "dropout": 0.3, "dropout2": 0.2},
}

import os
for d in [CHECKPOINT_DIR, RESULTS_DIR, ONNX_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Training: {MODEL_NAME}")
print(f"Data: {DATA_DIR}")
print(f"Config: {MODEL_CONFIGS[MODEL_NAME]}")
```

**Cell 4 (code): Mount Drive & Verify Data**
```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os

data_path = Path(DATA_DIR)
for split in ["train", "val", "test"]:
    split_dir = data_path / split
    if split_dir.exists():
        counts = {}
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                counts[cls] = len(list(cls_dir.iterdir()))
            else:
                counts[cls] = 0
        total = sum(counts.values())
        print(f"{split}: {total} images — {counts}")
    else:
        print(f"WARNING: {split_dir} not found!")
```

**Cell 5 (code): Imports**
```python
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
```

**Cell 6 (code): Transforms**
```python
def get_train_transforms(phase=1):
    """Phase 1: light augmentation. Phase 2: strong augmentation."""
    if phase == 1:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.1)),
        ], p=0.3),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1),
            A.OpticalDistortion(distort_limit=0.05),
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(int(IMAGE_SIZE * 0.05), int(IMAGE_SIZE * 0.25)),
            hole_width_range=(int(IMAGE_SIZE * 0.05), int(IMAGE_SIZE * 0.25)),
            fill=0, p=0.3,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

print("Transforms defined.")
```

**Cell 7 (code): Dataset (AlbumentationsImageFolder)**
```python
class AlbumentationsImageFolder(ImageFolder):
    """ImageFolder with albumentations support."""
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.albu_transform = transform
        self.labels = [s[1] for s in self.samples]

    def __getitem__(self, index):
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

    def set_transform(self, transform):
        self.albu_transform = transform

def get_class_weights(dataset):
    """Compute inverse-frequency class weights."""
    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)
    num_classes = max(label_counts.keys()) + 1
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = label_counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)
    return weights

print("Dataset classes defined.")
```

**Cell 8 (code): Model Architectures (CBAM + 4 models + factory)**
```python
# ── CBAM ──────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

# ── Shared head ───────────────────────────────────────────────
def _build_head(feature_dim, hidden_dim, dropout, dropout2):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True), nn.Dropout(dropout2),
        nn.Linear(hidden_dim, NUM_CLASSES),
    )

# ── Models ────────────────────────────────────────────────────
class EfficientNetB3Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        cfg = MODEL_CONFIGS["efficientnet_b3"]
        self.backbone = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="")
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])
    def forward(self, x): return self.head(self.backbone(x))
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        cfg = MODEL_CONFIGS["efficientnet_b0"]
        self.backbone = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="")
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])
    def forward(self, x): return self.head(self.backbone(x))
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

class MobileNetV3Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        cfg = MODEL_CONFIGS["mobilenet_v3"]
        self.backbone = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="")
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])
    def forward(self, x): return self.head(self.backbone(x))
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

class ResNet50CBAMClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        cfg = MODEL_CONFIGS["resnet50_cbam"]
        self.backbone = timm.create_model(cfg["timm_name"], pretrained=pretrained, num_classes=0, global_pool="")
        self.cbam1, self.cbam2, self.cbam3, self.cbam4 = CBAM(256), CBAM(512), CBAM(1024), CBAM(2048)
        self.head = _build_head(cfg["feature_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["dropout2"])
    def _forward_backbone(self, x):
        x = self.backbone.maxpool(self.backbone.act1(self.backbone.bn1(self.backbone.conv1(x))))
        x = self.cbam1(self.backbone.layer1(x))
        x = self.cbam2(self.backbone.layer2(x))
        x = self.cbam3(self.backbone.layer3(x))
        x = self.cbam4(self.backbone.layer4(x))
        return x
    def forward(self, x): return self.head(self._forward_backbone(x))
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

MODEL_REGISTRY = {
    "efficientnet_b3": EfficientNetB3Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "mobilenet_v3": MobileNetV3Classifier,
    "resnet50_cbam": ResNet50CBAMClassifier,
}

def create_model(name, pretrained=True):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](pretrained=pretrained)

print("Models defined:", list(MODEL_REGISTRY.keys()))
```

**Cell 9 (code): Loss Functions (FocalLoss + SoftTargetCE)**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_LOSS_GAMMA, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_oh = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_oh).sum(dim=1)
        log_pt = (log_probs * targets_oh).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, targets):
        if targets.dim() == 1:
            return F.cross_entropy(logits, targets)
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

class MixUpCutMix:
    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images, targets):
        soft_targets = F.one_hot(targets, self.num_classes).float()
        if torch.rand(1).item() > self.prob:
            return images, soft_targets
        if torch.rand(1).item() < 0.5:
            return self._mixup(images, soft_targets)
        return self._cutmix(images, soft_targets)

    def _mixup(self, images, soft_targets):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item() if self.mixup_alpha > 0 else 1.0
        idx = torch.randperm(images.size(0), device=images.device)
        return lam * images + (1 - lam) * images[idx], lam * soft_targets + (1 - lam) * soft_targets[idx]

    def _cutmix(self, images, soft_targets):
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item() if self.cutmix_alpha > 0 else 1.0
        idx = torch.randperm(images.size(0), device=images.device)
        _, _, h, w = images.shape
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
        cx, cy = torch.randint(0, w, (1,)).item(), torch.randint(0, h, (1,)).item()
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(w, cx + cut_w // 2), min(h, cy + cut_h // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        actual_lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
        return mixed, actual_lam * soft_targets + (1 - actual_lam) * soft_targets[idx]

print("Loss functions and MixUp/CutMix defined.")
```

**Cell 10 (code): Checkpoint/Resume System**
```python
def get_checkpoint_path(model_name):
    return f"{CHECKPOINT_DIR}/{model_name}_checkpoint.pth"

def save_checkpoint(model, optimizer, scheduler, epoch, phase, best_val_acc, history, model_name):
    """Save full training state to Drive."""
    path = get_checkpoint_path(model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "phase": phase,
        "best_val_acc": best_val_acc,
        "history": history,
        "model_name": model_name,
    }, path)
    print(f"  Checkpoint saved: phase={phase}, epoch={epoch}, best_acc={best_val_acc:.4f}")

def load_checkpoint(model, model_name):
    """Load checkpoint if exists. Returns (optimizer_state, scheduler_state, epoch, phase, best_val_acc, history) or None."""
    path = get_checkpoint_path(model_name)
    if not os.path.exists(path):
        print(f"No checkpoint found for {model_name}. Starting fresh.")
        return None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint loaded: {model_name}, phase={ckpt['phase']}, epoch={ckpt['epoch']}, best_acc={ckpt['best_val_acc']:.4f}")
    return ckpt

def save_best_model(model, model_name):
    """Save the best model separately (for ONNX export later)."""
    path = f"{CHECKPOINT_DIR}/{model_name}_best.pth"
    torch.save({"model_state_dict": model.state_dict(), "model_name": model_name}, path)

print("Checkpoint system defined.")
```

**Cell 11 (code): Training Engine with Resume**
```python
def run_epoch(model, loader, criterion, optimizer, is_train=True, mixup_cutmix=None):
    """Run one epoch."""
    model.train() if is_train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    with ctx:
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            if is_train and mixup_cutmix is not None:
                images, targets = mixup_cutmix(images, targets)
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            if targets.dim() == 2:
                correct += (preds == targets.argmax(dim=1)).sum().item()
            else:
                correct += (preds == targets).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(model_name):
    """Full 2-phase training with checkpoint/resume."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*70}")

    # Data
    train_ds = AlbumentationsImageFolder(f"{DATA_DIR}/train", transform=get_train_transforms(phase=1))
    val_ds = AlbumentationsImageFolder(f"{DATA_DIR}/val", transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = create_model(model_name, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Check for checkpoint
    ckpt = load_checkpoint(model, model_name)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_phase = 1
    start_epoch = 1
    best_val_acc = 0.0

    if ckpt is not None:
        history = ckpt["history"]
        start_phase = ckpt["phase"]
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

    # ── Phase 1: Head-only ────────────────────────────────────
    if start_phase <= 1:
        print(f"\n--- Phase 1: Head-only ({PHASE1_EPOCHS} epochs, lr={PHASE1_LR}) ---")
        model.freeze_backbone()
        class_weights = get_class_weights(train_ds).to(device)
        criterion = FocalLoss(alpha=class_weights).to(device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

        if ckpt and ckpt["phase"] == 1:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        p1_start = start_epoch if start_phase == 1 else 1
        for epoch in range(p1_start, PHASE1_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, is_train=False)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"[P1] {epoch}/{PHASE1_EPOCHS} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | {elapsed:.0f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_best_model(model, model_name)
                print(f"  -> New best: {best_val_acc:.4f}")

            save_checkpoint(model, optimizer, None, epoch, 1, best_val_acc, history, model_name)

        # Mark phase 1 complete
        start_epoch = 1
        start_phase = 2

    # ── Phase 2: Full fine-tuning ─────────────────────────────
    print(f"\n--- Phase 2: Full fine-tune ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR}) ---")
    model.unfreeze_backbone()
    train_ds.set_transform(get_train_transforms(phase=2))

    criterion = SoftTargetCrossEntropy().to(device)
    mixup_cutmix = MixUpCutMix(num_classes=NUM_CLASSES, mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=MIXUP_CUTMIX_PROB)
    optimizer = AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    if ckpt and ckpt["phase"] == 2:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    p2_start = start_epoch if start_phase == 2 else 1
    epochs_no_improve = 0

    for epoch in range(p2_start, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True, mixup_cutmix=mixup_cutmix)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        g_epoch = PHASE1_EPOCHS + epoch
        print(f"[P2] {epoch}/{PHASE2_EPOCHS} (g:{g_epoch}) | Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_best_model(model, model_name)
            print(f"  -> New best: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1

        save_checkpoint(model, optimizer, scheduler, epoch, 2, best_val_acc, history, model_name)

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs.")
            break

    # Load best for return
    best_path = f"{CHECKPOINT_DIR}/{model_name}_best.pth"
    best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\nTraining complete: {model_name}, best val acc = {best_val_acc:.4f}")
    return model, history, best_val_acc

print("Training engine defined.")
```

**Cell 12 (code): Evaluation Function**
```python
def evaluate_model(model, model_name):
    """Full evaluation on test set."""
    test_ds = AlbumentationsImageFolder(f"{DATA_DIR}/test", transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model.eval()
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_targets.extend(targets.tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_probs.append(probs)

    all_probs_np = np.concatenate(all_probs)
    all_targets_np = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1m = f1_score(all_targets, all_preds, average="macro")
    f1w = f1_score(all_targets, all_preds, average="weighted")
    prec = precision_score(all_targets, all_preds, average="macro")
    rec = recall_score(all_targets, all_preds, average="macro")

    print(f"\n{'='*50}")
    print(f"TEST RESULTS — {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 macro:  {f1m:.4f}")
    print(f"F1 weight: {f1w:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    # Classification report
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    print(f"\n{report}")
    with open(f"{RESULTS_DIR}/{model_name}_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        binary = (all_targets_np == i).astype(int)
        if binary.sum() == 0: continue
        fpr, tpr, _ = roc_curve(binary, all_probs_np[:, i])
        auc_val = roc_auc_score(binary, all_probs_np[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{model_name} — ROC Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_roc.png", dpi=150)
    plt.show()
    plt.close()

    # Inference time
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(times)
    print(f"Inference: {avg_ms:.2f} ms/frame")

    # Training curves
    # (loaded from checkpoint history if available)

    total_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "model": model_name,
        "params": total_params,
        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "precision_macro": prec, "recall_macro": rec,
        "inference_ms": avg_ms,
    }
    with open(f"{RESULTS_DIR}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

print("Evaluation function defined.")
```

**Cell 13 (code): ONNX Export Function**
```python
def export_to_onnx(model_name):
    """Export best model to ONNX."""
    import onnx
    import onnxruntime as ort

    best_path = f"{CHECKPOINT_DIR}/{model_name}_best.pth"
    onnx_path = f"{ONNX_DIR}/{model_name}.onnx"

    model = create_model(model_name, pretrained=False).to("cpu")
    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    # Validate
    onnx.checker.check_model(onnx.load(onnx_path))
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    if np.allclose(pt_out, onnx_out, atol=1e-5):
        print("Validation PASSED")
    else:
        print(f"WARNING: max diff = {np.max(np.abs(pt_out - onnx_out)):.6e}")

    # ONNX Runtime benchmark
    inp_name = session.get_inputs()[0].name
    dummy_np = dummy.numpy()
    for _ in range(10): session.run(None, {inp_name: dummy_np})
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy_np})
        times.append((time.perf_counter() - t0) * 1000)
    print(f"ONNX RT inference: {np.mean(times):.2f} ± {np.std(times):.2f} ms")

    return onnx_path

print("ONNX export function defined.")
```

**Cell 14 (code): TRAIN!**
```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN TRAINING — just run this cell                         ║
# ╚══════════════════════════════════════════════════════════════╝

model, history, best_val_acc = train_model(MODEL_NAME)
print(f"\n✓ {MODEL_NAME} training complete. Best val acc: {best_val_acc:.4f}")
```

**Cell 15 (code): EVALUATE**
```python
metrics = evaluate_model(model, MODEL_NAME)
print(f"\n✓ {MODEL_NAME} evaluation complete.")
print(json.dumps(metrics, indent=2))
```

**Cell 16 (code): EXPORT TO ONNX**
```python
onnx_path = export_to_onnx(MODEL_NAME)
print(f"\n✓ ONNX model saved: {onnx_path}")
```

**Cell 17 (code): Plot training curves**
```python
# Plot training history
epochs_range = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_loss"], "b-o", ms=3, label="Train")
axes[0].plot(epochs_range, history["val_loss"], "r-o", ms=3, label="Val")
axes[0].axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Phase boundary")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title(f"{MODEL_NAME} — Loss"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_range, history["train_acc"], "b-o", ms=3, label="Train")
axes[1].plot(epochs_range, history["val_acc"], "r-o", ms=3, label="Val")
axes[1].axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Phase boundary")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"{MODEL_NAME} — Accuracy"); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{MODEL_NAME}_curves.png", dpi=150)
plt.show()
```

**Cell 18 (markdown): Next steps**
```markdown
# Done!

## What was saved to Google Drive:
- `checkpoints/{MODEL_NAME}_best.pth` — Best PyTorch model
- `checkpoints/{MODEL_NAME}_checkpoint.pth` — Full training state (for resume)
- `onnx_models/{MODEL_NAME}.onnx` — ONNX model for deployment
- `results/{MODEL_NAME}_metrics.json` — Evaluation metrics
- `results/{MODEL_NAME}_report.txt` — Classification report
- `results/{MODEL_NAME}_confusion_matrix.png` — Confusion matrix
- `results/{MODEL_NAME}_roc.png` — ROC curves
- `results/{MODEL_NAME}_curves.png` — Training curves

## Next: Change MODEL_NAME and re-run
Change `MODEL_NAME` in Cell 3 and re-run all cells to train the next model.
```

**Step 2: Verify the notebook runs locally (dry test)**

Run: Open the notebook and verify all cells parse without syntax errors.

**Step 3: Commit**

```bash
git add notebooks/train_merged_parametric.ipynb
git commit -m "feat: add parametric Colab training notebook with checkpoint/resume"
```

---

### Task 3: Write the comparison notebook

**Files:**
- Create: `notebooks/compare_models.ipynb`

A small notebook that loads all 4 model metrics from Drive and generates the comparison table + charts.

**Step 1: Create the comparison notebook**

5 cells:
1. Mount Drive, set paths
2. Load all `{model}_metrics.json` files from results/
3. Create comparison DataFrame, display styled table
4. Bar chart comparison (accuracy, F1, inference time)
5. Weighted scoring to pick the best model (F1-macro 40%, inference 25%, accuracy 20%, size 15%)

**Step 2: Commit**

```bash
git add notebooks/compare_models.ipynb
git commit -m "feat: add model comparison notebook"
```

---

### Task 4: Upload dataset to Google Drive

**Files:** None (manual step)

**Step 1: Document the upload process**

The user needs to upload `data/processed_merged_split/` to Google Drive at `Tubitak-2209B/data/processed_merged_split/`.

Options:
- Zip and upload: `cd data && zip -r processed_merged_split.zip processed_merged_split/`
- Use `rclone` or Google Drive desktop app
- Upload via Colab: mount Drive + `!cp -r` from local

This is a manual step that the user performs.

---

### Task 5: Run training (manual Colab step)

**Step 1: Open Colab Pro, create 2 sessions**
- Session A: Set `MODEL_NAME = "efficientnet_b3"`, run all
- Session B: Set `MODEL_NAME = "mobilenet_v3"`, run all

**Step 2: After first model finishes in each session**
- Session A: Change to `MODEL_NAME = "efficientnet_b0"`, run all
- Session B: Change to `MODEL_NAME = "resnet50_cbam"`, run all

**Step 3: Run compare_models.ipynb to generate final comparison**

---

### Task 6: Copy best ONNX model to local project

**Files:**
- Copy: `models/emotion_model.onnx` (from Drive)

**Step 1: Download the best ONNX model from Drive to `models/`**

```bash
# After training, copy from Drive or download
cp ~/Google\ Drive/Tubitak-2209B/onnx_models/best_model.onnx models/emotion_model.onnx
```

**Step 2: Verify with existing export_onnx.py validation logic**

```bash
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/emotion_model.onnx')
print('Input:', session.get_inputs()[0].name, session.get_inputs()[0].shape)
print('Output:', session.get_outputs()[0].name, session.get_outputs()[0].shape)
print('ONNX model loaded successfully!')
"
```

**Step 3: Commit**

```bash
git add models/emotion_model.onnx
git commit -m "feat: add trained emotion classification ONNX model"
```
