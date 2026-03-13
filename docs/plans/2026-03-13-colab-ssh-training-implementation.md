# Colab Remote-SSH Training Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable training 4 models on Colab via SSH with checkpoint/resume, Drive persistence, and automated sequential execution.

**Architecture:** VS Code Remote-SSH → cloudflared tunnel → Colab VM (T4 GPU). Training via `python -m src.models.train` with `--output_dir` for Drive paths and `--resume` for crash recovery. Results persist on Drive, downloaded locally after training.

**Tech Stack:** PyTorch, timm, cloudflared, tmux, Google Drive FUSE mount

---

### Task 1: Add checkpoint/resume support to train.py

**Files:**
- Modify: `src/models/train.py:215-442`
- Test: `tests/test_models.py`

**Step 1: Write failing tests for new CLI args and checkpoint utilities**

Add to `tests/test_models.py`:

```python
class TestTrainCLI:
    """Test train.py CLI argument parsing."""

    def test_output_dir_arg(self):
        """--output_dir should override model and result dirs."""
        from src.models.train import _parse_args

        args = _parse_args(["--output_dir", "/tmp/test_out"])
        assert args.output_dir == "/tmp/test_out"

    def test_resume_flag(self):
        """--resume should default to True."""
        from src.models.train import _parse_args

        args = _parse_args([])
        assert args.resume is True

    def test_checkpoint_dir_arg(self):
        """--checkpoint_dir should override checkpoint location."""
        from src.models.train import _parse_args

        args = _parse_args(["--checkpoint_dir", "/tmp/ckpts"])
        assert args.checkpoint_dir == "/tmp/ckpts"


class TestCheckpointUtils:
    """Test checkpoint save/load round-trip."""

    def test_save_load_checkpoint_roundtrip(self, tmp_path):
        """save_training_state → load_training_state preserves all fields."""
        from src.models.train import load_training_state, save_training_state

        state = {
            "epoch": 3,
            "phase": 2,
            "best_val_acc": 0.75,
            "history": {"train_loss": [0.5, 0.4, 0.3]},
        }
        path = tmp_path / "ckpt.pth"
        save_training_state(state, path)
        loaded = load_training_state(path)

        assert loaded["epoch"] == 3
        assert loaded["phase"] == 2
        assert loaded["best_val_acc"] == 0.75
        assert loaded["history"]["train_loss"] == [0.5, 0.4, 0.3]

    def test_load_nonexistent_returns_none(self, tmp_path):
        """load_training_state returns None if file missing."""
        from src.models.train import load_training_state

        result = load_training_state(tmp_path / "nonexistent.pth")
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py::TestTrainCLI -v && pytest tests/test_models.py::TestCheckpointUtils -v`
Expected: FAIL — `_parse_args`, `save_training_state`, `load_training_state` not defined

**Step 3: Extract `_parse_args` from `main()`**

In `src/models/train.py`, refactor `main()` to extract argument parsing:

```python
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train emotion classifier.")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model name from MODEL_CONFIGS.")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR), help="Path to processed dataset.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_focal", action="store_true", help="Use weighted CE instead of Focal Loss.")
    parser.add_argument("--no_mixup_cutmix", action="store_true", help="Disable MixUp/CutMix in phase 2.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir for models/ and results/.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint dir (for resume state).")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore existing checkpoints.")
    return parser.parse_args(argv)
```

Note: `--resume` is True by default (via `--no_resume` flag). This way, if Colab disconnects and the user re-runs, it automatically resumes.

**Step 4: Add checkpoint utility functions**

In `src/models/train.py`, add before the `train()` function:

```python
def save_training_state(state: dict, path: Path) -> None:
    """Save full training state to disk."""
    torch.save(state, path)


def load_training_state(path: Path) -> dict | None:
    """Load training state from disk. Returns None if not found."""
    if not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py::TestTrainCLI tests/test_models.py::TestCheckpointUtils -v`
Expected: PASS

**Step 6: Add resume logic to `train()` function**

Modify `train()` signature to accept new params:

```python
def train(
    model_name: str,
    data_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    device_str: str = "cuda",
    focal_loss: bool = True,
    use_mixup_cutmix: bool = True,
    output_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> Path:
```

Key changes inside `train()`:

1. Resolve output directories:
```python
models_dir = Path(output_dir) / "models" if output_dir else MODELS_DIR
results_dir = Path(output_dir) / "results" if output_dir else RESULTS_DIR
ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else models_dir
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = models_dir / f"{model_name}_best.pth"
resume_path = ckpt_dir / f"{model_name}_training_state.pth"
```

2. Load resume state before training loop:
```python
start_phase = 1
start_epoch = 1
best_val_acc = 0.0
epochs_no_improve = 0

if resume:
    state = load_training_state(resume_path)
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        history = state["history"]
        start_phase = state["phase"]
        start_epoch = state["epoch"] + 1
        best_val_acc = state["best_val_acc"]
        print(f"Resumed from phase {start_phase}, epoch {start_epoch - 1}, best acc {best_val_acc:.4f}")
```

3. Save full state at end of each epoch (both phases):
```python
save_training_state({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "epoch": epoch,
    "phase": current_phase,
    "best_val_acc": best_val_acc,
    "history": history,
    "model_name": model_name,
}, resume_path)
```

4. Skip completed phases/epochs on resume:
```python
# Phase 1
if start_phase <= 1:
    # ... phase 1 loop ...
    for epoch in range(p1_start, PHASE1_EPOCHS + 1):
        # training ...

# Phase 2
p2_start = start_epoch if start_phase == 2 else 1
for epoch in range(p2_start, PHASE2_EPOCHS + 1):
    # training ...
```

5. Restore optimizer/scheduler state on resume:
```python
if resume and state is not None and state["phase"] == current_phase:
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
```

6. Update `main()` to use `_parse_args`:
```python
def main() -> None:
    args = _parse_args()
    train(
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        focal_loss=not args.no_focal,
        use_mixup_cutmix=not args.no_mixup_cutmix,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )
```

7. Replace hardcoded `RESULTS_DIR` and `MODELS_DIR` references with local variables `results_dir` and `models_dir`.

**Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All existing + new tests PASS

**Step 8: Run linter**

Run: `ruff check src/models/train.py tests/test_models.py`
Expected: Clean

**Step 9: Commit**

```bash
git add src/models/train.py tests/test_models.py
git commit -m "feat: add checkpoint/resume and output_dir to train.py"
```

---

### Task 2: Create SSH setup notebook

**Files:**
- Create: `notebooks/colab_ssh_setup.ipynb`

**Step 1: Create the notebook**

This is a minimal Colab notebook (6 cells):

**Cell 1 (markdown):** Instructions
```markdown
# Colab SSH Setup
1. "Runtime → Change runtime type → T4 GPU"
2. Run all cells
3. Copy the SSH command and connect from VS Code
```

**Cell 2:** Mount Drive + extract data
```python
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

TAR_PATH = "/content/drive/MyDrive/Tubitak-2209B/data_split.tar.gz"
LOCAL_DATA = "/content/data/processed_merged_split"

if not os.path.exists(LOCAL_DATA):
    print("Extracting data (~5-10 min)...")
    os.makedirs("/content/data", exist_ok=True)
    subprocess.run(["tar", "-xzf", TAR_PATH, "-C", "/content/data"], check=True)
    print("Done!")
else:
    print(f"Data already exists: {LOCAL_DATA}")

# Verify
for split in ["train", "val", "test"]:
    split_dir = os.path.join(LOCAL_DATA, split)
    if os.path.exists(split_dir):
        total = sum(len(os.listdir(os.path.join(split_dir, c))) for c in os.listdir(split_dir))
        print(f"  {split}: {total} images")
```

**Cell 3:** Install cloudflared + start SSH
```python
# Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

# Set password for SSH
import subprocess
subprocess.run(["bash", "-c", "echo 'root:tubitak2209' | chpasswd"], check=True)

# Configure SSH
!apt-get -qq install openssh-server > /dev/null 2>&1
!echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
!echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
!service ssh start

# Start tunnel (runs in background)
import threading, re, time

tunnel_url = None

def run_tunnel():
    global tunnel_url
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "ssh://localhost:22", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line.strip())
        match = re.search(r'https://(.+\.trycloudflare\.com)', line)
        if match:
            tunnel_url = match.group(1)

t = threading.Thread(target=run_tunnel, daemon=True)
t.start()

# Wait for tunnel
for _ in range(30):
    if tunnel_url:
        break
    time.sleep(1)

if tunnel_url:
    print(f"\n{'='*60}")
    print(f"SSH TUNNEL READY!")
    print(f"{'='*60}")
    print(f"\nAdd to ~/.ssh/config:")
    print(f"""
Host colab
    HostName {tunnel_url}
    User root
    ProxyCommand cloudflared access ssh --hostname %h
""")
    print(f"Then connect: ssh colab")
    print(f"Password: tubitak2209")
else:
    print("ERROR: Tunnel failed to start")
```

**Cell 4:** Clone repo + install deps
```python
import os
os.chdir("/content")

!git clone https://github.com/halitartun/Tubitak-2209B.git 2>/dev/null || (cd Tubitak-2209B && git pull)
!cd Tubitak-2209B && pip install -r requirements-train.txt -q

# Install tmux for persistent sessions
!apt-get -qq install tmux > /dev/null 2>&1

print("\nSetup complete!")
print("After SSH, run:")
print("  cd /content/Tubitak-2209B")
print("  tmux new -s train")
print("  bash scripts/train_all_models.sh")
```

**Cell 5:** Keep-alive (prevents Colab timeout)
```python
import time
from IPython.display import display, Javascript

# Auto-click to prevent idle timeout
display(Javascript('''
function keepAlive() {
    google.colab.kernel.invokeFunction('check', [], {});
}
setInterval(keepAlive, 60000);
'''))

# Also keep the cell running
print("Keep-alive active. Don't close this tab.")
while True:
    time.sleep(300)
    print(f"Still alive... {time.strftime('%H:%M:%S')}")
```

**Step 2: Commit**

```bash
git add notebooks/colab_ssh_setup.ipynb
git commit -m "feat: add Colab SSH setup notebook for remote training"
```

---

### Task 3: Create training shell script

**Files:**
- Create: `scripts/train_all_models.sh`

**Step 1: Create the script**

```bash
#!/bin/bash
# Train all 4 models sequentially on Colab GPU.
# Run inside tmux so training survives SSH disconnects:
#   tmux new -s train
#   bash scripts/train_all_models.sh

set -e

MODELS="efficientnet_b0 mobilenet_v3 efficientnet_b3 resnet50_cbam"
DATA_DIR="/content/data/processed_merged_split"
OUTPUT_DIR="/content/drive/MyDrive/Tubitak-2209B"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

echo "========================================"
echo " Training 4 models sequentially"
echo " Data: ${DATA_DIR}"
echo " Output: ${OUTPUT_DIR}"
echo " Checkpoints: ${CHECKPOINT_DIR}"
echo "========================================"

for model in $MODELS; do
    echo ""
    echo "========================================"
    echo " Starting: ${model}"
    echo " $(date)"
    echo "========================================"

    python -m src.models.train \
        --model "$model" \
        --data_dir "$DATA_DIR" \
        --device cuda \
        --batch_size 32 \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"

    echo ""
    echo "✓ ${model} complete at $(date)"
    echo ""
done

echo ""
echo "========================================"
echo " ALL MODELS TRAINED"
echo " $(date)"
echo "========================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/results/"
echo "Models saved to: ${OUTPUT_DIR}/models/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
```

**Step 2: Make executable**

Run: `chmod +x scripts/train_all_models.sh`

**Step 3: Commit**

```bash
git add scripts/train_all_models.sh
git commit -m "feat: add sequential training script for Colab SSH"
```

---

### Task 4: Create results download script

**Files:**
- Create: `scripts/download_results.sh`

**Step 1: Create the script**

This runs locally after training to download results from Drive (via gdown or manual copy).

```bash
#!/bin/bash
# Download training results from Google Drive to local project.
# Prerequisites: pip install gdown
#
# Usage: bash scripts/download_results.sh <path-to-drive-folder>
# Example: bash scripts/download_results.sh ~/Google\ Drive/My\ Drive/Tubitak-2209B

set -e

DRIVE_DIR="${1:?Usage: bash scripts/download_results.sh <drive-folder-path>}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Copying from: ${DRIVE_DIR}"
echo "Copying to:   ${PROJECT_ROOT}"

# Models (best checkpoints)
mkdir -p "${PROJECT_ROOT}/models"
for f in "${DRIVE_DIR}"/models/*_best.pth; do
    [ -f "$f" ] && cp -v "$f" "${PROJECT_ROOT}/models/"
done

# Results
mkdir -p "${PROJECT_ROOT}/results"
cp -rv "${DRIVE_DIR}"/results/* "${PROJECT_ROOT}/results/" 2>/dev/null || echo "No results to copy"

# ONNX models
mkdir -p "${PROJECT_ROOT}/models/onnx"
for f in "${DRIVE_DIR}"/onnx_models/*.onnx; do
    [ -f "$f" ] && cp -v "$f" "${PROJECT_ROOT}/models/onnx/"
done

echo ""
echo "Done! Files copied:"
ls -la "${PROJECT_ROOT}/models/"*_best.pth 2>/dev/null || echo "  No .pth files"
ls -la "${PROJECT_ROOT}/models/onnx/"*.onnx 2>/dev/null || echo "  No .onnx files"
ls -la "${PROJECT_ROOT}/results/"*.json 2>/dev/null || echo "  No result JSONs"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/download_results.sh
git add scripts/download_results.sh
git commit -m "feat: add results download script"
```

---

### Task 5: Push to GitHub and run on Colab

**Step 1: Push all changes**

```bash
git push origin main
```

**Step 2: Open `notebooks/colab_ssh_setup.ipynb` in Colab**

User actions:
1. Go to Colab → File → Open Notebook → GitHub tab → paste repo URL
2. Select `notebooks/colab_ssh_setup.ipynb`
3. Runtime → Change runtime type → T4 GPU → Save
4. Run all cells
5. Copy SSH config from output, add to `~/.ssh/config`
6. `ssh colab` from terminal
7. In SSH session:
   ```bash
   cd /content/Tubitak-2209B
   tmux new -s train
   bash scripts/train_all_models.sh
   ```
8. Detach from tmux: `Ctrl+B, D` — training continues even if SSH disconnects
9. Reconnect anytime: `ssh colab` → `tmux attach -t train`

**Step 3: Monitor training**

From SSH:
```bash
tmux attach -t train
# Watch output in real-time
```

Or check Drive:
```bash
ls /content/drive/MyDrive/Tubitak-2209B/checkpoints/
ls /content/drive/MyDrive/Tubitak-2209B/results/
```

---

### Task 6: Download results and verify locally

**Step 1: After all 4 models finish, download from Drive**

```bash
bash scripts/download_results.sh ~/path/to/drive/Tubitak-2209B
```

**Step 2: Verify ONNX models locally**

```bash
python -c "
import onnxruntime as ort, numpy as np
for m in ['efficientnet_b0', 'mobilenet_v3', 'efficientnet_b3', 'resnet50_cbam']:
    path = f'models/onnx/{m}.onnx'
    try:
        s = ort.InferenceSession(path)
        o = s.run(None, {s.get_inputs()[0].name: np.random.randn(1,3,224,224).astype(np.float32)})[0]
        print(f'{m}: shape={o.shape} ✓' if o.shape == (1,3) else f'{m}: WRONG shape {o.shape}')
    except Exception as e:
        print(f'{m}: {e}')
"
```

**Step 3: Run model comparison**

Use `notebooks/compare_models.ipynb` in Colab or locally to compare the 4 models.

**Step 4: Commit trained models**

```bash
git add models/ results/
git commit -m "feat: add trained emotion models and evaluation results"
```

---

## Execution Order & Dependencies

```
Task 1 (train.py changes) ──┐
Task 2 (SSH notebook)   ─────┼── Task 5 (push + run on Colab)
Task 3 (train script)   ─────┤       │
Task 4 (download script) ────┘       ▼
                              Task 6 (download + verify)
```

Tasks 1-4 are independent and can be parallelized.
Task 5 depends on all of 1-4.
Task 6 depends on Task 5 (training must complete on Colab).
