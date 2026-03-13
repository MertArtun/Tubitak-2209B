# Colab Remote-SSH Training Pipeline Design

**Date:** 2026-03-13
**Status:** Approved

## Problem

4 model (efficientnet_b0, mobilenet_v3, efficientnet_b3, resnet50_cbam) eğitilmeli. Data lokal (7.9 GB), GPU Colab'da. Browser otomasyon yaklaşımı fragile çıktı.

## Chosen Approach: VS Code Remote-SSH → Colab VM

Colab Pro runtime'a cloudflared SSH tüneli ile bağlanıp, doğrudan `python -m src.models.train` çalıştırmak.

## Architecture

```
Local (M3 Mac)                     Colab VM (T4 GPU)
──────────────                     ──────────────────
VS Code Remote-SSH  ──cloudflared──▶  SSH server
                                      │
                                      ├── /content/drive/MyDrive/Tubitak-2209B/
                                      │   ├── data_split.tar.gz ✅ (uploaded)
                                      │   ├── checkpoints/  (persistent)
                                      │   └── results/      (persistent)
                                      │
                                      ├── /content/data/processed_merged_split/
                                      │   (extracted from tar, ephemeral)
                                      │
                                      └── git clone → repo
                                          └── python -m src.models.train
                                              --model X
                                              --data_dir /content/data/processed_merged_split
                                              --device cuda
                                              --output_dir /content/drive/MyDrive/Tubitak-2209B
```

## Components

### 1. SSH Setup Notebook (`notebooks/colab_ssh_setup.ipynb`)
Minimal notebook — sadece ortam kurulumu:
- Drive mount
- cloudflared install + SSH tunnel start
- Git clone + pip install
- tar extract (Drive → /content/data/)
- tmux install (eğitim arka planda devam etsin)

### 2. train.py Changes
- `--output_dir` CLI parametresi ekle (checkpoint + results dizinini override)
- Default: mevcut davranış (PROJECT_ROOT/models, PROJECT_ROOT/results)

### 3. Training Script (`scripts/train_all_models.sh`)
```bash
#!/bin/bash
MODELS="efficientnet_b0 mobilenet_v3 efficientnet_b3 resnet50_cbam"
DATA_DIR="/content/data/processed_merged_split"
OUTPUT_DIR="/content/drive/MyDrive/Tubitak-2209B"

for model in $MODELS; do
    echo "=== Training $model ==="
    python -m src.models.train \
        --model $model \
        --data_dir $DATA_DIR \
        --device cuda \
        --output_dir $OUTPUT_DIR
done
```

## Data Flow
1. `data_split.tar.gz` (Drive) → extract → `/content/data/processed_merged_split/`
2. Training reads from `/content/data/processed_merged_split/{train,val,test}/`
3. Checkpoints → `$OUTPUT_DIR/models/{model}_best.pth`
4. Results → `$OUTPUT_DIR/results/{model}_curves.png`, `{model}_history.json`, etc.
5. Drive sync keeps everything persistent

## Risk Management
| Risk | Mitigation |
|------|------------|
| SSH tunnel drops | tmux/screen — training continues in background |
| Colab runtime timeout | Checkpoints on Drive — resume from last best |
| Disk space on VM | Extract tar to /content (ephemeral, ~8 GB free) |

## Success Criteria (per model)
- Val accuracy ≥ 70%
- F1-macro ≥ 65%
- Checkpoint saved to Drive
- Training curves PNG generated
