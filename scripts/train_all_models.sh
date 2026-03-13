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
    echo "Done: ${model} at $(date)"
    echo ""
done

echo ""
echo "========================================"
echo " ALL MODELS TRAINED"
echo " $(date)"
echo "========================================"
echo ""
echo "Results: ${OUTPUT_DIR}/results/"
echo "Models: ${OUTPUT_DIR}/models/"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
