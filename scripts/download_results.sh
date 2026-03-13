#!/bin/bash
# Download training results from Google Drive to local project.
# Usage: bash scripts/download_results.sh <path-to-drive-folder>

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
echo "Done! Copied files:"
ls -la "${PROJECT_ROOT}/models/"*_best.pth 2>/dev/null || echo "  No .pth files"
ls -la "${PROJECT_ROOT}/models/onnx/"*.onnx 2>/dev/null || echo "  No .onnx files"
ls -la "${PROJECT_ROOT}/results/"*.json 2>/dev/null || echo "  No result JSONs"
