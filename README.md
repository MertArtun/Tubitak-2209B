# Student Attention Detection System

**TÜBİTAK 2209-B Research Project**

## Overview

A two-phase student attention detection system that combines emotion analysis with face vectorization to monitor and assess student engagement in educational settings.

- **Phase 1:** 3-class emotion classification (Positive / Negative / Neutral) for online class environments.
- **Phase 2:** Head pose estimation and gaze direction analysis for face-to-face classroom settings.

The system uses face recognition to enable per-student tracking, providing individualized attention metrics over time.

## Architecture

| Component          | Technology                        |
|--------------------|-----------------------------------|
| Training           | PyTorch + timm (Google Colab GPU) |
| Inference          | ONNX Runtime (CPU)                |
| API Server         | Flask                             |
| Frontend           | HTML / CSS / JavaScript Dashboard |
| Database           | SQLite                            |
| Face Recognition   | InsightFace (ArcFace)             |

## Models

| Model              | Parameters | Notes                              |
|--------------------|------------|-------------------------------------|
| EfficientNet-B3    | 12M        | Primary model, best accuracy        |
| EfficientNet-B0    | 5.3M       | Lightweight alternative             |
| MobileNet-V3       | 5.4M       | Optimized for mobile/edge inference |
| ResNet50-CBAM      | 25.6M      | Attention-augmented baseline        |

## Project Structure

```
src/
  data/              - Data loading, transforms
  models/            - Model architectures, training
  attention/         - Attention scoring, head pose
  face_recognition/  - Face detection, recognition
  api/               - Flask server, inference
  dashboard/         - HTML/CSS/JS frontend
configs/             - Configuration
notebooks/           - Training notebooks
scripts/             - Data preparation, utilities
tests/               - Test suite (149+ tests)
docs/                - Documentation
```

## Quick Start

### Installation

```bash
pip install -r requirements-train.txt
```

### Training

```bash
python -m src.models.train --model efficientnet_b3 --data_dir data/processed
```

### Server

```bash
python -m src.api.run --model-path models/emotion_model.onnx
```

### Tests

```bash
pytest tests/ -v
```

## Training

The training pipeline uses a two-phase approach:

1. **Head-only training:** Freeze the backbone and train only the classification head to adapt pretrained features to the emotion recognition task.
2. **Full fine-tuning:** Unfreeze all layers and fine-tune the entire network with a lower learning rate.

Training supports mixed precision (FP16) for faster computation on GPU, and includes checkpoint/resume functionality to recover from interruptions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
