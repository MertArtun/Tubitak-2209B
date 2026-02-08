"""Export a trained PyTorch model to ONNX format.

Validates that the ONNX output matches the PyTorch output and measures
ONNX Runtime inference speed.

Usage:
    python -m src.models.export_onnx --model efficientnet_b3 \
        --checkpoint models/efficientnet_b3_best.pth \
        --output models/emotion_model.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from configs.config import IMAGE_SIZE, MODELS_DIR
from src.models.architectures import create_model


def export_onnx(
    model_name: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export the model to ONNX with dynamic batch size.

    Args:
        model_name: Model key from MODEL_CONFIGS.
        checkpoint_path: Path to the saved .pth checkpoint.
        output_path: Where to write the .onnx file.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load PyTorch model ────────────────────────────────────────────────
    model = create_model(model_name, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Export ────────────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to: {output_path}")

    # ── File size ─────────────────────────────────────────────────────────
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    # ── Validate ──────────────────────────────────────────────────────────
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name

    dummy_np = dummy_input.numpy()
    onnx_output = session.run(None, {input_name: dummy_np})[0]

    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    if np.allclose(pytorch_output, onnx_output, atol=1e-5):
        print("Validation PASSED: ONNX output matches PyTorch (atol=1e-5).")
    else:
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"Validation WARNING: Max difference = {max_diff:.6e}")

    # ── ONNX Runtime inference time ───────────────────────────────────────
    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: dummy_np})

    times: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_np})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"ONNX Runtime inference: {avg_ms:.2f} +/- {std_ms:.2f} ms/frame (100 runs)")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export emotion model to ONNX.")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model name from MODEL_CONFIGS.")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "efficientnet_b3_best.pth"))
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "emotion_model.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
