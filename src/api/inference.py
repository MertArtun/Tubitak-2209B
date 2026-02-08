"""ONNX-based emotion inference engine."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from configs.config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class EmotionInferenceEngine:
    """Run emotion classification on face crops using an ONNX model."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the ONNX model.

        Args:
            model_path: Path to the exported ``.onnx`` model file.
        """
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX model loaded from %s", self.model_path)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize, normalise and format a face crop for inference.

        Args:
            face_crop: BGR or RGB image as ``(H, W, 3)`` uint8 array.

        Returns:
            Float32 array of shape ``(1, 3, 224, 224)``.
        """
        img = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalise to [0, 1] then apply ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img = (img - mean) / std

        # HWC -> CHW, add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, face_crop: np.ndarray) -> dict:
        """Run inference on a single face crop.

        Args:
            face_crop: BGR image ``(H, W, 3)``.

        Returns:
            Dict with ``class``, ``confidence``, and ``probabilities``.
        """
        tensor = self.preprocess(face_crop)
        logits = self._session.run(None, {self._input_name: tensor})[0]
        probs = self._softmax(logits[0])

        class_idx = int(np.argmax(probs))
        return {
            "class": CLASS_NAMES[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)
            },
        }

    def predict_batch(self, face_crops: list[np.ndarray]) -> list[dict]:
        """Run inference on multiple face crops.

        Args:
            face_crops: List of BGR images.

        Returns:
            List of prediction dicts (same format as :meth:`predict`).
        """
        if not face_crops:
            return []

        batch = np.concatenate(
            [self.preprocess(crop) for crop in face_crops], axis=0
        )
        logits = self._session.run(None, {self._input_name: batch})[0]

        results: list[dict] = []
        for row in logits:
            probs = self._softmax(row)
            class_idx = int(np.argmax(probs))
            results.append(
                {
                    "class": CLASS_NAMES[class_idx],
                    "confidence": float(probs[class_idx]),
                    "probabilities": {
                        name: float(probs[i])
                        for i, name in enumerate(CLASS_NAMES)
                    },
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
