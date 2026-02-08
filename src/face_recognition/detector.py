"""Face detection using InsightFace."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from configs.config import FACE_DETECTION_CONFIDENCE, MIN_FACE_SIZE

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces in video frames using InsightFace RetinaFace model."""

    def __init__(
        self,
        min_face_size: int = MIN_FACE_SIZE,
        confidence: float = FACE_DETECTION_CONFIDENCE,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.min_face_size = min_face_size
        self.confidence = confidence
        self.det_size = det_size
        self._app: Any = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize InsightFace FaceAnalysis model."""
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=self.det_size)
            logger.info("InsightFace FaceAnalysis initialized successfully.")
        except ImportError:
            logger.error(
                "insightface is not installed. "
                "Install it with: pip install insightface"
            )
        except Exception as exc:
            logger.error("Failed to initialize InsightFace: %s", exc)

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._app is not None

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a single BGR frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of dicts, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates.
                - landmarks: (5, 2) facial landmark array.
                - confidence: detection confidence score.
                - face_crop: cropped face region from the original frame.
        """
        if not self.is_ready:
            logger.warning("FaceDetector is not initialized; returning empty list.")
            return []

        try:
            faces = self._app.get(frame)
        except Exception as exc:
            logger.error("Face detection failed: %s", exc)
            return []

        results: list[dict] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            conf: float = float(face.det_score)

            # Filter by confidence
            if conf < self.confidence:
                continue

            # Filter by minimum face size
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w < self.min_face_size or h < self.min_face_size:
                continue

            # Clamp coordinates to frame boundaries
            h_frame, w_frame = frame.shape[:2]
            x1_c = max(0, x1)
            y1_c = max(0, y1)
            x2_c = min(w_frame, x2)
            y2_c = min(h_frame, y2)

            face_crop = frame[y1_c:y2_c, x1_c:x2_c].copy()

            results.append(
                {
                    "bbox": bbox,
                    "landmarks": face.kps if face.kps is not None else np.empty((0, 2)),
                    "confidence": conf,
                    "face_crop": face_crop,
                    "_insightface": face,  # keep raw object for embedding extraction
                }
            )

        return results
