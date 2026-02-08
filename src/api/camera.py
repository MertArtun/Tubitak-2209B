"""Camera / video-source manager."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Thin wrapper around OpenCV VideoCapture with context-manager support."""

    def __init__(self, source: int | str = 0) -> None:
        """Create a camera manager.

        Args:
            source: Webcam index (``0``, ``1``, ...) or path to a video file.
        """
        self.source = source
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video source."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
        else:
            logger.info("Opened video source: %s", self.source)

    def release(self) -> None:
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Released video source: %s", self.source)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``np.ndarray`` or
            ``None`` on failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        """Return ``True`` if the capture device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        """Return the FPS reported by the video source (0.0 if unavailable)."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.release()
