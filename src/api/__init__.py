"""Flask API and inference engine for student attention detection."""

from src.api.camera import CameraManager
from src.api.inference import EmotionInferenceEngine

__all__ = [
    "CameraManager",
    "EmotionInferenceEngine",
]
