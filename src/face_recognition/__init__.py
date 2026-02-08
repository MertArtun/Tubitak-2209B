"""Face recognition sub-package: detection, embedding, database, pipeline."""

from src.face_recognition.database import StudentDatabase
from src.face_recognition.detector import FaceDetector
from src.face_recognition.pipeline import FaceRecognitionPipeline
from src.face_recognition.recognizer import FaceRecognizer

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "StudentDatabase",
    "FaceRecognitionPipeline",
]
