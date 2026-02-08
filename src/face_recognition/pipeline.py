"""End-to-end face recognition pipeline: detect, embed, identify."""

from __future__ import annotations

import logging

import numpy as np

from src.face_recognition.database import StudentDatabase
from src.face_recognition.detector import FaceDetector
from src.face_recognition.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """Combines face detection, recognition and database look-up into a single step."""

    def __init__(
        self,
        db: StudentDatabase,
        detector: FaceDetector | None = None,
        recognizer: FaceRecognizer | None = None,
    ) -> None:
        self.db = db
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Detect and identify all faces in a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of dicts with keys:
                - student_id: int or None (unknown face).
                - name: str or None.
                - bbox: [x1, y1, x2, y2].
                - confidence: float (recognition similarity; 0.0 for unknowns).
        """
        faces = self.detector.detect(frame)
        if not faces:
            return []

        known_embeddings = self.db.get_all_embeddings()
        results: list[dict] = []

        for face_info in faces:
            try:
                embedding = self.recognizer.get_embedding(face_info)
            except ValueError:
                # No embedding available, treat as unknown
                results.append(
                    {
                        "student_id": None,
                        "name": None,
                        "bbox": face_info["bbox"],
                        "confidence": 0.0,
                    }
                )
                continue

            student_id, similarity = self.recognizer.identify(
                embedding, known_embeddings
            )

            name: str | None = None
            if student_id is not None:
                student = self.db.get_student(student_id)
                name = student["name"] if student else None

            results.append(
                {
                    "student_id": student_id,
                    "name": name,
                    "bbox": face_info["bbox"],
                    "confidence": similarity,
                }
            )

        return results

    def register_student(
        self, name: str, email: str, frames: list[np.ndarray]
    ) -> int:
        """Register a new student from a set of captured frames.

        Detects the largest face in each frame, computes embeddings, averages
        them, and stores the result in the database.

        Args:
            name: Student full name.
            email: Student email.
            frames: List of BGR images containing the student's face.

        Returns:
            The new student_id.

        Raises:
            ValueError: If no face is detected in any of the provided frames.
        """
        embeddings: list[np.ndarray] = []

        for frame in frames:
            faces = self.detector.detect(frame)
            if not faces:
                continue
            # Pick the largest detected face
            largest = max(
                faces,
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            )
            try:
                emb = self.recognizer.get_embedding(largest)
                embeddings.append(emb)
            except ValueError:
                continue

        if not embeddings:
            raise ValueError(
                f"Could not extract any face embedding from {len(frames)} frame(s)."
            )

        student_id = self.db.add_student(name, email)

        # Save individual embeddings
        for emb in embeddings:
            self.db.save_embedding(student_id, emb)

        logger.info(
            "Registered student '%s' (id=%d) with %d embedding(s).",
            name,
            student_id,
            len(embeddings),
        )
        return student_id
