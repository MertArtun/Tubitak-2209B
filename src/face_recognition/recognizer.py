"""Face recognition (embedding extraction & comparison) using InsightFace ArcFace."""

from __future__ import annotations

import logging

import numpy as np

from configs.config import FACE_RECOGNITION_THRESHOLD

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Extract and compare face embeddings using InsightFace ArcFace model."""

    def __init__(self, threshold: float = FACE_RECOGNITION_THRESHOLD) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_embedding(self, face_info: dict) -> np.ndarray:
        """Extract a 512-dimensional embedding from an InsightFace result.

        Args:
            face_info: Dict returned by FaceDetector.detect(); must contain
                       the '_insightface' key with the raw InsightFace Face object.

        Returns:
            Normalised 512-d float32 embedding.

        Raises:
            ValueError: If the face object has no embedding attribute.
        """
        raw = face_info.get("_insightface")
        if raw is None or not hasattr(raw, "embedding") or raw.embedding is None:
            raise ValueError(
                "No embedding available; ensure FaceAnalysis ran with recognition model."
            )
        emb = np.array(raw.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compare(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Both embeddings are assumed to be L2-normalised, so the dot
        product equals cosine similarity.

        Returns:
            Similarity in [-1, 1].
        """
        return float(np.dot(embedding1, embedding2))

    def identify(
        self,
        embedding: np.ndarray,
        known_embeddings: dict[int, np.ndarray],
    ) -> tuple[int | None, float]:
        """Find the closest known student for a given embedding.

        Args:
            embedding: Query embedding (512-d, normalised).
            known_embeddings: Mapping student_id -> representative embedding.

        Returns:
            (student_id, similarity) of the best match, or (None, 0.0) if no
            match exceeds the recognition threshold.
        """
        if not known_embeddings:
            return None, 0.0

        best_id: int | None = None
        best_sim: float = -1.0

        for sid, ref_emb in known_embeddings.items():
            sim = self.compare(embedding, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        if best_sim >= self.threshold:
            return best_id, best_sim
        return None, 0.0

    # ------------------------------------------------------------------
    # Registration helper
    # ------------------------------------------------------------------

    @staticmethod
    def register_student(
        student_id: str, embeddings: list[np.ndarray]
    ) -> np.ndarray:
        """Compute a single representative embedding from multiple samples.

        Args:
            student_id: Identifier (used for logging only).
            embeddings: List of 512-d embeddings captured during registration.

        Returns:
            L2-normalised average embedding.

        Raises:
            ValueError: If the embeddings list is empty.
        """
        if not embeddings:
            raise ValueError(f"No embeddings provided for student {student_id}.")

        avg = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg
