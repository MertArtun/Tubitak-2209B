"""Hybrid attention scorer combining emotion and head-pose signals."""

from __future__ import annotations

from configs.config import (
    ATTENTION_THRESHOLDS,
    HYBRID_EMOTION_WEIGHT,
    HYBRID_POSE_WEIGHT,
)


class HybridAttentionScorer:
    """Weighted combination of emotion-based and pose-based attention scores."""

    def __init__(
        self,
        emotion_weight: float = HYBRID_EMOTION_WEIGHT,
        pose_weight: float = HYBRID_POSE_WEIGHT,
    ) -> None:
        self.emotion_weight = emotion_weight
        self.pose_weight = pose_weight

    def compute_score(self, emotion_score: float, pose_score: float) -> float:
        """Return the weighted hybrid attention score.

        Args:
            emotion_score: Score from the emotion-based scorer (0-1).
            pose_score: Score from head-pose estimation (0-1).

        Returns:
            Combined score in [0, 1].
        """
        return self.emotion_weight * emotion_score + self.pose_weight * pose_score

    @staticmethod
    def classify_attention(score: float) -> str:
        """Map a numeric score to an attention level label.

        Returns:
            "focused", "moderate", or "distracted".
        """
        if score > ATTENTION_THRESHOLDS["focused"]:
            return "focused"
        if score > ATTENTION_THRESHOLDS["moderate"]:
            return "moderate"
        return "distracted"
