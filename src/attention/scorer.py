"""Emotion-based attention scoring with sliding window."""

from __future__ import annotations

from configs.config import (
    ANOMALY_CHANGE_THRESHOLD,
    ANOMALY_FRAME_WINDOW,
    ATTENTION_THRESHOLDS,
    ATTENTION_WINDOW_SIZE,
    EMOTION_WEIGHTS,
)


class AttentionScorer:
    """Computes engagement scores from emotion predictions using a sliding window."""

    def __init__(self, window_size: int = ATTENTION_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._history: list[dict] = []

    def engagement_score(
        self, predictions: list[dict], window: int | None = None
    ) -> float:
        """Calculate engagement score over a sliding window of predictions.

        Args:
            predictions: List of {"class": "positive"|"negative"|"neutral",
                                   "confidence": float} dicts.
            window: Number of recent predictions to consider.
                    Defaults to self.window_size.

        Returns:
            Score in [0, 1].
        """
        if not predictions:
            return 0.0

        win = window if window is not None else self.window_size
        recent = predictions[-win:]

        total = sum(
            EMOTION_WEIGHTS.get(p["class"], 0.0) * p["confidence"]
            for p in recent
        )
        return total / len(recent)

    def classify_attention(self, score: float) -> str:
        """Map a numeric score to an attention level label.

        Returns:
            "focused", "moderate", or "distracted".
        """
        if score > ATTENTION_THRESHOLDS["focused"]:
            return "focused"
        if score > ATTENTION_THRESHOLDS["moderate"]:
            return "moderate"
        return "distracted"

    def detect_anomaly(self, scores: list[float]) -> bool:
        """Check for a sudden change in attention scores.

        Returns True if the absolute difference between any two
        consecutive scores in the last ANOMALY_FRAME_WINDOW frames
        exceeds ANOMALY_CHANGE_THRESHOLD.
        """
        if len(scores) < 2:
            return False

        recent = scores[-ANOMALY_FRAME_WINDOW:]
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) > ANOMALY_CHANGE_THRESHOLD:
                return True
        return False

    def reset(self) -> None:
        """Clear prediction history."""
        self._history.clear()
