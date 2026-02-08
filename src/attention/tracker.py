"""Per-student attention tracking over a session."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _StudentRecord:
    """Internal record that accumulates per-frame data for one student."""

    scores: list[float] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


class StudentAttentionTracker:
    """Track attention data per student over time."""

    def __init__(self) -> None:
        self._data: dict[str, _StudentRecord] = defaultdict(_StudentRecord)

    def update(
        self,
        student_id: str,
        emotion: str,
        confidence: float,
        attention_score: float,
        attention_level: str,
        timestamp: float | None = None,
    ) -> None:
        """Append a new observation for the given student.

        Args:
            student_id: Unique identifier for the student.
            emotion: Predicted emotion class.
            confidence: Model confidence for the prediction.
            attention_score: Numeric attention score (0-1).
            attention_level: "focused", "moderate", or "distracted".
            timestamp: Unix timestamp; defaults to current time.
        """
        rec = self._data[student_id]
        rec.scores.append(attention_score)
        rec.levels.append(attention_level)
        rec.emotions.append(emotion)
        rec.confidences.append(confidence)
        rec.timestamps.append(timestamp if timestamp is not None else time.time())

    def get_student_summary(self, student_id: str) -> dict:
        """Return aggregate statistics for a single student.

        Returns:
            Dict with avg_score, time_focused (%), time_moderate (%),
            time_distracted (%), dominant_emotion, and total_frames.
        """
        rec = self._data.get(student_id)
        if rec is None or not rec.scores:
            return {
                "avg_score": 0.0,
                "time_focused": 0.0,
                "time_moderate": 0.0,
                "time_distracted": 0.0,
                "dominant_emotion": "unknown",
                "total_frames": 0,
            }

        total = len(rec.scores)
        return {
            "avg_score": sum(rec.scores) / total,
            "time_focused": rec.levels.count("focused") / total * 100,
            "time_moderate": rec.levels.count("moderate") / total * 100,
            "time_distracted": rec.levels.count("distracted") / total * 100,
            "dominant_emotion": max(set(rec.emotions), key=rec.emotions.count),
            "total_frames": total,
        }

    def get_session_summary(self) -> dict:
        """Return summaries for every tracked student plus a class average.

        Returns:
            Dict mapping student IDs to their summaries, plus a
            "class_average" key with the mean attention score.
        """
        summaries: dict[str, dict] = {}
        all_scores: list[float] = []

        for sid in self._data:
            s = self.get_student_summary(sid)
            summaries[sid] = s
            all_scores.append(s["avg_score"])

        summaries["class_average"] = (
            sum(all_scores) / len(all_scores) if all_scores else 0.0
        )
        return summaries

    def get_timeline(self, student_id: str) -> list[dict]:
        """Return a chronological list of observations for plotting.

        Returns:
            List of {"timestamp", "score", "level", "emotion"} dicts.
        """
        rec = self._data.get(student_id)
        if rec is None:
            return []

        return [
            {
                "timestamp": rec.timestamps[i],
                "score": rec.scores[i],
                "level": rec.levels[i],
                "emotion": rec.emotions[i],
            }
            for i in range(len(rec.scores))
        ]

    def reset(self) -> None:
        """Clear all tracked data."""
        self._data.clear()
