"""Tests for attention scoring, head pose, hybrid scorer, and tracker."""

import pytest

from src.attention.scorer import AttentionScorer
from src.attention.head_pose import HeadPoseEstimator
from src.attention.hybrid_scorer import HybridAttentionScorer
from src.attention.tracker import StudentAttentionTracker


# ── AttentionScorer tests ────────────────────────────────────────────────────


class TestAttentionScorer:
    """Test emotion-based engagement scoring."""

    @pytest.fixture(autouse=True)
    def _scorer(self):
        self.scorer = AttentionScorer(window_size=30)

    def test_engagement_score_all_positive(self):
        """All positive predictions should yield a high score (>0.6)."""
        predictions = [
            {"class": "positive", "confidence": 0.9},
            {"class": "positive", "confidence": 0.85},
            {"class": "positive", "confidence": 0.95},
        ]
        score = self.scorer.engagement_score(predictions)
        assert score > 0.6

    def test_engagement_score_all_negative(self):
        """All negative predictions should yield a low score (<0.35)."""
        predictions = [
            {"class": "negative", "confidence": 0.9},
            {"class": "negative", "confidence": 0.85},
            {"class": "negative", "confidence": 0.8},
        ]
        score = self.scorer.engagement_score(predictions)
        assert score < 0.35

    def test_engagement_score_mixed(self, sample_predictions):
        """Mixed predictions should yield a moderate score."""
        score = self.scorer.engagement_score(sample_predictions)
        assert 0.2 < score < 0.8

    def test_engagement_score_empty(self):
        """Empty predictions should return 0.0."""
        score = self.scorer.engagement_score([])
        assert score == 0.0

    def test_classify_attention_focused(self):
        """Score 0.7 should classify as 'focused'."""
        assert self.scorer.classify_attention(0.7) == "focused"

    def test_classify_attention_moderate(self):
        """Score 0.5 should classify as 'moderate'."""
        assert self.scorer.classify_attention(0.5) == "moderate"

    def test_classify_attention_distracted(self):
        """Score 0.2 should classify as 'distracted'."""
        assert self.scorer.classify_attention(0.2) == "distracted"

    def test_classify_attention_boundary_focused(self):
        """Score exactly 0.6 is NOT focused (threshold is >, not >=)."""
        assert self.scorer.classify_attention(0.6) == "moderate"

    def test_classify_attention_boundary_moderate(self):
        """Score exactly 0.35 is NOT moderate (threshold is >, not >=)."""
        assert self.scorer.classify_attention(0.35) == "distracted"

    def test_sliding_window(self):
        """Window=3 should only use last 3 predictions."""
        # First 7 negative, last 3 positive
        predictions = [
            {"class": "negative", "confidence": 0.9},
        ] * 7 + [
            {"class": "positive", "confidence": 0.9},
        ] * 3

        # With window=3, only the 3 positive predictions are used
        score = self.scorer.engagement_score(predictions, window=3)
        # positive weight=0.8, confidence=0.9 -> 0.72 per prediction
        assert score > 0.6

    def test_anomaly_detection(self):
        """Score jump from 0.2 to 0.8 should be detected as anomaly."""
        scores = [0.5, 0.5, 0.5, 0.2, 0.8]
        assert self.scorer.detect_anomaly(scores) is True

    def test_no_anomaly(self):
        """Gradual change should not be detected as anomaly."""
        scores = [0.5, 0.52, 0.55, 0.57, 0.6]
        assert self.scorer.detect_anomaly(scores) is False

    def test_anomaly_single_score(self):
        """Single score should not trigger anomaly."""
        assert self.scorer.detect_anomaly([0.5]) is False

    def test_anomaly_empty_scores(self):
        """Empty scores should not trigger anomaly."""
        assert self.scorer.detect_anomaly([]) is False


# ── HeadPoseEstimator static method tests ────────────────────────────────────


class TestHeadPoseStatic:
    """Test static helpers of HeadPoseEstimator (no MediaPipe needed)."""

    def test_is_looking_forward_true(self):
        """Small yaw and pitch should be forward."""
        assert HeadPoseEstimator.is_looking_forward(yaw=5.0, pitch=10.0) is True

    def test_is_looking_forward_false_yaw(self):
        """Large yaw should not be forward."""
        assert HeadPoseEstimator.is_looking_forward(yaw=40.0, pitch=5.0) is False

    def test_is_looking_forward_false_pitch(self):
        """Large pitch should not be forward."""
        assert HeadPoseEstimator.is_looking_forward(yaw=5.0, pitch=30.0) is False

    def test_get_gaze_direction_forward(self):
        """Small angles yield 'forward'."""
        assert HeadPoseEstimator.get_gaze_direction(yaw=5.0, pitch=5.0) == "forward"

    def test_get_gaze_direction_right(self):
        """Large negative yaw yields 'right'."""
        assert HeadPoseEstimator.get_gaze_direction(yaw=-40.0, pitch=5.0) == "right"

    def test_get_gaze_direction_left(self):
        """Large positive yaw yields 'left'."""
        assert HeadPoseEstimator.get_gaze_direction(yaw=40.0, pitch=5.0) == "left"

    def test_get_gaze_direction_up(self):
        """Large negative pitch yields 'up'."""
        assert HeadPoseEstimator.get_gaze_direction(yaw=5.0, pitch=-30.0) == "up"

    def test_get_gaze_direction_down(self):
        """Large positive pitch yields 'down'."""
        assert HeadPoseEstimator.get_gaze_direction(yaw=5.0, pitch=30.0) == "down"

    def test_pose_score_forward(self):
        """Looking forward should return 1.0."""
        assert HeadPoseEstimator.get_pose_score(yaw=5.0, pitch=5.0) == 1.0

    def test_pose_score_partial(self):
        """Partially looking away should return 0.3."""
        # Yaw just outside forward threshold but within partial (30 < 35 < 45)
        assert HeadPoseEstimator.get_pose_score(yaw=35.0, pitch=5.0) == 0.3

    def test_pose_score_away(self):
        """Looking completely away should return 0.0."""
        assert HeadPoseEstimator.get_pose_score(yaw=60.0, pitch=50.0) == 0.0


# ── HybridAttentionScorer tests ──────────────────────────────────────────────


class TestHybridScorer:
    """Test hybrid attention scoring combining emotion and pose."""

    def test_hybrid_scorer_weights(self):
        """emotion_weight + pose_weight should combine correctly."""
        scorer = HybridAttentionScorer(emotion_weight=0.6, pose_weight=0.4)
        score = scorer.compute_score(emotion_score=0.8, pose_score=0.5)
        expected = 0.6 * 0.8 + 0.4 * 0.5
        assert abs(score - expected) < 1e-6

    def test_hybrid_scorer_forward_looking(self):
        """Looking forward + positive emotion should yield a high score."""
        scorer = HybridAttentionScorer(emotion_weight=0.6, pose_weight=0.4)
        score = scorer.compute_score(emotion_score=0.9, pose_score=1.0)
        assert score > 0.8

    def test_hybrid_scorer_all_zero(self):
        """Both zero should yield zero."""
        scorer = HybridAttentionScorer(emotion_weight=0.6, pose_weight=0.4)
        score = scorer.compute_score(emotion_score=0.0, pose_score=0.0)
        assert score == 0.0

    def test_hybrid_classify_focused(self):
        """High score -> 'focused'."""
        assert HybridAttentionScorer.classify_attention(0.7) == "focused"

    def test_hybrid_classify_moderate(self):
        """Medium score -> 'moderate'."""
        assert HybridAttentionScorer.classify_attention(0.5) == "moderate"

    def test_hybrid_classify_distracted(self):
        """Low score -> 'distracted'."""
        assert HybridAttentionScorer.classify_attention(0.2) == "distracted"


# ── StudentAttentionTracker tests ────────────────────────────────────────────


class TestTracker:
    """Test per-student attention tracking."""

    @pytest.fixture(autouse=True)
    def _tracker(self):
        self.tracker = StudentAttentionTracker()

    def test_tracker_update_and_summary(self):
        """Update multiple entries, check summary stats."""
        self.tracker.update("s1", "positive", 0.9, 0.8, "focused", timestamp=1.0)
        self.tracker.update("s1", "neutral", 0.7, 0.5, "moderate", timestamp=2.0)
        self.tracker.update("s1", "negative", 0.8, 0.3, "distracted", timestamp=3.0)

        summary = self.tracker.get_student_summary("s1")
        assert summary["total_frames"] == 3
        assert abs(summary["avg_score"] - (0.8 + 0.5 + 0.3) / 3) < 1e-6
        assert summary["dominant_emotion"] in {"positive", "neutral", "negative"}

    def test_tracker_summary_unknown_student(self):
        """Summary for unknown student should return defaults."""
        summary = self.tracker.get_student_summary("unknown")
        assert summary["total_frames"] == 0
        assert summary["avg_score"] == 0.0
        assert summary["dominant_emotion"] == "unknown"

    def test_tracker_timeline(self):
        """Timeline should return entries in correct order."""
        self.tracker.update("s1", "positive", 0.9, 0.8, "focused", timestamp=1.0)
        self.tracker.update("s1", "neutral", 0.7, 0.5, "moderate", timestamp=2.0)
        self.tracker.update("s1", "negative", 0.6, 0.2, "distracted", timestamp=3.0)

        timeline = self.tracker.get_timeline("s1")
        assert len(timeline) == 3
        assert timeline[0]["timestamp"] == 1.0
        assert timeline[1]["timestamp"] == 2.0
        assert timeline[2]["timestamp"] == 3.0
        assert timeline[0]["emotion"] == "positive"

    def test_tracker_timeline_unknown(self):
        """Timeline for unknown student should be empty."""
        assert self.tracker.get_timeline("unknown") == []

    def test_tracker_reset(self):
        """After reset, state should be empty."""
        self.tracker.update("s1", "positive", 0.9, 0.8, "focused")
        self.tracker.update("s2", "neutral", 0.7, 0.5, "moderate")

        self.tracker.reset()

        assert self.tracker.get_student_summary("s1")["total_frames"] == 0
        assert self.tracker.get_student_summary("s2")["total_frames"] == 0

    def test_session_summary(self):
        """Session summary should include per-student stats and class average."""
        self.tracker.update("s1", "positive", 0.9, 0.8, "focused", timestamp=1.0)
        self.tracker.update("s2", "negative", 0.8, 0.2, "distracted", timestamp=1.0)

        summary = self.tracker.get_session_summary()
        assert "s1" in summary
        assert "s2" in summary
        assert "class_average" in summary
        # Class average of (0.8 + 0.2) / 2 = 0.5
        assert abs(summary["class_average"] - 0.5) < 1e-6

    def test_tracker_attention_distribution(self):
        """Time percentages should sum to 100."""
        self.tracker.update("s1", "positive", 0.9, 0.8, "focused", timestamp=1.0)
        self.tracker.update("s1", "neutral", 0.7, 0.5, "moderate", timestamp=2.0)
        self.tracker.update("s1", "negative", 0.6, 0.2, "distracted", timestamp=3.0)
        self.tracker.update("s1", "positive", 0.9, 0.9, "focused", timestamp=4.0)

        summary = self.tracker.get_student_summary("s1")
        total_pct = (
            summary["time_focused"]
            + summary["time_moderate"]
            + summary["time_distracted"]
        )
        assert abs(total_pct - 100.0) < 1e-6
