"""Tests for the Flask API endpoints.

Since app.py may still be under development, these tests are designed to work
with the existing inference engine and camera modules, using mocks for heavy
dependencies (ONNX, InsightFace, MediaPipe).
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api.inference import EmotionInferenceEngine

# ── EmotionInferenceEngine tests ─────────────────────────────────────────────


class TestEmotionInferencePreprocess:
    """Test the preprocessing pipeline of the inference engine."""

    @pytest.fixture
    def mock_engine(self, tmp_path):
        """Create an engine with a mocked ONNX session."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch("onnxruntime.InferenceSession") as mock_cls:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_session.get_inputs.return_value = [mock_input]
            mock_cls.return_value = mock_session
            engine = EmotionInferenceEngine(model_path)
        return engine

    def test_preprocess_output_shape(self, mock_engine, sample_image):
        """Preprocess should return shape (1, 3, 224, 224)."""
        result = mock_engine.preprocess(sample_image)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_dtype(self, mock_engine, sample_image):
        """Preprocess output should be float32."""
        result = mock_engine.preprocess(sample_image)
        assert result.dtype == np.float32

    def test_preprocess_value_range(self, mock_engine, sample_image):
        """After ImageNet normalization, values should be in reasonable range."""
        result = mock_engine.preprocess(sample_image)
        assert result.min() >= -4.0
        assert result.max() <= 4.0


class TestEmotionInferencePredict:
    """Test prediction with mocked ONNX session."""

    @pytest.fixture
    def engine_with_mock_session(self, tmp_path):
        """Create an engine with controlled mock outputs."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch("onnxruntime.InferenceSession") as mock_cls:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_session.get_inputs.return_value = [mock_input]

            # Return logits that strongly predict "positive" (index 2)
            mock_session.run.return_value = [np.array([[0.1, 0.2, 5.0]])]
            mock_cls.return_value = mock_session

            engine = EmotionInferenceEngine(model_path)
        return engine

    def test_predict_returns_dict(self, engine_with_mock_session, sample_image):
        """predict() should return a dict with class, confidence, probabilities."""
        result = engine_with_mock_session.predict(sample_image)
        assert "class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_class(self, engine_with_mock_session, sample_image):
        """With logits biased toward positive, class should be 'positive'."""
        result = engine_with_mock_session.predict(sample_image)
        assert result["class"] == "positive"

    def test_predict_confidence_range(self, engine_with_mock_session, sample_image):
        """Confidence should be in [0, 1]."""
        result = engine_with_mock_session.predict(sample_image)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_probabilities_sum(self, engine_with_mock_session, sample_image):
        """Probabilities should sum to ~1.0."""
        result = engine_with_mock_session.predict(sample_image)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_batch_empty(self, engine_with_mock_session):
        """Empty batch should return empty list."""
        result = engine_with_mock_session.predict_batch([])
        assert result == []


class TestEmotionInferenceInit:
    """Test inference engine initialization."""

    def test_init_file_not_found(self, tmp_path):
        """Non-existent model path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            EmotionInferenceEngine(tmp_path / "nonexistent.onnx")

    def test_softmax_numerically_stable(self):
        """_softmax should handle large values without overflow."""
        large_logits = np.array([1000.0, 1001.0, 999.0])
        result = EmotionInferenceEngine._softmax(large_logits)
        assert abs(result.sum() - 1.0) < 1e-5
        assert np.all(np.isfinite(result))


# ── CameraManager tests ─────────────────────────────────────────────────────


class TestCameraManager:
    """Test camera manager with mocked OpenCV VideoCapture."""

    def test_camera_manager_lifecycle(self):
        """Open and release should work without errors."""
        from src.api.camera import CameraManager

        with patch("src.api.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(source=0)
            cam.open()
            assert cam.is_opened()

            cam.release()
            mock_cap.release.assert_called_once()

    def test_camera_manager_read(self):
        """Read should return (True, frame) when capture is open."""
        from src.api.camera import CameraManager

        with patch("src.api.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap

            cam = CameraManager(source=0)
            cam.open()
            success, frame = cam.read()
            assert success is True
            assert frame is not None

    def test_camera_manager_read_closed(self):
        """Read on closed camera should return (False, None)."""
        from src.api.camera import CameraManager

        cam = CameraManager(source=0)
        success, frame = cam.read()
        assert success is False
        assert frame is None

    def test_camera_manager_context_manager(self):
        """Context manager should open and release."""
        from src.api.camera import CameraManager

        with patch("src.api.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = mock_cap

            with CameraManager(source=0) as cam:
                assert cam.is_opened()

            mock_cap.release.assert_called_once()

    def test_camera_manager_get_fps(self):
        """get_fps should return float from VideoCapture."""
        from src.api.camera import CameraManager

        with patch("src.api.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FPS = 5  # OpenCV constant

            cam = CameraManager(source=0)
            cam.open()
            assert cam.get_fps() == 30.0

    def test_camera_manager_get_fps_closed(self):
        """get_fps on closed camera should return 0.0."""
        from src.api.camera import CameraManager

        cam = CameraManager(source=0)
        assert cam.get_fps() == 0.0


# ── Flask App /api/analyze endpoint tests ──────────────────────────────────


class TestAnalyzeEndpoint:
    """Test the /api/analyze endpoint with mocked dependencies."""

    @pytest.fixture
    def client_with_mocks(self, tmp_path):
        """Create a Flask test client with all dependencies mocked."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        patches = [
            patch("src.api.app.EmotionInferenceEngine"),
            patch("src.api.app.FaceDetector"),
            patch("src.api.app.FaceRecognizer"),
            patch("src.api.app.StudentDatabase"),
            patch("src.api.app.AttentionScorer"),
            patch("src.api.app.HeadPoseEstimator"),
            patch("src.api.app.HybridAttentionScorer"),
            patch("src.api.app.StudentAttentionTracker"),
            patch("src.api.app.cv2"),
        ]
        started = [p.start() for p in patches]
        (
            engine_cls, detector_cls, recognizer_cls, db_cls,
            scorer_cls, hpe_cls, hybrid_cls, tracker_cls, mock_cv2,
        ) = started

        # cv2 mock – always returns a valid image
        mock_cv2.imdecode.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_cv2.IMREAD_COLOR = 1

        # Engine – always predicts "positive"
        engine_cls.return_value.predict.return_value = {
            "class": "positive",
            "confidence": 0.9,
            "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
        }

        # Detector – one face found
        detector_cls.return_value.detect.return_value = [
            {
                "bbox": [10, 20, 100, 120],
                "face_crop": np.zeros((112, 112, 3), dtype=np.uint8),
            }
        ]

        # Recognizer – known student with 0.85 confidence
        recognizer_cls.return_value.get_embedding.return_value = np.zeros(
            512, dtype=np.float32
        )
        recognizer_cls.return_value.identify.return_value = (1, 0.85)

        # Database
        db_cls.return_value.get_all_embeddings.return_value = {}
        db_cls.return_value.get_student.return_value = {"id": 1, "name": "Test"}
        db_cls.return_value.create_session.return_value = 1

        # Scorer
        scorer_cls.return_value.engagement_score.return_value = 0.72
        scorer_cls.return_value.classify_attention.return_value = "focused"
        scorer_cls.return_value.detect_anomaly.return_value = False

        # Head Pose Estimator
        hpe_cls.return_value.estimate_pose.return_value = (5.0, -3.0, 1.0)
        hpe_cls.get_pose_score = MagicMock(return_value=1.0)
        hpe_cls.get_gaze_direction = MagicMock(return_value="forward")

        # Hybrid Scorer
        hybrid_cls.return_value.compute_score.return_value = 0.84
        hybrid_cls.return_value.classify_attention.return_value = "focused"

        # Tracker
        tracker_cls.return_value.get_timeline.return_value = []

        # Reset module-level state and create app
        import src.api.app as app_mod

        app_mod.active_session_id = None
        app_mod.frame_count = 0
        app_mod.active_mode = "online"

        from src.api.app import create_app

        app = create_app(model_path, tmp_path / "test.db")
        app.testing = True

        mocks = {
            "engine": engine_cls.return_value,
            "detector": detector_cls.return_value,
            "recognizer": recognizer_cls.return_value,
            "db": db_cls.return_value,
            "scorer": scorer_cls.return_value,
            "hpe": hpe_cls.return_value,
            "hpe_cls": hpe_cls,
            "hybrid": hybrid_cls.return_value,
            "tracker": tracker_cls.return_value,
        }
        yield app.test_client(), mocks

        for p in patches:
            p.stop()

    def _post_image(self, client):
        """POST a fake image to /api/analyze."""
        data = {"image": (io.BytesIO(b"\xff\xd8\xff\xe0fake"), "test.jpg")}
        return client.post(
            "/api/analyze", data=data, content_type="multipart/form-data"
        )

    # ── Task 9: is_known and identity_confidence ──────────────────

    def test_response_has_is_known_true(self, client_with_mocks):
        """Known face should have is_known=True."""
        client, _ = client_with_mocks
        resp = self._post_image(client)
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["results"][0]["is_known"] is True

    def test_response_unknown_face(self, client_with_mocks):
        """Unrecognized face should have is_known=False, identity_confidence=0."""
        client, mocks = client_with_mocks
        mocks["recognizer"].identify.return_value = (None, 0.0)
        resp = self._post_image(client)
        data = resp.get_json()
        assert data["results"][0]["is_known"] is False
        assert data["results"][0]["identity_confidence"] == 0.0

    def test_response_has_identity_confidence(self, client_with_mocks):
        """Known face should have positive identity_confidence."""
        client, _ = client_with_mocks
        resp = self._post_image(client)
        data = resp.get_json()
        assert isinstance(data["results"][0]["identity_confidence"], float)
        assert data["results"][0]["identity_confidence"] == 0.85

    # ── Task 8: anomaly_detected ──────────────────────────────────

    def test_response_has_anomaly_detected(self, client_with_mocks):
        """Response should include anomaly_detected boolean."""
        client, _ = client_with_mocks
        resp = self._post_image(client)
        data = resp.get_json()
        assert "anomaly_detected" in data["results"][0]
        assert isinstance(data["results"][0]["anomaly_detected"], bool)

    def test_anomaly_detected_true(self, client_with_mocks):
        """anomaly_detected should be True when scorer detects anomaly."""
        client, mocks = client_with_mocks
        mocks["scorer"].detect_anomaly.return_value = True
        resp = self._post_image(client)
        data = resp.get_json()
        assert data["results"][0]["anomaly_detected"] is True

    # ── Task 7: face-to-face mode with pose ───────────────────────

    def test_online_mode_no_pose(self, client_with_mocks):
        """In online mode, response should NOT contain pose data."""
        client, _ = client_with_mocks
        resp = self._post_image(client)
        data = resp.get_json()
        assert "pose" not in data["results"][0]

    def test_face_to_face_mode_has_pose(self, client_with_mocks):
        """In face-to-face mode, response should include pose data."""
        client, _ = client_with_mocks
        # Start a face-to-face session
        client.post(
            "/api/sessions/start",
            json={"name": "Test", "mode": "face-to-face"},
        )
        resp = self._post_image(client)
        data = resp.get_json()
        result = data["results"][0]
        assert "pose" in result
        assert "yaw" in result["pose"]
        assert "pitch" in result["pose"]
        assert "roll" in result["pose"]
        assert "gaze_direction" in result["pose"]
        assert "pose_score" in result["pose"]

    def test_face_to_face_uses_hybrid_scorer(self, client_with_mocks):
        """In face-to-face mode, attention score should use hybrid scorer."""
        client, mocks = client_with_mocks
        client.post(
            "/api/sessions/start",
            json={"name": "Test", "mode": "face-to-face"},
        )
        self._post_image(client)
        mocks["hybrid"].compute_score.assert_called_once()

    # ── Task 8: tracker integration ───────────────────────────────

    def test_tracker_called_for_known_student(self, client_with_mocks):
        """Tracker should be updated for recognized students."""
        client, mocks = client_with_mocks
        self._post_image(client)
        mocks["tracker"].update.assert_called_once()

    def test_tracker_not_called_for_unknown(self, client_with_mocks):
        """Tracker should not update for unrecognized faces."""
        client, mocks = client_with_mocks
        mocks["recognizer"].identify.return_value = (None, 0.0)
        self._post_image(client)
        mocks["tracker"].update.assert_not_called()

    def test_stop_session_resets_mode(self, client_with_mocks):
        """Stopping a session should reset mode to online."""
        client, _ = client_with_mocks
        # Start face-to-face session
        resp = client.post(
            "/api/sessions/start",
            json={"name": "Test", "mode": "face-to-face"},
        )
        session_id = resp.get_json()["session_id"]

        # Stop it
        client.post(f"/api/sessions/{session_id}/stop")

        # Analyze should be in online mode (no pose)
        resp = self._post_image(client)
        data = resp.get_json()
        assert "pose" not in data["results"][0]
