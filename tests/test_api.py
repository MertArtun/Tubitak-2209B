"""Tests for the Flask API endpoints.

Since app.py may still be under development, these tests are designed to work
with the existing inference engine and camera modules, using mocks for heavy
dependencies (ONNX, InsightFace, MediaPipe).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.api.inference import EmotionInferenceEngine


# ── EmotionInferenceEngine tests ─────────────────────────────────────────────


class TestEmotionInferencePreprocess:
    """Test the preprocessing pipeline of the inference engine."""

    @pytest.fixture
    def mock_engine(self, tmp_path):
        """Create an engine with a mocked ONNX session."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch("src.api.inference.ort") as mock_ort:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_session.get_inputs.return_value = [mock_input]
            mock_ort.InferenceSession.return_value = mock_session
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

        with patch("src.api.inference.ort") as mock_ort:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_session.get_inputs.return_value = [mock_input]

            # Return logits that strongly predict "positive" (index 2)
            mock_session.run.return_value = [np.array([[0.1, 0.2, 5.0]])]
            mock_ort.InferenceSession.return_value = mock_session

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
