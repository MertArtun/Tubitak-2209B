"""Shared fixtures for the Student Attention Detection test suite."""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_image():
    """Random 224x224x3 uint8 image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_48():
    """Random 48x48 grayscale image (FER2013 format)."""
    return np.random.randint(0, 255, (48, 48), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """Random batch tensor (2, 3, 224, 224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def in_memory_db(tmp_path):
    """SQLite database in temp directory."""
    from src.face_recognition.database import StudentDatabase

    db_path = tmp_path / "test.db"
    db = StudentDatabase(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def sample_embedding():
    """Random 512-dim normalized embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_predictions():
    """List of emotion prediction dicts."""
    return [
        {"class": "positive", "confidence": 0.9},
        {"class": "neutral", "confidence": 0.7},
        {"class": "negative", "confidence": 0.8},
        {"class": "positive", "confidence": 0.85},
        {"class": "neutral", "confidence": 0.6},
    ]
