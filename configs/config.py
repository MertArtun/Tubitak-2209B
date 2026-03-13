"""Project-wide configuration constants."""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "students.db"

# ─── Emotion Mapping (7 → 3 classes) ────────────────────────────────────────
# FER2013 original labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
ORIGINAL_EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# 3-class mapping (surprise excluded)
EMOTION_3CLASS_MAP = {
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "happy": "positive",
    "sad": "negative",
    "neutral": "neutral",
}

CLASS_NAMES = ["negative", "neutral", "positive"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 3

# FER2013 original label → 3-class index (surprise maps to None)
FER_TO_3CLASS_IDX = {
    0: CLASS_TO_IDX["negative"],   # angry → negative
    1: CLASS_TO_IDX["negative"],   # disgust → negative
    2: CLASS_TO_IDX["negative"],   # fear → negative
    3: CLASS_TO_IDX["positive"],   # happy → positive
    4: CLASS_TO_IDX["negative"],   # sad → negative
    5: None,                        # surprise → excluded
    6: CLASS_TO_IDX["neutral"],    # neutral → neutral
}

# ─── Data Pipeline ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Merged dataset split ratios
MERGED_TEST_RATIO = 0.10
MERGED_VAL_RATIO = 0.10

# ─── Data Cleaning ──────────────────────────────────────────────────────────
PROCESSED_MERGED_DIR = DATA_DIR / "processed_merged"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
CLEANING_REPORTS_DIR = RESULTS_DIR / "data_cleaning"
QUARANTINE_DIR = DATA_DIR / "quarantine"

PHASH_HASH_SIZE = 16
PHASH_HAMMING_THRESHOLD = 6
ENABLE_NEAR_DUP_SCAN = True
MAX_NEAR_DUP_CANDIDATES_PER_IMAGE = 4000
MIN_IMAGE_SIZE = 10           # pixels
MIN_FILE_SIZE_BYTES = 100

FERPLUS_CONFIDENCE_THRESHOLD = 0.4

# ─── MixUp / CutMix ────────────────────────────────────────────────────────
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_CUTMIX_PROB = 0.5

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 2

# Phase 1: Head-only training
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3

# Phase 2: Full fine-tuning
PHASE2_EPOCHS = 25
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 7

# Focal Loss
FOCAL_LOSS_GAMMA = 2.0

# ─── Model Configs ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "efficientnet_b3": {
        "timm_name": "efficientnet_b3",
        "feature_dim": 1536,
        "hidden_dim": 512,
        "dropout": 0.3,
        "dropout2": 0.2,
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "feature_dim": 1280,
        "hidden_dim": 256,
        "dropout": 0.3,
        "dropout2": 0.2,
    },
    "mobilenet_v3": {
        "timm_name": "mobilenetv3_large_100",
        "feature_dim": 1280,
        "hidden_dim": 256,
        "dropout": 0.3,
        "dropout2": 0.2,
    },
    "resnet50_cbam": {
        "timm_name": "resnet50",
        "feature_dim": 2048,
        "hidden_dim": 512,
        "dropout": 0.3,
        "dropout2": 0.2,
    },
}

# ─── Attention Scoring ───────────────────────────────────────────────────────
EMOTION_WEIGHTS = {
    "positive": 0.8,
    "neutral": 0.5,
    "negative": 0.2,
}

ATTENTION_THRESHOLDS = {
    "focused": 0.6,     # score > 0.6
    "moderate": 0.35,   # 0.35 < score <= 0.6
    # "distracted": score <= 0.35
}

ATTENTION_WINDOW_SIZE = 30  # frames for sliding window
ANOMALY_CHANGE_THRESHOLD = 0.4  # sudden change detection
ANOMALY_FRAME_WINDOW = 5

# ─── Head Pose (Face-to-face mode) ──────────────────────────────────────────
HEAD_POSE_YAW_THRESHOLD = 30.0    # degrees
HEAD_POSE_PITCH_THRESHOLD = 25.0  # degrees

HYBRID_EMOTION_WEIGHT = 0.6
HYBRID_POSE_WEIGHT = 0.4

POSE_SCORE_FORWARD = 1.0
POSE_SCORE_PARTIAL = 0.3
POSE_SCORE_AWAY = 0.0

# ─── Face Recognition ───────────────────────────────────────────────────────
FACE_RECOGNITION_THRESHOLD = 0.5  # cosine similarity
MIN_FACE_SIZE = 30  # pixels
FACE_DETECTION_CONFIDENCE = 0.5

# ─── Flask Server ────────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
ANALYZE_FPS = 2  # frames per second for analysis
