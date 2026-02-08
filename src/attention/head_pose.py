"""Head pose estimation using MediaPipe Face Mesh and cv2.solvePnP."""

from __future__ import annotations

import numpy as np

from configs.config import (
    HEAD_POSE_PITCH_THRESHOLD,
    HEAD_POSE_YAW_THRESHOLD,
    POSE_SCORE_AWAY,
    POSE_SCORE_FORWARD,
    POSE_SCORE_PARTIAL,
)

# 3D model points for a generic face (nose-centred coordinate system)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh landmark indices for the 6 key points above
_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    """Estimate head yaw / pitch / roll from a single RGB frame."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate_pose(
        self, frame: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Return (yaw, pitch, roll) in degrees, or None if no face found.

        Args:
            frame: BGR image (H, W, 3).
        """
        import cv2

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 2D image points from detected landmarks
        image_points = np.array(
            [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in _LANDMARK_IDS],
            dtype=np.float64,
        )

        # Approximate camera internals from frame dimensions
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Decompose rotation matrix into Euler angles
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1]))
            yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
            roll = 0.0

        return float(yaw), float(pitch), float(roll)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_looking_forward(yaw: float, pitch: float) -> bool:
        """True when the student is roughly facing the camera."""
        return (
            abs(yaw) < HEAD_POSE_YAW_THRESHOLD
            and abs(pitch) < HEAD_POSE_PITCH_THRESHOLD
        )

    @staticmethod
    def get_gaze_direction(yaw: float, pitch: float) -> str:
        """Classify gaze into a human-readable direction label."""
        if (
            abs(yaw) < HEAD_POSE_YAW_THRESHOLD
            and abs(pitch) < HEAD_POSE_PITCH_THRESHOLD
        ):
            return "forward"
        if yaw < -HEAD_POSE_YAW_THRESHOLD:
            return "right"
        if yaw > HEAD_POSE_YAW_THRESHOLD:
            return "left"
        if pitch < -HEAD_POSE_PITCH_THRESHOLD:
            return "up"
        return "down"

    @staticmethod
    def get_pose_score(yaw: float, pitch: float) -> float:
        """Numeric pose score: 1.0 (forward), 0.3 (partial), 0.0 (away)."""
        if HeadPoseEstimator.is_looking_forward(yaw, pitch):
            return POSE_SCORE_FORWARD

        partial_yaw = HEAD_POSE_YAW_THRESHOLD * 1.5
        partial_pitch = HEAD_POSE_PITCH_THRESHOLD * 1.5
        if abs(yaw) < partial_yaw and abs(pitch) < partial_pitch:
            return POSE_SCORE_PARTIAL

        return POSE_SCORE_AWAY

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._face_mesh.close()
