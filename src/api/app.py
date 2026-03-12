"""Flask application for student attention detection."""

from __future__ import annotations

import base64
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file
from flask_cors import CORS

from configs.config import DB_PATH
from src.api.inference import EmotionInferenceEngine
from src.attention.head_pose import HeadPoseEstimator
from src.attention.hybrid_scorer import HybridAttentionScorer
from src.attention.scorer import AttentionScorer
from src.attention.tracker import StudentAttentionTracker
from src.face_recognition.database import StudentDatabase
from src.face_recognition.detector import FaceDetector
from src.face_recognition.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────
engine: EmotionInferenceEngine | None = None
detector: FaceDetector | None = None
recognizer: FaceRecognizer | None = None
db: StudentDatabase | None = None
scorer: AttentionScorer | None = None
head_pose_estimator: HeadPoseEstimator | None = None
hybrid_scorer: HybridAttentionScorer | None = None
tracker: StudentAttentionTracker | None = None
active_session_id: int | None = None
active_mode: str = "online"
frame_count: int = 0


def create_app(
    model_path: str | Path,
    db_path: str | Path = DB_PATH,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        model_path: Path to the ONNX emotion model.
        db_path: Path to the SQLite database.

    Returns:
        Configured Flask app instance.
    """
    global engine, detector, recognizer, db, scorer, head_pose_estimator, hybrid_scorer, tracker

    app = Flask(
        __name__,
        template_folder=str(
            Path(__file__).resolve().parent.parent / "dashboard" / "templates"
        ),
        static_folder=str(
            Path(__file__).resolve().parent.parent / "dashboard" / "static"
        ),
    )
    CORS(app)

    # Initialise components
    engine = EmotionInferenceEngine(model_path)
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    db = StudentDatabase(db_path)
    scorer = AttentionScorer()
    head_pose_estimator = HeadPoseEstimator()
    hybrid_scorer = HybridAttentionScorer()
    tracker = StudentAttentionTracker()

    _register_routes(app)
    logger.info("Flask app created. Model: %s  DB: %s", model_path, db_path)
    return app


# ── Helpers ──────────────────────────────────────────────────────────────────


def _decode_image(data: str | bytes) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR numpy array."""
    if isinstance(data, str):
        # Strip optional data-URI prefix
        if "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
    else:
        raw = data

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data.")
    return img


def _read_upload(file_storage) -> np.ndarray:  # noqa: ANN001
    """Read a Werkzeug FileStorage object into a BGR numpy array."""
    raw = file_storage.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image file.")
    return img


# ── Route registration ──────────────────────────────────────────────────────


def _register_routes(app: Flask) -> None:  # noqa: C901
    """Register all API routes on the given Flask app."""

    # ── POST /api/analyze ────────────────────────────────────────────
    @app.route("/api/analyze", methods=["POST"])
    def analyze():  # noqa: ANN202
        """Analyse a single frame: detect, recognise, predict, score."""
        global frame_count

        try:
            # Accept JSON (base64) or multipart form upload
            if request.is_json:
                body = request.get_json()
                if not body or "image" not in body:
                    return jsonify({"error": "Missing 'image' field in JSON body."}), 400
                frame = _decode_image(body["image"])
            elif "image" in request.files:
                frame = _read_upload(request.files["image"])
            else:
                return jsonify({"error": "No image provided."}), 400

            frame_count += 1

            # Detect faces
            faces = detector.detect(frame)
            known_embeddings = db.get_all_embeddings()

            results: list[dict] = []
            for face_info in faces:
                bbox = face_info["bbox"]
                face_crop = face_info["face_crop"]

                # Recognise student
                student_id: int | None = None
                student_name: str | None = None
                identity_confidence: float = 0.0
                try:
                    embedding = recognizer.get_embedding(face_info)
                    student_id, similarity = recognizer.identify(
                        embedding, known_embeddings
                    )
                    if student_id is not None:
                        identity_confidence = similarity
                        student = db.get_student(student_id)
                        student_name = student["name"] if student else None
                except ValueError:
                    pass

                is_known = student_id is not None

                # Predict emotion
                prediction = engine.predict(face_crop)

                # Compute attention score (emotion-based)
                emotion_score = scorer.engagement_score([prediction])
                attention_score = emotion_score
                attention_level = scorer.classify_attention(attention_score)

                # Head pose for face-to-face mode
                pose_data = None
                if active_mode == "face-to-face" and head_pose_estimator is not None:
                    pose_result = head_pose_estimator.estimate_pose(frame)
                    if pose_result is not None:
                        yaw, pitch, roll = pose_result
                        pose_score = HeadPoseEstimator.get_pose_score(yaw, pitch)
                        gaze_dir = HeadPoseEstimator.get_gaze_direction(yaw, pitch)
                        pose_data = {
                            "yaw": round(yaw, 2),
                            "pitch": round(pitch, 2),
                            "roll": round(roll, 2),
                            "gaze_direction": gaze_dir,
                            "pose_score": round(pose_score, 2),
                        }
                        # Use hybrid scorer in face-to-face mode
                        attention_score = hybrid_scorer.compute_score(
                            emotion_score, pose_score
                        )
                        attention_level = hybrid_scorer.classify_attention(
                            attention_score
                        )

                # Track student and detect anomalies
                anomaly_detected = False
                if is_known and tracker is not None:
                    tracker.update(
                        student_id=str(student_id),
                        emotion=prediction["class"],
                        confidence=prediction["confidence"],
                        attention_score=attention_score,
                        attention_level=attention_level,
                    )
                    timeline = tracker.get_timeline(str(student_id))
                    score_history = [t["score"] for t in timeline]
                    anomaly_detected = scorer.detect_anomaly(score_history)

                # Log to database if a session is active and the student is known
                if active_session_id is not None and student_id is not None:
                    db.log_attention(
                        student_id=student_id,
                        session_id=active_session_id,
                        emotion=prediction["class"],
                        confidence=prediction["confidence"],
                        attention_score=attention_score,
                        attention_level=attention_level,
                    )

                result_entry = {
                    "student_id": student_id,
                    "name": student_name,
                    "is_known": is_known,
                    "identity_confidence": round(identity_confidence, 4),
                    "emotion": prediction["class"],
                    "confidence": round(prediction["confidence"], 4),
                    "attention_score": round(attention_score, 4),
                    "attention_level": attention_level,
                    "anomaly_detected": anomaly_detected,
                    "bbox": bbox,
                }
                if pose_data is not None:
                    result_entry["pose"] = pose_data

                results.append(result_entry)

            return jsonify(
                {
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "frame_count": frame_count,
                }
            )

        except Exception as exc:
            logger.exception("Error in /api/analyze")
            return jsonify({"error": str(exc)}), 500

    # ── GET /api/students ────────────────────────────────────────────
    @app.route("/api/students", methods=["GET"])
    def list_students():  # noqa: ANN202
        """Return all registered students."""
        try:
            students = db.list_students()
            return jsonify({"students": students})
        except Exception as exc:
            logger.exception("Error in /api/students")
            return jsonify({"error": str(exc)}), 500

    # ── POST /api/students/register ──────────────────────────────────
    @app.route("/api/students/register", methods=["POST"])
    def register_student():  # noqa: ANN202
        """Register a new student with face images."""
        try:
            name = request.form.get("name")
            if not name:
                return jsonify({"error": "Missing 'name' field."}), 400

            email = request.form.get("email")
            images = request.files.getlist("images")
            if not images:
                return jsonify({"error": "No images provided."}), 400

            student_id = db.add_student(name, email)

            embeddings_saved = 0
            for img_file in images:
                try:
                    frame = _read_upload(img_file)
                except ValueError:
                    continue

                faces = detector.detect(frame)
                if not faces:
                    continue

                # Take the largest face
                largest = max(
                    faces,
                    key=lambda f: (
                        (f["bbox"][2] - f["bbox"][0])
                        * (f["bbox"][3] - f["bbox"][1])
                    ),
                )
                try:
                    embedding = recognizer.get_embedding(largest)
                    db.save_embedding(student_id, embedding)
                    embeddings_saved += 1
                except ValueError:
                    continue

            if embeddings_saved == 0:
                return jsonify(
                    {
                        "error": "No face could be detected in the provided images.",
                        "student_id": student_id,
                    }
                ), 400

            return jsonify(
                {
                    "student_id": student_id,
                    "name": name,
                    "message": f"Student registered with {embeddings_saved} embedding(s).",
                }
            )

        except Exception as exc:
            logger.exception("Error in /api/students/register")
            return jsonify({"error": str(exc)}), 500

    # ── GET /api/students/<id>/stats ─────────────────────────────────
    @app.route("/api/students/<int:student_id>/stats", methods=["GET"])
    def student_stats(student_id: int):  # noqa: ANN202
        """Return attention stats for a student."""
        try:
            student = db.get_student(student_id)
            if student is None:
                return jsonify({"error": "Student not found."}), 404

            session_id = request.args.get("session_id", type=int)
            stats = db.get_student_stats(student_id, session_id)
            return jsonify({"student_id": student_id, **stats})
        except Exception as exc:
            logger.exception("Error in /api/students/<id>/stats")
            return jsonify({"error": str(exc)}), 500

    # ── POST /api/sessions/start ─────────────────────────────────────
    @app.route("/api/sessions/start", methods=["POST"])
    def start_session():  # noqa: ANN202
        """Create and activate a new session."""
        global active_session_id, active_mode
        try:
            body = request.get_json()
            if not body or "name" not in body:
                return jsonify({"error": "Missing 'name' field."}), 400

            name = body["name"]
            mode = body.get("mode", "online")
            if mode not in ("online", "face-to-face"):
                return jsonify({"error": "Invalid mode. Use 'online' or 'face-to-face'."}), 400

            session_id = db.create_session(name, mode)
            active_session_id = session_id
            active_mode = mode

            return jsonify({"session_id": session_id, "name": name, "mode": mode})
        except Exception as exc:
            logger.exception("Error in /api/sessions/start")
            return jsonify({"error": str(exc)}), 500

    # ── POST /api/sessions/<id>/stop ─────────────────────────────────
    @app.route("/api/sessions/<int:session_id>/stop", methods=["POST"])
    def stop_session(session_id: int):  # noqa: ANN202
        """Stop an active session."""
        global active_session_id, active_mode
        try:
            db.end_session(session_id)
            if active_session_id == session_id:
                active_session_id = None
                active_mode = "online"
                if tracker is not None:
                    tracker.reset()

            return jsonify(
                {"session_id": session_id, "message": "Session stopped."}
            )
        except Exception as exc:
            logger.exception("Error in /api/sessions/<id>/stop")
            return jsonify({"error": str(exc)}), 500

    # ── GET /api/sessions ────────────────────────────────────────────
    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():  # noqa: ANN202
        """Return all sessions."""
        try:
            rows = db.conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC"
            ).fetchall()
            sessions = [dict(r) for r in rows]
            return jsonify({"sessions": sessions})
        except Exception as exc:
            logger.exception("Error in /api/sessions")
            return jsonify({"error": str(exc)}), 500

    # ── GET /api/sessions/<id>/stats ─────────────────────────────────
    @app.route("/api/sessions/<int:session_id>/stats", methods=["GET"])
    def session_stats(session_id: int):  # noqa: ANN202
        """Return stats for a session."""
        try:
            stats = db.get_session_stats(session_id)
            return jsonify({"session_id": session_id, **stats})
        except Exception as exc:
            logger.exception("Error in /api/sessions/<id>/stats")
            return jsonify({"error": str(exc)}), 500

    # ── GET /api/export/excel ────────────────────────────────────────
    @app.route("/api/export/excel", methods=["GET"])
    def export_excel():  # noqa: ANN202
        """Export attention data to an Excel file and return it as a download."""
        try:
            session_id = request.args.get("session_id", type=int)

            with tempfile.NamedTemporaryFile(
                suffix=".xlsx", delete=False
            ) as tmp:
                tmp_path = tmp.name

            db.export_to_excel(tmp_path, session_id)

            filename = "attention_report"
            if session_id is not None:
                filename += f"_session_{session_id}"
            filename += ".xlsx"

            return send_file(
                tmp_path,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name=filename,
            )
        except Exception as exc:
            logger.exception("Error in /api/export/excel")
            return jsonify({"error": str(exc)}), 500

    # ── Dashboard ────────────────────────────────────────────────────
    @app.route("/dashboard")
    def dashboard():  # noqa: ANN202
        """Serve the dashboard HTML page."""
        return render_template("dashboard.html")

    @app.route("/")
    def index():  # noqa: ANN202
        """Redirect root to the dashboard."""
        return redirect("/dashboard")
