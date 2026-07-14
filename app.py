"""
KYC REST API — production routes for Flutter (JSON + base64 only).

Run:  python app.py
"""

from __future__ import annotations

import time

from flask import Flask, jsonify, request

try:
    from flask_cors import CORS
except ImportError:
    def CORS(app, *args, **kwargs):  # noqa: ARG001
        return app

import config
from api.cin_service import extract_cin_from_image, normalize_engine
from api.utils import json_image
from face_utils import compare_faces
from kyc_demo import _head_pose, _make_detector, detect_face

API_SPEC = {
    "base_url": "http://{host}:{port}",
    "content_type": "application/json",
    "endpoints": [
        {
            "method": "GET",
            "path": "/health",
            "description": "Health check",
            "body": None,
        },
        {
            "method": "POST",
            "path": "/extract-cin",
            "description": "Extract Mauritanian CIN fields from card photo",
            "body": {
                "image_base64": "string (required) — JPEG/PNG base64",
                "engine": "string (optional) — auto | offline | gemini (default: auto)",
            },
        },
        {
            "method": "POST",
            "path": "/compare-faces",
            "description": "Compare selfie with face on CIN image",
            "body": {
                "selfie_base64": "string (required)",
                "cin_image_base64": "string (required)",
                "threshold": "float (optional, default 0.6)",
            },
        },
        {
            "method": "POST",
            "path": "/verify-kyc",
            "description": "Extract CIN + compare faces in one call",
            "body": {
                "selfie_base64": "string (required)",
                "cin_image_base64": "string (required)",
                "engine": "string (optional, default: auto)",
                "threshold": "float (optional, default 0.6)",
            },
        },
        {
            "method": "POST",
            "path": "/face/pose",
            "description": "Head pose for liveness (yaw/pitch)",
            "body": {
                "image_base64": "string (required) — selfie frame",
            },
        },
    ],
}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    # Register Gemini API routes
    from api.routes_gemini import gemini_bp
    app.register_blueprint(gemini_bp)

    @app.errorhandler(413)
    def too_large(_e):
        return _err("file too large (max 20 MB)", 413)

    @app.get("/")
    def api_spec():
        spec = dict(API_SPEC)
        spec["base_url"] = f"http://{config.API_HOST}:{config.API_PORT}"
        return jsonify(spec)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/extract-cin")
    def extract_cin():
        data = request.get_json(silent=True)
        if not data:
            return _err("JSON body required", 400)
        try:
            engine = normalize_engine(data.get("engine", "auto"))
            img = json_image(data, "image_base64")
        except ValueError as exc:
            return _err(str(exc), 400)
        try:
            out = extract_cin_from_image(img, engine=engine, use_ml=False)
        except Exception as exc:
            return _err(str(exc), 500)
        return jsonify({"success": True, **out})

    @app.post("/compare-faces")
    def compare_faces_route():
        data = request.get_json(silent=True)
        if not data:
            return _err("JSON body required", 400)
        try:
            selfie = json_image(data, "selfie_base64")
            cin_img = json_image(data, "cin_image_base64")
            threshold = float(data.get("threshold", config.FACE_MATCH_THRESHOLD))
        except (ValueError, TypeError) as exc:
            return _err(str(exc), 400)
        t0 = time.time()
        result = compare_faces(selfie, cin_img, threshold)
        result["processing_time_ms"] = int((time.time() - t0) * 1000)
        return jsonify(result)

    @app.post("/verify-kyc")
    def verify_kyc():
        data = request.get_json(silent=True)
        if not data:
            return _err("JSON body required", 400)
        try:
            selfie = json_image(data, "selfie_base64")
            cin_img = json_image(data, "cin_image_base64")
            engine = normalize_engine(data.get("engine", "auto"))
            threshold = float(data.get("threshold", config.FACE_MATCH_THRESHOLD))
        except (ValueError, TypeError) as exc:
            return _err(str(exc), 400)
        t0 = time.time()
        cin_out = extract_cin_from_image(cin_img, engine=engine, use_ml=False)
        fields = cin_out["data"]
        face = compare_faces(selfie, cin_img, threshold)
        verified = all(fields.values()) and face.get("match", False)
        return jsonify({
            "success": True,
            "verified": verified,
            "cin_data": {
                "data": fields,
                "complete": all(fields.values()),
                "blur_score": cin_out["blur_score"],
                "card_detected": cin_out["card_detected"],
            },
            "face_match": {
                "match": face.get("match", False),
                "score": face.get("score", 0.0),
                "confidence": face.get("confidence", "LOW"),
            },
            "processing_time_ms": int((time.time() - t0) * 1000),
        })

    @app.post("/face/pose")
    def face_pose():
        data = request.get_json(silent=True)
        if not data:
            return _err("JSON body required", 400)
        try:
            img = json_image(data, "image_base64")
        except ValueError as exc:
            return _err(str(exc), 400)
        h, w = img.shape[:2]
        detector = _make_detector(w, h)
        face = detect_face(detector, img)
        if face is None:
            return _err("no face detected", 422)
        bbox, landmarks, score = face
        pose = _head_pose(landmarks, w, h)
        if pose is None:
            return _err("could not estimate head pose", 422)
        pitch, yaw = pose
        return jsonify({
            "success": True,
            "face_score": round(score, 3),
            "pose": {"pitch_deg": round(pitch, 2), "yaw_deg": round(yaw, 2)},
            "thresholds": {"yaw_left": -12.0, "yaw_right": 12.0, "pitch_up": 9.0},
        })

    return app


def _err(message: str, code: int):
    return jsonify({"success": False, "error": message}), code
