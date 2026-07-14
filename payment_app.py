"""Small Flask API for payment screenshot OCR. Run: python payment_app.py"""

from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request

from payment_ocr import extract_with_gemini, image_quality, validate_payment
from utils import json_image


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))})

    @app.post("/payments/ocr")
    def payment_ocr():
        body = request.get_json(silent=True) or {}
        try:
            image = json_image(body, "image_base64")
            max_age = float(body.get("max_age_hours", 6))
            if not 0.5 <= max_age <= 72:
                raise ValueError("max_age_hours must be between 0.5 and 72")
        except (ValueError, TypeError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        quality = image_quality(image)
        if not quality["acceptable"]:
            return jsonify({"success": False, "error": "image quality too low; retake capture",
                            "quality": quality}), 422
        started = time.monotonic()
        try:
            extracted = extract_with_gemini(image)
        except RuntimeError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        result = validate_payment(extracted, expected_code=body.get("expected_payment_code"),
                                  max_age_hours=max_age)
        return jsonify({"success": True, "quality": quality, "ocr": extracted,
                        "validation": result,
                        "processing_time_ms": int((time.monotonic() - started) * 1000)})

    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("API_HOST", "0.0.0.0"),
                     port=int(os.getenv("API_PORT", "5000")), debug=False)
