"""Small Flask API for payment screenshot OCR. Run: python payment_app.py"""

from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request

from payment_ocr import extract_with_gemini, image_quality, validate_payment
from payment_contract import PaymentContractError, parse_payment_validation_request
from payment_ocr_enrichment import (
    PaymentOcrEnrichmentError,
    enrich_screening_with_ocr_fields,
)
from payment_validation_service import (
    PaymentValidationServiceError,
    evaluate_payment_validation,
)
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

    @app.post("/api/v1/payment-proofs/validate")
    def validate_payment_proof():
        body = request.get_json(silent=True)
        try:
            contract = parse_payment_validation_request(body)
        except PaymentContractError as exc:
            return jsonify({"success": False, "error": exc.to_dict()}), 400

        started = time.monotonic()
        try:
            image = json_image({"image_base64": contract.image_base64}, "image_base64")
        except ValueError as exc:
            return jsonify({"success": False, "error": {
                "field": "image_base64", "code": "invalid_image", "message": str(exc)}}), 400

        quality = image_quality(image)
        if not quality["acceptable"]:
            return jsonify({
                "success": False,
                "error": {"field": "image_base64", "code": "quality_too_low",
                          "message": "image quality too low; retake capture"},
                "quality": quality,
            }), 422

        try:
            raw_ocr = extract_with_gemini(image)
        except RuntimeError as exc:
            return jsonify({"success": False, "error": {
                "field": "ocr", "code": "service_unavailable", "message": str(exc)}}), 503
        if not isinstance(raw_ocr, dict):
            return jsonify({"success": False, "error": {
                "field": "ocr", "code": "invalid_response",
                "message": "OCR returned an invalid object"}}), 502

        screening = validate_payment(raw_ocr, max_age_hours=contract.max_age_hours)
        try:
            enriched_screening = enrich_screening_with_ocr_fields(raw_ocr, screening)
            validation = evaluate_payment_validation(contract, enriched_screening)
        except (PaymentOcrEnrichmentError, PaymentValidationServiceError):
            return jsonify({"success": False, "error": {
                "field": "validation", "code": "internal_error",
                "message": "payment validation pipeline failed"}}), 500

        return jsonify({
            "success": True,
            "api_version": "v1",
            "decision": validation["decision"],
            "quality": quality,
            "ocr": raw_ocr,
            "validation": validation,
            "processing_time_ms": int((time.monotonic() - started) * 1000),
        })
    return app


if __name__ == "__main__":
    create_app().run(host=os.getenv("API_HOST", "0.0.0.0"),
                     port=int(os.getenv("API_PORT", "5000")), debug=False)
