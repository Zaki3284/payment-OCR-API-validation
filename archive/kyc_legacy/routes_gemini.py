"""
Dedicated Gemini API Routes
Endpoints specifically for Gemini-based CIN extraction
"""

from flask import Blueprint, jsonify, request

from api.gemini_api import extract_cin_with_gemini, batch_extract_cin, verify_cin_data
from api.utils import json_image

# Create Blueprint for Gemini routes
gemini_bp = Blueprint('gemini', __name__, url_prefix='/gemini')


@gemini_bp.route('/extract', methods=['POST'])
def gemini_extract():
    """
    Extract CIN data using Gemini Vision API.

    POST /gemini/extract
    Body:
    {
        "image_base64": "base64_encoded_image"
    }

    Response:
    {
        "success": true,
        "data": {
            "nni": "1234567890",
            "prenom": "Name",
            ...
        },
        "blur_score": 450.2,
        "quality_status": "GOOD",
        "complete": true,
        "processing_time_ms": 1234,
        "engine": "gemini"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "JSON body required"
        }), 400

    try:
        img = json_image(data, "image_base64")
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    result = extract_cin_with_gemini(img)

    if not result["success"]:
        return jsonify(result), 500

    return jsonify(result)


@gemini_bp.route('/extract-batch', methods=['POST'])
def gemini_extract_batch():
    """
    Extract CIN data from multiple images using Gemini.

    POST /gemini/extract-batch
    Body:
    {
        "images": [
            "base64_image_1",
            "base64_image_2",
            ...
        ]
    }

    Response:
    {
        "success": true,
        "results": [
            { "image_index": 0, "data": {...}, ... },
            { "image_index": 1, "data": {...}, ... }
        ],
        "total_images": 2,
        "successful": 2,
        "failed": 0
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "JSON body required"
        }), 400

    images_b64 = data.get("images", [])
    if not images_b64 or not isinstance(images_b64, list):
        return jsonify({
            "success": False,
            "error": "images array required"
        }), 400

    # Decode all images
    try:
        from api.utils import decode_base64_image
        images = [decode_base64_image(img_b64) for img_b64 in images_b64]
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid image in batch: {e}"
        }), 400

    # Process batch
    results = batch_extract_cin(images)

    # Count successes
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful

    return jsonify({
        "success": True,
        "results": results,
        "total_images": len(results),
        "successful": successful,
        "failed": failed
    })


@gemini_bp.route('/verify', methods=['POST'])
def gemini_verify():
    """
    Verify extracted CIN data for completeness and format.

    POST /gemini/verify
    Body:
    {
        "data": {
            "nni": "1234567890",
            "prenom": "Name",
            ...
        }
    }

    Response:
    {
        "valid": true,
        "issues": [],
        "missing_fields": [],
        "completeness": 100.0
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "JSON body required"
        }), 400

    cin_data = data.get("data")
    if not cin_data:
        return jsonify({
            "success": False,
            "error": "data field required"
        }), 400

    result = verify_cin_data(cin_data)
    return jsonify(result)


@gemini_bp.route('/extract-and-verify', methods=['POST'])
def gemini_extract_and_verify():
    """
    Extract CIN data and verify it in one call.

    POST /gemini/extract-and-verify
    Body:
    {
        "image_base64": "base64_encoded_image"
    }

    Response:
    {
        "success": true,
        "extraction": {
            "data": {...},
            "blur_score": 450.2,
            ...
        },
        "verification": {
            "valid": true,
            "issues": [],
            ...
        }
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "JSON body required"
        }), 400

    try:
        img = json_image(data, "image_base64")
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    # Extract
    extraction = extract_cin_with_gemini(img)

    if not extraction["success"]:
        return jsonify({
            "success": False,
            "error": extraction.get("error", "Extraction failed")
        }), 500

    # Verify
    verification = verify_cin_data(extraction["data"])

    return jsonify({
        "success": True,
        "extraction": extraction,
        "verification": verification,
        "overall_valid": verification["valid"]
    })


@gemini_bp.route('/health', methods=['GET'])
def gemini_health():
    """
    Check if Gemini API is configured and working.

    GET /gemini/health

    Response:
    {
        "status": "ok",
        "gemini_configured": true,
        "api_key_present": true,
        "model": "gemini-2.5-flash"
    }
    """
    import config

    return jsonify({
        "status": "ok",
        "gemini_configured": True,
        "api_key_present": bool(config.GEMINI_API_KEY),
        "model": config.GEMINI_MODEL,
        "project": config.PROJECT_NUMBER
    })
