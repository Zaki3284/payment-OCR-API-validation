"""
Dedicated Gemini API for CIN Data Extraction
Pure Gemini-based extraction with hardcoded API key
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import config
from cin_extractor import extract_cin_fields, estimate_blur
from kyc_demo import prepare_gemini_image
from api.utils import blur_metrics


def extract_cin_with_gemini(img: np.ndarray) -> dict[str, Any]:
    """
    Extract CIN fields using Gemini Vision API.

    Args:
        img: CIN card image as numpy array

    Returns:
        Dictionary with:
        - success: bool
        - data: dict of extracted CIN fields
        - blur_score: float
        - quality_status: str
        - processing_time_ms: int
        - engine: str (always "gemini")
    """
    t0 = time.time()

    try:
        # Get blur metrics
        blur = estimate_blur(img)

        # Quality check
        if blur < 100:
            quality_status = "VERY_POOR"
            quality_message = "Image too blurry - retake photo"
        elif blur < 200:
            quality_status = "POOR"
            quality_message = "Low quality - some fields may be missing"
        elif blur < 300:
            quality_status = "ACCEPTABLE"
            quality_message = "Acceptable quality"
        else:
            quality_status = "GOOD"
            quality_message = "Good quality"

        # Extract fields with Gemini (French column crop — same as CLI)
        fields = extract_cin_fields(prepare_gemini_image(img))

        # Check completeness
        complete = all(fields.values())
        missing_fields = [k for k, v in fields.items() if not v]

        processing_time = int((time.time() - t0) * 1000)

        return {
            "success": True,
            "data": fields,
            "blur_score": float(blur),
            "quality_status": quality_status,
            "quality_message": quality_message,
            "complete": complete,
            "missing_fields": missing_fields if not complete else [],
            "processing_time_ms": processing_time,
            "engine": "gemini",
            "model": config.GEMINI_MODEL,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": int((time.time() - t0) * 1000),
            "engine": "gemini",
        }


def batch_extract_cin(images: list[np.ndarray]) -> list[dict[str, Any]]:
    """
    Extract CIN fields from multiple images.

    Args:
        images: List of CIN card images

    Returns:
        List of extraction results
    """
    results = []
    for idx, img in enumerate(images):
        result = extract_cin_with_gemini(img)
        result["image_index"] = idx
        results.append(result)

    return results


def verify_cin_data(fields: dict[str, str | None]) -> dict[str, Any]:
    """
    Verify extracted CIN data for completeness and format.

    Args:
        fields: Extracted CIN fields

    Returns:
        Verification result with issues if any
    """
    issues = []

    # Check NNI format (10 digits)
    if fields.get("nni"):
        if not fields["nni"].isdigit() or len(fields["nni"]) != 10:
            issues.append({
                "field": "nni",
                "issue": "NNI must be exactly 10 digits",
                "value": fields["nni"]
            })

    # Check sex format (M or F)
    if fields.get("sexe"):
        if fields["sexe"] not in ("M", "F"):
            issues.append({
                "field": "sexe",
                "issue": "Sexe must be M or F",
                "value": fields["sexe"]
            })

    # Check required fields
    required = ["nni", "prenom", "prenom_pere", "nom_famille",
                "date_naissance", "lieu_naissance", "sexe"]

    missing = [f for f in required if not fields.get(f)]

    return {
        "valid": len(issues) == 0 and len(missing) == 0,
        "issues": issues,
        "missing_fields": missing,
        "completeness": (len(required) - len(missing)) / len(required) * 100
    }
