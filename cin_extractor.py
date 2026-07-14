"""
CIN Data Extractor using Gemini API
Extracts all fields from Mauritanian ID cards
"""

import json
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Optional
import config

# Import Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install with: pip install google-genai")


def to_jpeg_bytes(image) -> bytes:
    """Convert image to JPEG bytes"""
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Failed to encode image")
        return buf.tobytes()
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def extract_cin_fields(image) -> dict[str, Optional[str]]:
    """
    Extract CIN fields from image using Gemini.

    Args:
        image: Image as numpy array, bytes, or file path

    Returns:
        Dict with CIN fields (nni, prenom, prenom_pere, nom_famille,
        date_naissance, lieu_naissance, sexe)
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed")

    # Convert image to bytes
    img_bytes = to_jpeg_bytes(image)

    # Initialize Gemini client with API key
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Prompt for CIN extraction
    prompt = """Extract data from this Mauritanian National Identity Card (CIN).

LAYOUT: Left = portrait photo. CENTER = French labels + values (USE THIS ONLY).
Right = Arabic (IGNORE). Bottom-right may show a faint duplicate portrait (IGNORE).

Read ONLY the French center column (green labels, black values).

Extract these 7 fields EXACTLY as printed:
1. nni: 10-digit number under "Numero National d'Identification" / NNI (NOT the date)
2. prenom: Name under "Prenom" or "Given name"
3. prenom_pere: Name under "Prenom de pere" or "Father's given name"
4. nom_famille: Full surname under "Nom de famille" / "Surname" (keep all words, e.g. "Mohamed El Kory")
5. sexe: Single letter "M" or "F" from the French "Sexe" row only (not Arabic ذكر/أنثى)
6. date_naissance: Date under "Date de naissance" (keep exact format, e.g. "01 Fév/Feb 2000")
7. lieu_naissance: Place under "Lieu de naissance"

Rules:
- Extract text EXACTLY as printed (preserve accents, apostrophes)
- Do NOT translate or correct spelling
- Do NOT mix digits from the birth date into the NNI
- If field is missing/unclear, return null
- For NNI, extract exactly 10 digits with no spaces

Return ONLY valid JSON:
{
  "nni": "1234567890" or null,
  "prenom": "Name" or null,
  "prenom_pere": "Name" or null,
  "nom_famille": "Name" or null,
  "sexe": "M" or "F" or null,
  "date_naissance": "DD Month YYYY" or null,
  "lieu_naissance": "Place" or null
}"""

    # Call Gemini API
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    # Parse response
    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("Empty response from Gemini")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Gemini: {raw_text[:200]}")

    # Normalize output
    result = {}
    for key in ["nni", "prenom", "prenom_pere", "nom_famille",
                "date_naissance", "lieu_naissance", "sexe"]:
        value = data.get(key)
        result[key] = value if value not in ("", "null", None) else None

    return _validate_cin_fields(result)


def _validate_cin_fields(fields: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    """Drop obvious hallucinations (wrong NNI length, invalid sexe)."""
    out = dict(fields)
    nni = out.get("nni")
    if nni:
        digits = re.sub(r"\D", "", str(nni))
        out["nni"] = digits if len(digits) == 10 else None
    sexe = out.get("sexe")
    if sexe:
        s = str(sexe).strip().upper()
        out["sexe"] = s if s in ("M", "F") else None
    return out


def estimate_blur(image: np.ndarray) -> float:
    """
    Estimate image blur using Laplacian variance.
    Higher score = sharper image.

    Args:
        image: Image as numpy array

    Returns:
        Blur score (>300 = good, 200-300 = ok, <200 = poor)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
