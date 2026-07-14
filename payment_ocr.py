"""OCR and validation for Mauritanian mobile-payment screenshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

PROVIDERS = {"bankily": "Bankily", "masrivi": "Masrivi", "sedad": "Sedad", "click": "Click"}
REQUIRED_FIELDS = ("provider", "reference", "amount", "paid_at", "phone")


def _clean_phone(value: Any) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if digits.startswith("00222"):
        digits = digits[5:]
    elif digits.startswith("222") and len(digits) == 11:
        digits = digits[3:]
    # Mauritanian NSN: 8 digits. Broad first-digit check avoids rejecting new ranges.
    return f"+222{digits}" if len(digits) == 8 and digits[0] in "234" else None


def _clean_provider(value: Any) -> str | None:
    key = re.sub(r"[^a-z]", "", str(value or "").lower())
    aliases = {"bankili": "bankily", "masrvi": "masrivi", "masrivy": "masrivi", "sadad": "sedad"}
    key = aliases.get(key, key)
    return PROVIDERS.get(key)


def _clean_amount(value: Any) -> str | None:
    text = re.sub(r"(?i)\b(MRU|UM|MRo|ouguiya)\b", "", str(value or "")).strip()
    text = text.replace(" ", "").replace(",", ".")
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    if amount <= 0 or amount > Decimal("100000000"):
        return None
    return format(amount.quantize(Decimal("0.01")), "f")


def _parse_date(value: Any, default_tz=timezone.utc) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = None
        for fmt in ("%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                pass
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(timezone.utc)


def image_quality(image: Any) -> dict[str, Any]:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = gray.shape[:2]
    return {"blur_score": round(blur, 1), "width": w, "height": h,
            "acceptable": blur >= 80 and w >= 500 and h >= 500}


def extract_with_gemini(image: Any) -> dict[str, Any]:
    """Read a receipt screenshot. Values not visibly present must be null."""
    try:
        from google import genai
        from google.genai import types
        import cv2
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed") from exc
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError("could not encode image")
    prompt = """Read this Mauritanian payment receipt/screenshot. Supported apps are
Bankily, Masrivi, Sedad (sometimes written Sadad), and Click. Extract only text that
is visibly printed. Never infer or invent a value. `reference` is the transaction
reference/ID/code, not the phone. `paid_at` must be ISO 8601 when date and time are
visible; otherwise null. Amount is numeric MRU without currency. Phone includes the
country code if printed. Return JSON only:
{"provider":null,"reference":null,"amount":null,"currency":"MRU","paid_at":null,
 "phone":null,"status_text":null,"confidence":{"provider":0,"reference":0,
 "amount":0,"paid_at":0,"phone":0}}. Confidence values are numbers from 0 to 1."""
    response = genai.Client(api_key=api_key).models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    try:
        return json.loads(response.text or "")
    except json.JSONDecodeError as exc:
        raise RuntimeError("OCR returned invalid JSON") from exc


def validate_payment(raw: dict[str, Any], *, expected_code: str | None = None,
                     max_age_hours: float = 6, now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    provider = _clean_provider(raw.get("provider"))
    reference = re.sub(r"[^A-Za-z0-9_-]", "", str(raw.get("reference") or "")) or None
    amount = _clean_amount(raw.get("amount"))
    phone = _clean_phone(raw.get("phone"))
    paid_at = _parse_date(raw.get("paid_at"))
    data = {"provider": provider, "reference": reference, "amount": amount,
            "currency": "MRU", "paid_at": paid_at.isoformat() if paid_at else None, "phone": phone}
    issues: list[dict[str, str]] = []
    for field in REQUIRED_FIELDS:
        if not data[field]:
            issues.append({"field": field, "code": "missing_or_invalid"})
    if reference and len(reference) < 6:
        issues.append({"field": "reference", "code": "too_short"})
    if paid_at:
        age_hours = (now - paid_at).total_seconds() / 3600
        if age_hours < -0.5:
            issues.append({"field": "paid_at", "code": "future_date"})
        elif age_hours > max_age_hours:
            issues.append({"field": "paid_at", "code": "payment_too_old"})
    if expected_code and (not reference or expected_code.casefold() not in reference.casefold()):
        issues.append({"field": "reference", "code": "expected_code_not_found"})
    confidence = raw.get("confidence") if isinstance(raw.get("confidence"), dict) else {}
    for field in REQUIRED_FIELDS:
        try:
            score = float(confidence.get(field, 0))
        except (TypeError, ValueError):
            score = 0
        if data[field] and score < 0.65:
            issues.append({"field": field, "code": "low_ocr_confidence"})
    canonical = "|".join(str(data[k] or "") for k in ("provider", "reference", "amount", "paid_at", "phone"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    return {"data": data, "issues": issues, "screening_passed": not issues,
            "fingerprint": fingerprint,
            "review_status": "pending_admin_review" if not issues else "needs_resubmission"}
