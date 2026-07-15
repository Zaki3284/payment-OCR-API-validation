"""Pure enrichment of historical screening with verified OCR fields."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from payment_status import PaymentStatusError, enrich_screening_with_payment_status


MIN_RECIPIENT_CONFIDENCE = 0.65


class PaymentOcrEnrichmentError(ValueError):
    pass


def _recipient_confidence(raw_ocr: dict) -> float | None:
    confidence = raw_ocr.get("confidence", {})
    value = confidence.get("recipient") if isinstance(confidence, dict) else None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        return None
    return float(parsed)


def enrich_screening_with_ocr_fields(
    raw_ocr: object,
    screening: object,
) -> dict[str, object]:
    try:
        result = enrich_screening_with_payment_status(raw_ocr, screening)
    except PaymentStatusError as exc:
        raise PaymentOcrEnrichmentError(str(exc)) from None

    # The status layer has already validated both inputs and returned a deep copy.
    recipient_value = raw_ocr.get("recipient")
    recipient = (" ".join(recipient_value.split())
                 if isinstance(recipient_value, str) and recipient_value.strip() else None)
    confidence = _recipient_confidence(raw_ocr)
    verified = recipient is not None and confidence is not None and confidence >= MIN_RECIPIENT_CONFIDENCE

    result["data"]["recipient"] = recipient if verified else None
    result["data"]["recipient_confidence"] = confidence
    if recipient is not None and not verified:
        issue = {"field": "recipient", "code": "low_ocr_confidence"}
        if not any(existing.get("field") == "recipient"
                   and existing.get("code") == "low_ocr_confidence"
                   for existing in result["issues"]):
            result["issues"].append(issue)
    return result
