"""Pure normalization and screening enrichment for visible payment status."""

from __future__ import annotations

import copy
import unicodedata
from decimal import Decimal, InvalidOperation


MIN_STATUS_CONFIDENCE = 0.65


class PaymentStatusError(ValueError):
    pass


_PHRASES = {
    "successful": {
        "success", "successful", "succeeded", "completed", "paid",
        "payment successful", "transaction successful", "payment completed",
        "transaction completed", "succes", "reussi", "reussie", "effectue",
        "effectuee", "paiement reussi", "transaction reussie", "operation reussie",
        "paiement effectue", "transaction effectuee", "تم بنجاح", "تمت بنجاح",
        "ناجح", "ناجحة", "العملية ناجحة", "تمت العملية بنجاح",
    },
    "failed": {
        "failed", "failure", "unsuccessful", "payment failed", "transaction failed",
        "echec", "echoue", "echouee", "paiement echoue", "transaction echouee",
        "operation echouee", "فشل", "فشلت", "فشل الدفع", "العملية فشلت",
    },
    "cancelled": {
        "cancelled", "canceled", "payment cancelled", "payment canceled",
        "transaction cancelled", "transaction canceled", "annule", "annulee",
        "paiement annule", "transaction annulee", "operation annulee", "ملغي",
        "ملغاة", "تم الالغاء", "تم الغاء العملية",
    },
    "rejected": {
        "rejected", "declined", "payment rejected", "transaction rejected",
        "payment declined", "transaction declined", "rejete", "rejetee", "refuse",
        "refusee", "paiement refuse", "transaction rejetee", "operation rejetee",
        "مرفوض", "مرفوضة", "تم الرفض",
    },
}


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip()).casefold()
    characters = []
    for char in decomposed:
        category = unicodedata.category(char)
        if category.startswith("M"):
            continue
        characters.append(" " if category.startswith(("P", "Z")) else char)
    return " ".join("".join(characters).split())


def normalize_payment_status(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = _normalize_text(value)
    for canonical, phrases in _PHRASES.items():
        if normalized in phrases:
            return canonical
    return "unknown"


def _confidence(raw_ocr: dict) -> float | None:
    confidence = raw_ocr.get("confidence", {})
    value = confidence.get("status") if isinstance(confidence, dict) else None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        return None
    return float(parsed)


def enrich_screening_with_payment_status(
    raw_ocr: object,
    screening: object,
) -> dict[str, object]:
    if not isinstance(raw_ocr, dict):
        raise PaymentStatusError("raw_ocr must be an object")
    if not isinstance(screening, dict):
        raise PaymentStatusError("screening must be an object")
    if not isinstance(screening.get("data"), dict):
        raise PaymentStatusError("screening data must be an object")
    if not isinstance(screening.get("issues"), list):
        raise PaymentStatusError("screening issues must be a list")
    if any(not isinstance(issue, dict) for issue in screening["issues"]):
        raise PaymentStatusError("invalid screening issue")

    result = copy.deepcopy(screening)
    preferred = raw_ocr.get("status")
    source = (preferred if isinstance(preferred, str) and preferred.strip()
              else raw_ocr.get("status_text"))
    canonical = normalize_payment_status(source)
    confidence = _confidence(raw_ocr)
    verified = canonical != "unknown" and confidence is not None and confidence >= MIN_STATUS_CONFIDENCE

    result["data"]["status"] = canonical if verified else "unknown"
    result["data"]["status_confidence"] = confidence
    if canonical != "unknown" and not verified:
        issue = {"field": "status", "code": "low_ocr_confidence"}
        if not any(existing.get("field") == "status"
                   and existing.get("code") == "low_ocr_confidence"
                   for existing in result["issues"]):
            result["issues"].append(issue)
    return result
