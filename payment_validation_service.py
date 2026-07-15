"""Pure comparison service for normalized payment screening results."""

from __future__ import annotations

import copy
import re
from decimal import Decimal, InvalidOperation

from payment_contract import PaymentValidationRequest


class PaymentValidationServiceError(ValueError):
    pass


_PROVIDERS = {"bankily", "masrivi", "sedad", "click"}
_HARD_SCREENING_CODES = {
    "missing_or_invalid", "too_short", "future_date", "payment_too_old",
    "expected_code_not_found",
}


def _provider(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = re.sub(r"[^a-z]", "", value.strip().casefold())
    normalized = {"sadad": "sedad"}.get(normalized, normalized)
    return normalized if normalized in _PROVIDERS else None


def _amount(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value).strip())
        if not amount.is_finite() or amount <= 0:
            return None
        return format(amount.quantize(Decimal("0.01")), "f")
    except (InvalidOperation, ValueError):
        return None


def _check(expected: object, actual: object, *, optional: bool = False) -> dict[str, object]:
    if optional and expected is None:
        status = "not_requested"
    elif actual is None:
        status = "unverified"
    elif actual == expected:
        status = "matched"
    else:
        status = "mismatched"
    return {"status": status, "expected": expected, "actual": actual}


def _comparison_issue(field: str, code: str, check: dict[str, object]) -> dict[str, object]:
    return {"source": "comparison", "field": field, "code": code,
            "expected": check["expected"], "actual": check["actual"]}


def evaluate_payment_validation(
    request: PaymentValidationRequest,
    screening: object,
) -> dict[str, object]:
    if not isinstance(request, PaymentValidationRequest):
        raise PaymentValidationServiceError("request must be a PaymentValidationRequest")
    if not isinstance(screening, dict):
        raise PaymentValidationServiceError("screening must be an object")
    data = screening.get("data")
    if not isinstance(data, dict):
        raise PaymentValidationServiceError("screening data must be an object")
    screening_issues = screening.get("issues")
    if not isinstance(screening_issues, list):
        raise PaymentValidationServiceError("screening issues must be a list")
    for issue in screening_issues:
        if (not isinstance(issue, dict)
                or not isinstance(issue.get("field"), str)
                or not isinstance(issue.get("code"), str)):
            raise PaymentValidationServiceError("invalid screening issue")

    result_issues: list[dict[str, object]] = []
    rejected = False
    manual_review = False
    for issue in screening_issues:
        copied = copy.deepcopy(issue)
        copied["source"] = "screening"
        result_issues.append(copied)
        code = issue["code"]
        if code in _HARD_SCREENING_CODES:
            rejected = True
        else:
            manual_review = True

    expected_amount = format(request.expected_amount, ".2f")
    actual_provider = _provider(data.get("provider"))
    actual_amount = _amount(data.get("amount"))
    currency_value = data.get("currency")
    actual_currency = (currency_value.strip().upper()
                       if isinstance(currency_value, str) and currency_value.strip() else None)
    phone_value = data.get("phone")
    actual_phone = phone_value.strip() if isinstance(phone_value, str) and phone_value.strip() else None
    recipient_value = data.get("recipient")
    actual_recipient_display = (" ".join(recipient_value.split())
                                if isinstance(recipient_value, str) and recipient_value.strip() else None)
    expected_recipient = request.expected_recipient

    status_value = data.get("status")
    actual_status = status_value.strip().casefold() if isinstance(status_value, str) and status_value.strip() else None

    checks = {
        "provider": _check(request.expected_provider, actual_provider),
        "amount": _check(expected_amount, actual_amount),
        "currency": _check(request.expected_currency, actual_currency),
        "sender_phone": _check(request.expected_sender_phone, actual_phone, optional=True),
        "recipient": _check(
            expected_recipient,
            actual_recipient_display,
            optional=True,
        ),
        "payment_status": {
            "status": (
                "matched" if actual_status == "successful"
                else "mismatched" if actual_status in {"failed", "cancelled", "rejected"}
                else "unverified"
            ),
            "expected": "successful",
            "actual": actual_status,
        },
    }
    if expected_recipient is not None and actual_recipient_display is not None:
        checks["recipient"]["status"] = (
            "matched" if expected_recipient.casefold() == actual_recipient_display.casefold()
            else "mismatched"
        )

    codes = {
        "provider": ("provider_mismatch", "provider_unverified"),
        "amount": ("amount_mismatch", "amount_unverified"),
        "currency": ("currency_mismatch", "currency_unverified"),
        "sender_phone": ("sender_phone_mismatch", "sender_phone_unverified"),
        "recipient": ("recipient_mismatch", "recipient_unverified"),
        "payment_status": ("payment_status_not_successful", "payment_status_unverified"),
    }
    issue_fields = {"payment_status": "payment_status"}
    for name, check in checks.items():
        if check["status"] == "mismatched":
            result_issues.append(_comparison_issue(
                issue_fields.get(name, name), codes[name][0], check))
            rejected = True
        elif check["status"] == "unverified":
            result_issues.append(_comparison_issue(
                issue_fields.get(name, name), codes[name][1], check))
            manual_review = True

    fingerprint_value = screening.get("fingerprint")
    fingerprint = (fingerprint_value if isinstance(fingerprint_value, str)
                   and fingerprint_value.strip() else None)
    if fingerprint is None:
        result_issues.append({"source": "service", "field": "fingerprint",
                              "code": "fingerprint_unverified"})
        manual_review = True

    if screening.get("screening_passed") is False and not screening_issues:
        result_issues.append({"source": "service", "field": "screening",
                              "code": "inconsistent_screening_result"})
        manual_review = True

    decision = "rejected" if rejected else "manual_review" if manual_review else "accepted"
    return {"decision": decision, "checks": checks, "issues": result_issues,
            "payment": copy.deepcopy(data), "fingerprint": fingerprint}
