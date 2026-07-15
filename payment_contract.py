"""Strict input contract for payment-proof validation requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class PaymentValidationRequest:
    image_base64: str
    expected_provider: str
    expected_amount: Decimal
    expected_currency: str
    expected_sender_phone: str | None
    expected_recipient: str | None
    max_age_hours: float


class PaymentContractError(ValueError):
    def __init__(self, field: str, code: str, message: str):
        super().__init__(message)
        self.field = field
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code, "message": self.message}


def _error(field: str, code: str, message: str) -> None:
    raise PaymentContractError(field=field, code=code, message=message)


def _required(body: dict, field: str):
    if field not in body:
        _error(field, "missing", f"missing {field}")
    return body[field]


def _parse_amount(value: object) -> Decimal:
    field = "expected_amount"
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        _error(field, "invalid_type", f"{field} must be numeric")
    if isinstance(value, str):
        text = re.sub(r"(?i)\bMRU\b", "", value).replace(" ", "").strip()
        if not text:
            _error(field, "invalid_format", f"invalid {field} format")
    else:
        text = str(value)
    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        _error(field, "invalid_format", f"invalid {field} format")
    if not amount.is_finite():
        _error(field, "invalid_format", f"invalid {field} format")
    if amount <= 0:
        _error(field, "must_be_positive", f"{field} must be greater than zero")
    try:
        return amount.quantize(Decimal("0.01"))
    except InvalidOperation:
        _error(field, "invalid_format", f"invalid {field} format")


def _parse_phone(value: object) -> str | None:
    field = "expected_sender_phone"
    if value is None:
        return None
    if not isinstance(value, str):
        _error(field, "invalid_type", f"{field} must be a string")
    if not value.strip():
        _error(field, "empty", f"{field} must not be empty")
    compact = re.sub(r"[\s()\-]", "", value)
    if re.fullmatch(r"\d{8}", compact):
        return "+222" + compact
    if re.fullmatch(r"\+222\d{8}", compact):
        return compact
    if re.fullmatch(r"00222\d{8}", compact):
        return "+222" + compact[5:]
    _error(field, "invalid_format", f"invalid {field} format")


def _parse_max_age(value: object) -> float:
    field = "max_age_hours"
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        _error(field, "invalid_type", f"{field} must be numeric")
    if isinstance(value, str) and not value.strip():
        _error(field, "invalid_format", f"invalid {field} format")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        _error(field, "invalid_format", f"invalid {field} format")
    if not number.is_finite():
        _error(field, "invalid_format", f"invalid {field} format")
    if number < Decimal("0.5") or number > Decimal("72"):
        _error(field, "out_of_range", f"{field} must be between 0.5 and 72")
    return float(number)


def parse_payment_validation_request(body: object) -> PaymentValidationRequest:
    if not isinstance(body, dict):
        _error("body", "invalid_type", "body must be an object")

    image = _required(body, "image_base64")
    if not isinstance(image, str):
        _error("image_base64", "invalid_type", "image_base64 must be a string")
    if not image.strip():
        _error("image_base64", "empty", "image_base64 must not be empty")

    provider_value = _required(body, "expected_provider")
    if not isinstance(provider_value, str):
        _error("expected_provider", "invalid_type", "expected_provider must be a string")
    provider = provider_value.strip().lower()
    if not provider:
        _error("expected_provider", "empty", "expected_provider must not be empty")
    provider = {"sadad": "sedad"}.get(provider, provider)
    if provider not in {"bankily", "masrivi", "sedad", "click"}:
        _error("expected_provider", "unsupported", "unsupported expected_provider")

    amount = _parse_amount(_required(body, "expected_amount"))

    currency_value = body.get("expected_currency", "MRU")
    if not isinstance(currency_value, str):
        _error("expected_currency", "invalid_type", "expected_currency must be a string")
    currency = currency_value.strip().upper()
    if not currency:
        _error("expected_currency", "empty", "expected_currency must not be empty")
    if currency != "MRU":
        _error("expected_currency", "unsupported", "unsupported expected_currency")

    phone = _parse_phone(body.get("expected_sender_phone"))

    recipient_value = body.get("expected_recipient")
    if recipient_value is None:
        recipient = None
    else:
        if not isinstance(recipient_value, str):
            _error("expected_recipient", "invalid_type", "expected_recipient must be a string")
        recipient = " ".join(recipient_value.split())
        if not recipient:
            _error("expected_recipient", "empty", "expected_recipient must not be empty")

    max_age = _parse_max_age(body.get("max_age_hours", 6.0))
    return PaymentValidationRequest(
        image_base64=image,
        expected_provider=provider,
        expected_amount=amount,
        expected_currency=currency,
        expected_sender_phone=phone,
        expected_recipient=recipient,
        max_age_hours=max_age,
    )
