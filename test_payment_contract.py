import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal

from payment_contract import (
    PaymentContractError,
    PaymentValidationRequest,
    parse_payment_validation_request,
)


def minimal(**overrides):
    body = {"image_base64": " image-data ", "expected_provider": "bankily",
            "expected_amount": 1250}
    body.update(overrides)
    return body


class PaymentContractTests(unittest.TestCase):
    def parse(self, **overrides):
        return parse_payment_validation_request(minimal(**overrides))

    def assert_error(self, body, field, code):
        with self.assertRaises(PaymentContractError) as caught:
            parse_payment_validation_request(body)
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_valid_complete_request(self):
        parsed = self.parse(expected_provider=" Sadad ", expected_amount="1 250 MRU",
                            expected_currency=" mru ", expected_sender_phone="+222 22 12 34 56",
                            expected_recipient="  ACPEC   GAS  ", max_age_hours="12")
        self.assertEqual(parsed, PaymentValidationRequest(
            image_base64=" image-data ", expected_provider="sedad",
            expected_amount=Decimal("1250.00"), expected_currency="MRU",
            expected_sender_phone="+22222123456", expected_recipient="ACPEC GAS",
            max_age_hours=12.0))

    def test_valid_minimal_request(self):
        parsed = self.parse()
        self.assertEqual(parsed.expected_currency, "MRU")
        self.assertIsNone(parsed.expected_sender_phone)
        self.assertIsNone(parsed.expected_recipient)
        self.assertEqual(parsed.max_age_hours, 6.0)

    def test_body_is_not_object(self):
        for body in (None, [], "text", 1, True):
            with self.subTest(body=body):
                self.assert_error(body, "body", "invalid_type")

    def test_missing_image(self):
        body = minimal(); del body["image_base64"]
        self.assert_error(body, "image_base64", "missing")

    def test_image_invalid_type(self):
        self.assert_error(minimal(image_base64=1), "image_base64", "invalid_type")

    def test_empty_image(self):
        self.assert_error(minimal(image_base64="  "), "image_base64", "empty")

    def test_missing_provider(self):
        body = minimal(); del body["expected_provider"]
        self.assert_error(body, "expected_provider", "missing")

    def test_provider_invalid_type(self):
        self.assert_error(minimal(expected_provider=1), "expected_provider", "invalid_type")

    def test_empty_provider(self):
        self.assert_error(minimal(expected_provider=" "), "expected_provider", "empty")

    def test_unsupported_provider(self):
        self.assert_error(minimal(expected_provider="other"), "expected_provider", "unsupported")

    def test_provider_alias_sadad(self):
        self.assertEqual(self.parse(expected_provider="sadad").expected_provider, "sedad")

    def test_provider_normalization(self):
        self.assertEqual(self.parse(expected_provider=" BANKILY ").expected_provider, "bankily")

    def test_missing_amount(self):
        body = minimal(); del body["expected_amount"]
        self.assert_error(body, "expected_amount", "missing")

    def test_amount_invalid_type(self):
        self.assert_error(minimal(expected_amount=None), "expected_amount", "invalid_type")

    def test_amount_invalid_format(self):
        self.assert_error(minimal(expected_amount="money"), "expected_amount", "invalid_format")

    def test_amount_zero(self):
        self.assert_error(minimal(expected_amount=0), "expected_amount", "must_be_positive")

    def test_amount_negative(self):
        self.assert_error(minimal(expected_amount=-1), "expected_amount", "must_be_positive")

    def test_amount_integer(self):
        self.assertEqual(self.parse(expected_amount=1250).expected_amount, Decimal("1250.00"))

    def test_amount_decimal_string(self):
        self.assertEqual(self.parse(expected_amount="1250.50").expected_amount, Decimal("1250.50"))

    def test_amount_with_spaces_and_currency(self):
        self.assertEqual(self.parse(expected_amount="1 250 MRU").expected_amount, Decimal("1250.00"))

    def test_amount_is_rounded_to_two_decimal_places(self):
        self.assertEqual(self.parse(expected_amount="1.239").expected_amount, Decimal("1.24"))

    def test_amount_rejects_boolean(self):
        self.assert_error(minimal(expected_amount=True), "expected_amount", "invalid_type")

    def test_amount_rejects_nan(self):
        self.assert_error(minimal(expected_amount=float("nan")), "expected_amount", "invalid_format")

    def test_amount_rejects_infinity(self):
        self.assert_error(minimal(expected_amount=float("inf")), "expected_amount", "invalid_format")

    def test_amount_extreme_integer_returns_contract_error(self):
        self.assert_error(
            minimal(expected_amount="9" * 1000),
            "expected_amount",
            "invalid_format",
        )

    def test_amount_extreme_exponent_returns_contract_error(self):
        self.assert_error(
            minimal(expected_amount="1e999999"),
            "expected_amount",
            "invalid_format",
        )

    def test_default_currency(self):
        self.assertEqual(self.parse().expected_currency, "MRU")

    def test_currency_normalization(self):
        self.assertEqual(self.parse(expected_currency=" mru ").expected_currency, "MRU")

    def test_currency_invalid_type(self):
        self.assert_error(minimal(expected_currency=1), "expected_currency", "invalid_type")

    def test_empty_currency(self):
        self.assert_error(minimal(expected_currency=" "), "expected_currency", "empty")

    def test_unsupported_currency(self):
        self.assert_error(minimal(expected_currency="EUR"), "expected_currency", "unsupported")

    def test_sender_phone_absent(self):
        self.assertIsNone(self.parse().expected_sender_phone)

    def test_sender_phone_none(self):
        self.assertIsNone(self.parse(expected_sender_phone=None).expected_sender_phone)

    def test_sender_phone_local(self):
        self.assertEqual(self.parse(expected_sender_phone="22123456").expected_sender_phone, "+22222123456")

    def test_sender_phone_country_code(self):
        self.assertEqual(self.parse(expected_sender_phone="+22222123456").expected_sender_phone, "+22222123456")

    def test_sender_phone_00222_prefix(self):
        self.assertEqual(self.parse(expected_sender_phone="0022222123456").expected_sender_phone, "+22222123456")

    def test_sender_phone_invalid_type(self):
        self.assert_error(minimal(expected_sender_phone=22123456), "expected_sender_phone", "invalid_type")

    def test_sender_phone_empty(self):
        self.assert_error(minimal(expected_sender_phone=" "), "expected_sender_phone", "empty")

    def test_sender_phone_invalid_format(self):
        self.assert_error(minimal(expected_sender_phone="+222123"), "expected_sender_phone", "invalid_format")

    def test_recipient_absent(self):
        self.assertIsNone(self.parse().expected_recipient)

    def test_recipient_none(self):
        self.assertIsNone(self.parse(expected_recipient=None).expected_recipient)

    def test_recipient_normalization(self):
        self.assertEqual(self.parse(expected_recipient="  ACPEC   GAS  ").expected_recipient, "ACPEC GAS")

    def test_recipient_invalid_type(self):
        self.assert_error(minimal(expected_recipient=1), "expected_recipient", "invalid_type")

    def test_recipient_empty(self):
        self.assert_error(minimal(expected_recipient="  "), "expected_recipient", "empty")

    def test_default_max_age(self):
        self.assertEqual(self.parse().max_age_hours, 6.0)

    def test_valid_minimum_max_age(self):
        self.assertEqual(self.parse(max_age_hours=Decimal("0.5")).max_age_hours, 0.5)

    def test_valid_maximum_max_age(self):
        self.assertEqual(self.parse(max_age_hours=72).max_age_hours, 72.0)

    def test_max_age_numeric_string(self):
        self.assertEqual(self.parse(max_age_hours="12.5").max_age_hours, 12.5)

    def test_max_age_invalid_type(self):
        self.assert_error(minimal(max_age_hours=None), "max_age_hours", "invalid_type")

    def test_max_age_invalid_format(self):
        self.assert_error(minimal(max_age_hours="hours"), "max_age_hours", "invalid_format")

    def test_max_age_below_minimum(self):
        self.assert_error(minimal(max_age_hours=0.49), "max_age_hours", "out_of_range")

    def test_max_age_above_maximum(self):
        self.assert_error(minimal(max_age_hours=72.1), "max_age_hours", "out_of_range")

    def test_max_age_rejects_boolean(self):
        self.assert_error(minimal(max_age_hours=True), "max_age_hours", "invalid_type")

    def test_contract_error_to_dict(self):
        error = PaymentContractError("field", "code", "message")
        self.assertEqual(error.to_dict(), {"field": "field", "code": "code", "message": "message"})

    def test_dataclass_is_frozen(self):
        parsed = self.parse()
        with self.assertRaises(FrozenInstanceError):
            parsed.expected_provider = "sedad"


if __name__ == "__main__":
    unittest.main()
