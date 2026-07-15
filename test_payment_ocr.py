import unittest
from datetime import datetime, timezone

from payment_ocr import validate_payment


NOW = datetime(2026, 7, 14, 22, 0, tzinfo=timezone.utc)
FIELDS = ("provider", "reference", "amount", "paid_at", "phone")


def valid_raw(**overrides):
    raw = {
        "provider": "Bankily",
        "reference": "PAY-ABC123",
        "amount": "1250",
        "paid_at": "2026-07-14T20:30:00+00:00",
        "phone": "22123456",
        "confidence": {field: 0.95 for field in FIELDS},
    }
    raw.update(overrides)
    return raw


def issue_codes(result, field=None):
    return [issue["code"] for issue in result["issues"]
            if field is None or issue["field"] == field]


class PaymentValidationTests(unittest.TestCase):
    def validate(self, raw=None, **kwargs):
        return validate_payment(raw or valid_raw(), now=NOW, **kwargs)

    def test_missing_provider(self):
        out = self.validate(valid_raw(provider=None))
        self.assertIn("missing_or_invalid", issue_codes(out, "provider"))

    def test_unknown_provider(self):
        out = self.validate(valid_raw(provider="Unknown Pay"))
        self.assertIsNone(out["data"]["provider"])
        self.assertIn("missing_or_invalid", issue_codes(out, "provider"))

    def test_provider_alias_sadad(self):
        out = self.validate(valid_raw(provider="Sadad"))
        self.assertEqual(out["data"]["provider"], "Sedad")

    def test_missing_reference(self):
        out = self.validate(valid_raw(reference=None))
        self.assertIn("missing_or_invalid", issue_codes(out, "reference"))

    def test_reference_too_short(self):
        out = self.validate(valid_raw(reference="A123"))
        self.assertIn("too_short", issue_codes(out, "reference"))

    def test_invalid_amount(self):
        out = self.validate(valid_raw(amount="not money"))
        self.assertIsNone(out["data"]["amount"])
        self.assertIn("missing_or_invalid", issue_codes(out, "amount"))

    def test_negative_amount(self):
        out = self.validate(valid_raw(amount="-10"))
        self.assertIsNone(out["data"]["amount"])

    def test_amount_with_mru_currency(self):
        out = self.validate(valid_raw(amount="1 250 MRU"))
        self.assertEqual(out["data"]["amount"], "1250.00")

    def test_invalid_phone(self):
        out = self.validate(valid_raw(phone="123"))
        self.assertIsNone(out["data"]["phone"])
        self.assertIn("missing_or_invalid", issue_codes(out, "phone"))

    def test_phone_with_country_code(self):
        out = self.validate(valid_raw(phone="+222 22 12 34 56"))
        self.assertEqual(out["data"]["phone"], "+22222123456")

    def test_missing_payment_date(self):
        out = self.validate(valid_raw(paid_at=None))
        self.assertIn("missing_or_invalid", issue_codes(out, "paid_at"))

    def test_future_payment(self):
        out = self.validate(valid_raw(paid_at="2026-07-14T23:00:00+00:00"))
        self.assertIn("future_date", issue_codes(out, "paid_at"))

    def test_old_payment(self):
        out = self.validate(valid_raw(paid_at="2026-07-14T10:00:00+00:00"))
        self.assertIn("payment_too_old", issue_codes(out, "paid_at"))

    def test_low_ocr_confidence(self):
        confidence = {field: 0.95 for field in FIELDS}
        confidence["amount"] = 0.50
        out = self.validate(valid_raw(confidence=confidence))
        self.assertIn("low_ocr_confidence", issue_codes(out, "amount"))

    def test_expected_code_not_found(self):
        out = self.validate(expected_code="OTHER")
        self.assertIn("expected_code_not_found", issue_codes(out, "reference"))

    def test_valid_payment_generates_fingerprint(self):
        out = self.validate(expected_code="ABC123")
        self.assertTrue(out["screening_passed"])
        self.assertEqual(len(out["fingerprint"]), 64)
        int(out["fingerprint"], 16)

    def test_same_payment_generates_same_fingerprint(self):
        first = self.validate()["fingerprint"]
        second = self.validate()["fingerprint"]
        self.assertEqual(first, second)

    def test_valid_recent_payment_waits_for_admin(self):
        out = self.validate(expected_code="ABC123")
        self.assertEqual(out["review_status"], "pending_admin_review")

    def test_old_payment_is_rejected(self):
        raw = valid_raw(provider="Sadad", reference="ABC12345", amount="100",
                        paid_at="14/07/2026 10:00", phone="36123456")
        out = self.validate(raw)
        self.assertIn("payment_too_old", issue_codes(out))


if __name__ == "__main__":
    unittest.main()
