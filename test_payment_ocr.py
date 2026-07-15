import unittest
from datetime import datetime, timezone

from payment_ocr import PAYMENT_OCR_PROMPT, validate_payment


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

    def test_ocr_prompt_requests_recipient(self):
        self.assertIn('"recipient":null', "".join(PAYMENT_OCR_PROMPT.split()))

    def test_ocr_prompt_requests_status_text(self):
        self.assertIn('"status_text":null', "".join(PAYMENT_OCR_PROMPT.split()))

    def test_ocr_prompt_requests_recipient_confidence(self):
        self.assertIn('"recipient":0', "".join(PAYMENT_OCR_PROMPT.split()))

    def test_ocr_prompt_requests_status_confidence(self):
        self.assertIn('"status":0', "".join(PAYMENT_OCR_PROMPT.split()))

    def test_ocr_prompt_forbids_status_inference(self):
        prompt = PAYMENT_OCR_PROMPT.casefold()
        self.assertIn("does not prove", prompt)
        self.assertIn("do not translate, normalize, or infer success", prompt)
        self.assertIn("visibly printed", prompt)

    def test_ocr_prompt_forbids_recipient_inference(self):
        prompt = PAYMENT_OCR_PROMPT.casefold()
        self.assertIn("never infer it", prompt)
        self.assertIn("visibly printed beneficiary", prompt)

    def test_recipient_is_not_a_required_legacy_field(self):
        out = self.validate(valid_raw())
        self.assertNotIn("recipient", out["data"])
        self.assertNotIn("recipient", [issue["field"] for issue in out["issues"]])

    def test_status_is_not_a_required_legacy_field(self):
        out = self.validate(valid_raw())
        self.assertNotIn("status", out["data"])
        self.assertNotIn("status", [issue["field"] for issue in out["issues"]])

    def test_extra_recipient_does_not_change_legacy_screening(self):
        baseline = self.validate(valid_raw())
        enriched = self.validate(valid_raw(recipient="ACPEC"))
        self.assertEqual(enriched, baseline)

    def test_extra_status_does_not_change_legacy_screening(self):
        baseline = self.validate(valid_raw())
        enriched = self.validate(valid_raw(status_text="Paiement réussi"))
        self.assertEqual(enriched, baseline)

    def test_extra_status_confidence_does_not_change_legacy_screening(self):
        baseline_raw = valid_raw()
        confidence = {**baseline_raw["confidence"], "status": 0.99}
        enriched = self.validate(valid_raw(status_text="Paiement réussi", confidence=confidence))
        self.assertEqual(enriched, self.validate(baseline_raw))

    def test_extra_fields_do_not_change_fingerprint(self):
        baseline = self.validate(valid_raw())
        raw = valid_raw(recipient="ACPEC", status_text="successful")
        raw["confidence"] = {**raw["confidence"], "recipient": 0.98, "status": 0.99}
        enriched = self.validate(raw)
        self.assertEqual(enriched["fingerprint"], baseline["fingerprint"])

if __name__ == "__main__":
    unittest.main()
