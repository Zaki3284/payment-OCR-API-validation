import unittest
from datetime import datetime, timezone

from payment_ocr import validate_payment


class PaymentValidationTests(unittest.TestCase):
    def test_valid_recent_payment_waits_for_admin(self):
        raw = {"provider": "Bankily", "reference": "PAY-ABC123", "amount": "1 250 MRU",
               "paid_at": "2026-07-14T20:30:00+00:00", "phone": "+222 22 12 34 56",
               "confidence": {k: .95 for k in ("provider", "reference", "amount", "paid_at", "phone")}}
        out = validate_payment(raw, expected_code="ABC123", max_age_hours=6,
                               now=datetime(2026, 7, 14, 22, 0, tzinfo=timezone.utc))
        self.assertTrue(out["screening_passed"])
        self.assertEqual(out["review_status"], "pending_admin_review")
        self.assertEqual(out["data"]["phone"], "+22222123456")

    def test_old_payment_is_rejected(self):
        raw = {"provider": "Sadad", "reference": "ABC12345", "amount": "100",
               "paid_at": "14/07/2026 10:00", "phone": "36123456",
               "confidence": {k: .9 for k in ("provider", "reference", "amount", "paid_at", "phone")}}
        out = validate_payment(raw, now=datetime(2026, 7, 14, 22, tzinfo=timezone.utc))
        self.assertIn("payment_too_old", [x["code"] for x in out["issues"]])


if __name__ == "__main__":
    unittest.main()
