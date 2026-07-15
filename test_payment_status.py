import copy
import json
import unittest
from decimal import Decimal

from payment_contract import PaymentValidationRequest
from payment_status import (
    MIN_STATUS_CONFIDENCE,
    PaymentStatusError,
    enrich_screening_with_payment_status,
    normalize_payment_status,
)
from payment_validation_service import evaluate_payment_validation


def screening():
    return {
        "data": {"provider": "Bankily", "reference": "PAY-ABC123",
                 "amount": "1250.00", "currency": "MRU",
                 "paid_at": "2026-07-14T20:30:00+00:00", "phone": "+22222123456"},
        "issues": [], "screening_passed": True, "fingerprint": "f" * 64,
        "review_status": "pending_admin_review",
    }


def request():
    return PaymentValidationRequest("image", "bankily", Decimal("1250.00"),
                                    "MRU", None, None, 6.0)


def enrich(text="successful", confidence=0.95, **raw):
    payload = {"status_text": text, "confidence": {"status": confidence}}
    payload.update(raw)
    return enrich_screening_with_payment_status(payload, screening())


class PaymentStatusTests(unittest.TestCase):
    def test_successful_english(self):
        self.assertEqual(normalize_payment_status("payment successful"), "successful")

    def test_successful_french_with_accents(self):
        self.assertEqual(normalize_payment_status("Paiement réussi"), "successful")

    def test_successful_french_without_accents(self):
        self.assertEqual(normalize_payment_status("paiement reussi"), "successful")

    def test_successful_arabic(self):
        self.assertEqual(normalize_payment_status("تمت العملية بنجاح"), "successful")

    def test_successful_with_extra_whitespace(self):
        self.assertEqual(normalize_payment_status("  payment   successful  "), "successful")

    def test_successful_with_punctuation(self):
        self.assertEqual(normalize_payment_status("Payment successful!"), "successful")

    def test_successful_is_case_insensitive(self):
        self.assertEqual(normalize_payment_status("SUCCESSFUL"), "successful")

    def test_failed_english(self):
        self.assertEqual(normalize_payment_status("transaction failed"), "failed")

    def test_failed_french(self):
        self.assertEqual(normalize_payment_status("Paiement échoué"), "failed")

    def test_failed_arabic(self):
        self.assertEqual(normalize_payment_status("فشل الدفع"), "failed")

    def test_cancelled_british_spelling(self):
        self.assertEqual(normalize_payment_status("cancelled"), "cancelled")

    def test_canceled_american_spelling(self):
        self.assertEqual(normalize_payment_status("canceled"), "cancelled")

    def test_cancelled_french(self):
        self.assertEqual(normalize_payment_status("Paiement annulé"), "cancelled")

    def test_cancelled_arabic(self):
        self.assertEqual(normalize_payment_status("تم إلغاء العملية"), "cancelled")

    def test_rejected_english(self):
        self.assertEqual(normalize_payment_status("payment rejected"), "rejected")

    def test_declined_english(self):
        self.assertEqual(normalize_payment_status("declined"), "rejected")

    def test_rejected_french(self):
        self.assertEqual(normalize_payment_status("Paiement refusé"), "rejected")

    def test_rejected_arabic(self):
        self.assertEqual(normalize_payment_status("مرفوض"), "rejected")

    def test_none_status_is_unknown(self): self.assertEqual(normalize_payment_status(None), "unknown")
    def test_non_string_status_is_unknown(self): self.assertEqual(normalize_payment_status(1), "unknown")
    def test_empty_status_is_unknown(self): self.assertEqual(normalize_payment_status(" "), "unknown")
    def test_pending_status_is_unknown(self): self.assertEqual(normalize_payment_status("pending"), "unknown")
    def test_processing_status_is_unknown(self): self.assertEqual(normalize_payment_status("processing"), "unknown")
    def test_not_successful_is_unknown(self): self.assertEqual(normalize_payment_status("not successful"), "unknown")
    def test_unrecognized_status_is_unknown(self): self.assertEqual(normalize_payment_status("verification required"), "unknown")

    def test_status_field_has_priority(self):
        result = enrich_screening_with_payment_status(
            {"status": "failed", "status_text": "successful", "confidence": {"status": 1}}, screening())
        self.assertEqual(result["data"]["status"], "failed")

    def test_empty_status_field_falls_back_to_status_text(self):
        result = enrich_screening_with_payment_status(
            {"status": " ", "status_text": "successful", "confidence": {"status": 1}}, screening())
        self.assertEqual(result["data"]["status"], "successful")

    def test_missing_status_field_uses_status_text(self):
        self.assertEqual(enrich()["data"]["status"], "successful")

    def test_confidence_integer(self): self.assertEqual(enrich(confidence=1)["data"]["status_confidence"], 1.0)
    def test_confidence_float(self): self.assertEqual(enrich(confidence=0.8)["data"]["status_confidence"], 0.8)
    def test_confidence_decimal(self): self.assertEqual(enrich(confidence=Decimal("0.8"))["data"]["status_confidence"], 0.8)
    def test_confidence_numeric_string(self): self.assertEqual(enrich(confidence="0.8")["data"]["status_confidence"], 0.8)
    def test_confidence_minimum_is_accepted(self): self.assertEqual(enrich(confidence=MIN_STATUS_CONFIDENCE)["data"]["status"], "successful")
    def test_confidence_below_minimum(self): self.assertEqual(enrich(confidence=0.64)["data"]["status"], "unknown")

    def test_confidence_missing(self):
        result = enrich_screening_with_payment_status({"status_text": "successful"}, screening())
        self.assertIsNone(result["data"]["status_confidence"])

    def test_confidence_boolean_is_invalid(self): self.assertIsNone(enrich(confidence=True)["data"]["status_confidence"])
    def test_confidence_nan_is_invalid(self): self.assertIsNone(enrich(confidence=float("nan"))["data"]["status_confidence"])
    def test_confidence_infinity_is_invalid(self): self.assertIsNone(enrich(confidence=float("inf"))["data"]["status_confidence"])
    def test_confidence_negative_is_invalid(self): self.assertIsNone(enrich(confidence=-0.1)["data"]["status_confidence"])
    def test_confidence_above_one_is_invalid(self): self.assertIsNone(enrich(confidence=1.1)["data"]["status_confidence"])

    def test_successful_high_confidence_is_preserved(self): self.assertEqual(enrich()["data"]["status"], "successful")
    def test_failed_high_confidence_is_preserved(self): self.assertEqual(enrich("failed")["data"]["status"], "failed")
    def test_cancelled_high_confidence_is_preserved(self): self.assertEqual(enrich("cancelled")["data"]["status"], "cancelled")
    def test_rejected_high_confidence_is_preserved(self): self.assertEqual(enrich("rejected")["data"]["status"], "rejected")
    def test_recognized_low_confidence_becomes_unknown(self): self.assertEqual(enrich("failed", 0.2)["data"]["status"], "unknown")
    def test_recognized_missing_confidence_becomes_unknown(self):
        result = enrich_screening_with_payment_status({"status_text": "failed"}, screening())
        self.assertEqual(result["data"]["status"], "unknown")
    def test_unknown_text_remains_unknown(self): self.assertEqual(enrich("pending")["data"]["status"], "unknown")

    def test_low_confidence_issue_is_added(self):
        self.assertIn({"field": "status", "code": "low_ocr_confidence"}, enrich("failed", 0.2)["issues"])

    def test_low_confidence_issue_is_not_duplicated(self):
        source = screening(); source["issues"].append({"field": "status", "code": "low_ocr_confidence"})
        result = enrich_screening_with_payment_status(
            {"status_text": "failed", "confidence": {"status": 0.2}}, source)
        self.assertEqual(result["issues"].count({"field": "status", "code": "low_ocr_confidence"}), 1)

    def test_unknown_text_does_not_add_hard_rejection(self):
        self.assertFalse(any(i.get("code") == "missing_or_invalid" for i in enrich("pending")["issues"]))
    def test_fingerprint_is_preserved(self): self.assertEqual(enrich()["fingerprint"], "f" * 64)
    def test_screening_passed_is_preserved(self): self.assertTrue(enrich()["screening_passed"])
    def test_review_status_is_preserved(self): self.assertEqual(enrich()["review_status"], "pending_admin_review")

    def _decision(self, text, confidence=0.95):
        return evaluate_payment_validation(request(), enrich(text, confidence))["decision"]

    def test_verified_successful_status_can_be_accepted(self): self.assertEqual(self._decision("successful"), "accepted")
    def test_verified_failed_status_is_rejected(self): self.assertEqual(self._decision("failed"), "rejected")
    def test_verified_cancelled_status_is_rejected(self): self.assertEqual(self._decision("cancelled"), "rejected")
    def test_verified_rejected_status_is_rejected(self): self.assertEqual(self._decision("rejected"), "rejected")
    def test_low_confidence_success_requires_manual_review(self): self.assertEqual(self._decision("successful", 0.2), "manual_review")
    def test_unknown_status_requires_manual_review(self): self.assertEqual(self._decision("pending"), "manual_review")

    def test_raw_ocr_is_not_mutated(self):
        raw = {"status_text": "successful", "confidence": {"status": 0.9}}; before = copy.deepcopy(raw)
        enrich_screening_with_payment_status(raw, screening()); self.assertEqual(raw, before)

    def test_screening_is_not_mutated(self):
        source = screening(); before = copy.deepcopy(source)
        enrich_screening_with_payment_status({}, source); self.assertEqual(source, before)

    def test_screening_data_is_not_mutated(self):
        source = screening(); before = copy.deepcopy(source["data"])
        enrich_screening_with_payment_status({}, source); self.assertEqual(source["data"], before)

    def test_screening_issues_are_not_mutated(self):
        source = screening(); before = copy.deepcopy(source["issues"])
        enrich_screening_with_payment_status({"status_text": "failed"}, source)
        self.assertEqual(source["issues"], before)

    def test_result_is_a_deep_copy(self):
        source = screening()
        result = enrich_screening_with_payment_status(
            {"status_text": "successful", "confidence": {"status": 0.95}}, source)
        self.assertIsNot(result, source)
        self.assertIsNot(result["data"], source["data"])
        self.assertIsNot(result["issues"], source["issues"])

    def test_invalid_raw_ocr(self):
        with self.assertRaisesRegex(PaymentStatusError, "raw_ocr must be an object"):
            enrich_screening_with_payment_status([], screening())
    def test_invalid_screening(self):
        with self.assertRaisesRegex(PaymentStatusError, "screening must be an object"):
            enrich_screening_with_payment_status({}, [])
    def test_missing_screening_data(self):
        source = screening(); del source["data"]
        with self.assertRaisesRegex(PaymentStatusError, "screening data must be an object"):
            enrich_screening_with_payment_status({}, source)
    def test_invalid_screening_data(self):
        source = screening(); source["data"] = []
        with self.assertRaisesRegex(PaymentStatusError, "screening data must be an object"):
            enrich_screening_with_payment_status({}, source)
    def test_missing_screening_issues(self):
        source = screening(); del source["issues"]
        with self.assertRaisesRegex(PaymentStatusError, "screening issues must be a list"):
            enrich_screening_with_payment_status({}, source)
    def test_invalid_screening_issues(self):
        source = screening(); source["issues"] = {}
        with self.assertRaisesRegex(PaymentStatusError, "screening issues must be a list"):
            enrich_screening_with_payment_status({}, source)
    def test_invalid_screening_issue_item(self):
        source = screening(); source["issues"] = ["bad"]
        with self.assertRaisesRegex(PaymentStatusError, "invalid screening issue"):
            enrich_screening_with_payment_status({}, source)

    def test_enriched_result_is_json_serializable(self):
        json.dumps(enrich("successful", Decimal("0.95")))

    def test_status_confidence_is_float_or_none(self):
        self.assertIsInstance(enrich(confidence=Decimal("0.95"))["data"]["status_confidence"], float)
        self.assertIsNone(enrich(confidence="bad")["data"]["status_confidence"])


if __name__ == "__main__":
    unittest.main()
