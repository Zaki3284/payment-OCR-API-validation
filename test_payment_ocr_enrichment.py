import copy
import json
import unittest
from decimal import Decimal

from payment_contract import PaymentValidationRequest
from payment_ocr_enrichment import (
    MIN_RECIPIENT_CONFIDENCE,
    PaymentOcrEnrichmentError,
    enrich_screening_with_ocr_fields,
)
from payment_validation_service import evaluate_payment_validation


def base_screening():
    return {
        "data": {"provider": "Bankily", "reference": "PAY-ABC123",
                 "amount": "1250.00", "currency": "MRU",
                 "paid_at": "2026-07-15T16:00:00+00:00", "phone": "+22222123456"},
        "issues": [], "screening_passed": True, "fingerprint": "f" * 64,
        "review_status": "pending_admin_review",
    }


def raw(recipient="ACPEC", recipient_confidence=0.95, status="successful", status_confidence=0.95):
    return {"recipient": recipient, "status_text": status,
            "confidence": {"recipient": recipient_confidence, "status": status_confidence}}


def expected_request(recipient="ACPEC"):
    return PaymentValidationRequest("image", "bankily", Decimal("1250.00"),
                                    "MRU", None, recipient, 6.0)


class PaymentOcrEnrichmentTests(unittest.TestCase):
    def enrich(self, raw_ocr=None, screening=None):
        return enrich_screening_with_ocr_fields(
            raw() if raw_ocr is None else raw_ocr,
            base_screening() if screening is None else screening)

    def test_verified_recipient_is_added(self):
        self.assertEqual(self.enrich()["data"]["recipient"], "ACPEC")

    def test_recipient_whitespace_is_normalized(self):
        self.assertEqual(self.enrich(raw("  ACPEC   GAS  "))["data"]["recipient"], "ACPEC GAS")

    def test_recipient_capitalization_is_preserved(self):
        self.assertEqual(self.enrich(raw("Acpec Gas"))["data"]["recipient"], "Acpec Gas")

    def test_recipient_arabic_text_is_preserved(self):
        self.assertEqual(self.enrich(raw("محمد سيدي"))["data"]["recipient"], "محمد سيدي")

    def test_recipient_confidence_integer(self): self.assertEqual(self.enrich(raw(recipient_confidence=1))["data"]["recipient_confidence"], 1.0)
    def test_recipient_confidence_float(self): self.assertEqual(self.enrich(raw(recipient_confidence=0.8))["data"]["recipient_confidence"], 0.8)
    def test_recipient_confidence_decimal(self): self.assertEqual(self.enrich(raw(recipient_confidence=Decimal("0.8")))["data"]["recipient_confidence"], 0.8)
    def test_recipient_confidence_numeric_string(self): self.assertEqual(self.enrich(raw(recipient_confidence="0.8"))["data"]["recipient_confidence"], 0.8)
    def test_minimum_recipient_confidence_is_accepted(self): self.assertEqual(self.enrich(raw(recipient_confidence=MIN_RECIPIENT_CONFIDENCE))["data"]["recipient"], "ACPEC")

    def test_missing_recipient_is_none(self): self.assertIsNone(self.enrich(raw_ocr={})["data"]["recipient"])
    def test_empty_recipient_is_none(self): self.assertIsNone(self.enrich(raw(" "))["data"]["recipient"])
    def test_non_string_recipient_is_none(self): self.assertIsNone(self.enrich(raw(123))["data"]["recipient"])

    def test_missing_recipient_confidence_marks_visible_recipient_low_confidence(self):
        value = raw(); del value["confidence"]["recipient"]
        self.assertIsNone(self.enrich(value)["data"]["recipient"])
    def test_invalid_recipient_confidence_marks_visible_recipient_low_confidence(self): self.assertIsNone(self.enrich(raw(recipient_confidence="bad"))["data"]["recipient"])
    def test_low_recipient_confidence_marks_visible_recipient_low_confidence(self): self.assertIsNone(self.enrich(raw(recipient_confidence=0.2))["data"]["recipient"])
    def test_boolean_recipient_confidence_is_invalid(self): self.assertIsNone(self.enrich(raw(recipient_confidence=True))["data"]["recipient_confidence"])
    def test_nan_recipient_confidence_is_invalid(self): self.assertIsNone(self.enrich(raw(recipient_confidence=float("nan")))["data"]["recipient_confidence"])
    def test_infinite_recipient_confidence_is_invalid(self): self.assertIsNone(self.enrich(raw(recipient_confidence=float("inf")))["data"]["recipient_confidence"])
    def test_negative_recipient_confidence_is_invalid(self): self.assertIsNone(self.enrich(raw(recipient_confidence=-1))["data"]["recipient_confidence"])
    def test_recipient_confidence_above_one_is_invalid(self): self.assertIsNone(self.enrich(raw(recipient_confidence=1.1))["data"]["recipient_confidence"])

    def test_low_confidence_issue_is_added(self):
        self.assertIn({"field": "recipient", "code": "low_ocr_confidence"}, self.enrich(raw(recipient_confidence=0.2))["issues"])
    def test_low_confidence_issue_is_not_duplicated(self):
        source = base_screening(); source["issues"].append({"field": "recipient", "code": "low_ocr_confidence"})
        result = self.enrich(raw(recipient_confidence=0.2), source)
        self.assertEqual(result["issues"].count({"field": "recipient", "code": "low_ocr_confidence"}), 1)

    def test_status_enrichment_is_preserved(self): self.assertEqual(self.enrich()["data"]["status"], "successful")
    def test_fingerprint_is_preserved(self): self.assertEqual(self.enrich()["fingerprint"], "f" * 64)
    def test_screening_passed_is_preserved(self): self.assertTrue(self.enrich()["screening_passed"])
    def test_review_status_is_preserved(self): self.assertEqual(self.enrich()["review_status"], "pending_admin_review")

    def test_raw_ocr_is_not_mutated(self):
        value = raw(); before = copy.deepcopy(value); self.enrich(value); self.assertEqual(value, before)
    def test_screening_is_not_mutated(self):
        value = base_screening(); before = copy.deepcopy(value); self.enrich(screening=value); self.assertEqual(value, before)
    def test_screening_data_is_not_mutated(self):
        value = base_screening(); before = copy.deepcopy(value["data"]); self.enrich(screening=value); self.assertEqual(value["data"], before)
    def test_screening_issues_are_not_mutated(self):
        value = base_screening(); before = copy.deepcopy(value["issues"]); self.enrich(screening=value); self.assertEqual(value["issues"], before)
    def test_result_is_a_deep_copy(self):
        value = base_screening(); result = self.enrich(screening=value)
        self.assertIsNot(result, value); self.assertIsNot(result["data"], value["data"]); self.assertIsNot(result["issues"], value["issues"])

    def test_invalid_raw_ocr(self):
        with self.assertRaisesRegex(PaymentOcrEnrichmentError, "raw_ocr must be an object"): self.enrich([])
    def test_invalid_screening(self):
        with self.assertRaisesRegex(PaymentOcrEnrichmentError, "screening must be an object"): self.enrich(screening=[])
    def test_invalid_screening_data(self):
        value = base_screening(); value["data"] = []
        with self.assertRaisesRegex(PaymentOcrEnrichmentError, "screening data must be an object"): self.enrich(screening=value)
    def test_invalid_screening_issues(self):
        value = base_screening(); value["issues"] = {}
        with self.assertRaisesRegex(PaymentOcrEnrichmentError, "screening issues must be a list"): self.enrich(screening=value)
    def test_invalid_screening_issue_item(self):
        value = base_screening(); value["issues"] = ["bad"]
        with self.assertRaisesRegex(PaymentOcrEnrichmentError, "invalid screening issue"): self.enrich(screening=value)

    def test_result_is_json_serializable(self): json.dumps(self.enrich())
    def test_recipient_confidence_is_float_or_none(self):
        self.assertIsInstance(self.enrich()["data"]["recipient_confidence"], float)
        self.assertIsNone(self.enrich(raw(recipient_confidence="bad"))["data"]["recipient_confidence"])

    def test_verified_expected_recipient_can_match(self):
        self.assertEqual(evaluate_payment_validation(expected_request(), self.enrich())["decision"], "accepted")
    def test_low_confidence_expected_recipient_requires_manual_review(self):
        result = evaluate_payment_validation(expected_request(), self.enrich(raw(recipient_confidence=0.2)))
        self.assertEqual(result["decision"], "manual_review")
    def test_missing_expected_recipient_requires_manual_review(self):
        result = evaluate_payment_validation(expected_request(), self.enrich(raw_ocr={}))
        self.assertEqual(result["decision"], "manual_review")
    def test_recipient_mismatch_is_rejected(self):
        result = evaluate_payment_validation(expected_request("OTHER"), self.enrich())
        self.assertEqual(result["decision"], "rejected")


if __name__ == "__main__":
    unittest.main()
