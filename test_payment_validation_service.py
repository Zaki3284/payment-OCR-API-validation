import copy
import unittest
from decimal import Decimal

from payment_contract import PaymentValidationRequest
from payment_validation_service import (
    PaymentValidationServiceError,
    evaluate_payment_validation,
)


def valid_request(**overrides):
    values = {
        "image_base64": "image",
        "expected_provider": "bankily",
        "expected_amount": Decimal("1250.00"),
        "expected_currency": "MRU",
        "expected_sender_phone": None,
        "expected_recipient": None,
        "max_age_hours": 6.0,
    }
    values.update(overrides)
    return PaymentValidationRequest(**values)


def valid_screening(**overrides):
    result = {
        "data": {"provider": "Bankily", "reference": "PAY-ABC123",
                 "amount": "1250.00", "currency": "MRU",
                 "paid_at": "2026-07-14T20:30:00+00:00",
                 "phone": "+22222123456", "status": "successful"},
        "issues": [], "screening_passed": True,
        "fingerprint": "f" * 64, "review_status": "pending_admin_review",
    }
    result.update(overrides)
    return result


def comparison_issue(result, field):
    return next((item for item in result["issues"]
                 if item.get("source") == "comparison" and item.get("field") == field), None)


class PaymentValidationServiceTests(unittest.TestCase):
    def evaluate(self, request=None, screening=None):
        return evaluate_payment_validation(request or valid_request(), screening or valid_screening())

    def assert_check(self, result, name, status, expected=None, actual=None):
        check = result["checks"][name]
        self.assertEqual(check["status"], status)
        if expected is not None:
            self.assertEqual(check["expected"], expected)
        if actual is not None:
            self.assertEqual(check["actual"], actual)

    def test_all_required_values_match_with_successful_status(self):
        result = self.evaluate()
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["issues"], [])

    def test_current_screening_without_status_requires_manual_review(self):
        screening = valid_screening(); del screening["data"]["status"]
        result = self.evaluate(screening=screening)
        self.assertEqual(result["decision"], "manual_review")
        self.assertEqual(comparison_issue(result, "payment_status")["code"], "payment_status_unverified")

    def test_provider_match(self):
        self.assert_check(self.evaluate(), "provider", "matched", "bankily", "bankily")

    def test_provider_alias_sadad_matches_sedad(self):
        screening = valid_screening(); screening["data"]["provider"] = "Sadad"
        result = self.evaluate(valid_request(expected_provider="sedad"), screening)
        self.assert_check(result, "provider", "matched", "sedad", "sedad")

    def test_provider_mismatch_is_rejected(self):
        result = self.evaluate(valid_request(expected_provider="masrivi"))
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(comparison_issue(result, "provider")["code"], "provider_mismatch")

    def test_provider_missing_is_unverified(self):
        screening = valid_screening(); screening["data"]["provider"] = None
        result = self.evaluate(screening=screening)
        self.assert_check(result, "provider", "unverified")

    def test_amount_match(self):
        self.assert_check(self.evaluate(), "amount", "matched", "1250.00", "1250.00")

    def test_amount_equivalent_formats_match(self):
        for value in (1250, "1250.0", Decimal("1250.00")):
            with self.subTest(value=value):
                screening = valid_screening(); screening["data"]["amount"] = value
                self.assert_check(self.evaluate(screening=screening), "amount", "matched")

    def test_amount_mismatch_is_rejected(self):
        screening = valid_screening(); screening["data"]["amount"] = "1200"
        result = self.evaluate(screening=screening)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(comparison_issue(result, "amount")["actual"], "1200.00")

    def test_amount_invalid_is_unverified(self):
        screening = valid_screening(); screening["data"]["amount"] = "NaN"
        result = self.evaluate(screening=screening)
        self.assert_check(result, "amount", "unverified")

    def test_amount_comparison_does_not_use_float(self):
        screening = valid_screening(); screening["data"]["amount"] = "0.30"
        request = valid_request(expected_amount=Decimal("0.10") + Decimal("0.20"))
        self.assert_check(self.evaluate(request, screening), "amount", "matched", "0.30", "0.30")

    def test_currency_match(self):
        self.assert_check(self.evaluate(), "currency", "matched")

    def test_currency_normalization(self):
        screening = valid_screening(); screening["data"]["currency"] = " mru "
        self.assert_check(self.evaluate(screening=screening), "currency", "matched", "MRU", "MRU")

    def test_currency_mismatch_is_rejected(self):
        screening = valid_screening(); screening["data"]["currency"] = "EUR"
        self.assertEqual(self.evaluate(screening=screening)["decision"], "rejected")

    def test_currency_missing_is_unverified(self):
        screening = valid_screening(); del screening["data"]["currency"]
        result = self.evaluate(screening=screening)
        self.assertEqual(comparison_issue(result, "currency")["code"], "currency_unverified")

    def test_sender_phone_not_requested(self):
        self.assert_check(self.evaluate(), "sender_phone", "not_requested")

    def test_sender_phone_match(self):
        request = valid_request(expected_sender_phone="+22222123456")
        self.assert_check(self.evaluate(request), "sender_phone", "matched")

    def test_sender_phone_mismatch_is_rejected(self):
        request = valid_request(expected_sender_phone="+22233123456")
        self.assertEqual(self.evaluate(request)["decision"], "rejected")

    def test_sender_phone_missing_is_unverified(self):
        screening = valid_screening(); screening["data"]["phone"] = None
        result = self.evaluate(valid_request(expected_sender_phone="+22222123456"), screening)
        self.assert_check(result, "sender_phone", "unverified")

    def test_recipient_not_requested(self):
        self.assert_check(self.evaluate(), "recipient", "not_requested")

    def test_recipient_case_insensitive_match(self):
        screening = valid_screening(); screening["data"]["recipient"] = "acpec"
        self.assert_check(self.evaluate(valid_request(expected_recipient="ACPEC"), screening),
                          "recipient", "matched", "ACPEC", "acpec")

    def test_recipient_whitespace_normalization(self):
        screening = valid_screening(); screening["data"]["recipient"] = " ACPEC   GAS "
        self.assert_check(self.evaluate(valid_request(expected_recipient="ACPEC GAS"), screening),
                          "recipient", "matched", "ACPEC GAS", "ACPEC GAS")

    def test_recipient_mismatch_is_rejected(self):
        screening = valid_screening(); screening["data"]["recipient"] = "OTHER"
        self.assertEqual(self.evaluate(valid_request(expected_recipient="ACPEC"), screening)["decision"], "rejected")

    def test_recipient_missing_requires_manual_review(self):
        result = self.evaluate(valid_request(expected_recipient="ACPEC"))
        self.assertEqual(result["decision"], "manual_review")
        self.assertEqual(comparison_issue(result, "recipient")["code"], "recipient_unverified")

    def test_successful_status_matches(self):
        self.assert_check(self.evaluate(), "payment_status", "matched", "successful", "successful")

    def _status_result(self, status):
        screening = valid_screening(); screening["data"]["status"] = status
        return self.evaluate(screening=screening)

    def test_failed_status_is_rejected(self):
        self.assertEqual(self._status_result("failed")["decision"], "rejected")

    def test_cancelled_status_is_rejected(self):
        self.assertEqual(self._status_result("cancelled")["decision"], "rejected")

    def test_rejected_status_is_rejected(self):
        self.assertEqual(self._status_result("rejected")["decision"], "rejected")

    def test_unknown_status_requires_manual_review(self):
        result = self._status_result("pending")
        self.assertEqual(result["decision"], "manual_review")
        self.assert_check(result, "payment_status", "unverified")

    def test_missing_status_requires_manual_review(self):
        screening = valid_screening(); del screening["data"]["status"]
        self.assertEqual(self.evaluate(screening=screening)["decision"], "manual_review")

    def test_hard_screening_issue_is_rejected(self):
        screening = valid_screening(issues=[{"field": "amount", "code": "missing_or_invalid"}])
        self.assertEqual(self.evaluate(screening=screening)["decision"], "rejected")

    def test_payment_too_old_is_rejected(self):
        screening = valid_screening(issues=[{"field": "paid_at", "code": "payment_too_old"}])
        self.assertEqual(self.evaluate(screening=screening)["decision"], "rejected")

    def test_low_confidence_requires_manual_review(self):
        screening = valid_screening(issues=[{"field": "amount", "code": "low_ocr_confidence"}])
        self.assertEqual(self.evaluate(screening=screening)["decision"], "manual_review")

    def test_unknown_screening_issue_requires_manual_review(self):
        screening = valid_screening(issues=[{"field": "x", "code": "new_code"}])
        self.assertEqual(self.evaluate(screening=screening)["decision"], "manual_review")

    def test_rejection_has_priority_over_manual_review(self):
        screening = valid_screening(issues=[{"field": "amount", "code": "low_ocr_confidence"},
                                            {"field": "paid_at", "code": "future_date"}])
        self.assertEqual(self.evaluate(screening=screening)["decision"], "rejected")

    def test_missing_fingerprint_requires_manual_review(self):
        screening = valid_screening(); del screening["fingerprint"]
        result = self.evaluate(screening=screening)
        self.assertEqual(result["decision"], "manual_review")

    def test_empty_fingerprint_requires_manual_review(self):
        self.assertEqual(self.evaluate(screening=valid_screening(fingerprint=" "))["decision"], "manual_review")

    def test_valid_fingerprint_is_preserved(self):
        self.assertEqual(self.evaluate()["fingerprint"], "f" * 64)

    def test_inconsistent_screening_state_requires_manual_review(self):
        result = self.evaluate(screening=valid_screening(screening_passed=False))
        self.assertEqual(result["decision"], "manual_review")
        self.assertIn("inconsistent_screening_result", [i["code"] for i in result["issues"]])

    def test_result_contains_payment_copy(self):
        screening = valid_screening(); result = self.evaluate(screening=screening)
        self.assertEqual(result["payment"], screening["data"])
        self.assertIsNot(result["payment"], screening["data"])

    def test_screening_input_is_not_mutated(self):
        screening = valid_screening(); before = copy.deepcopy(screening)
        self.evaluate(screening=screening); self.assertEqual(screening, before)

    def test_screening_data_is_not_mutated(self):
        screening = valid_screening(); before = copy.deepcopy(screening["data"])
        self.evaluate(screening=screening); self.assertEqual(screening["data"], before)

    def test_screening_issues_are_not_mutated(self):
        screening = valid_screening(issues=[{"field": "amount", "code": "low_ocr_confidence"}])
        before = copy.deepcopy(screening["issues"]); self.evaluate(screening=screening)
        self.assertEqual(screening["issues"], before)

    def test_invalid_request_type(self):
        with self.assertRaisesRegex(PaymentValidationServiceError, "request must be"):
            evaluate_payment_validation({}, valid_screening())

    def test_invalid_screening_type(self):
        with self.assertRaisesRegex(PaymentValidationServiceError, "screening must be an object"):
            evaluate_payment_validation(valid_request(), [])

    def test_missing_screening_data(self):
        screening = valid_screening(); del screening["data"]
        with self.assertRaisesRegex(PaymentValidationServiceError, "screening data must be an object"):
            evaluate_payment_validation(valid_request(), screening)

    def test_invalid_screening_data(self):
        with self.assertRaisesRegex(PaymentValidationServiceError, "screening data must be an object"):
            evaluate_payment_validation(valid_request(), valid_screening(data=[]))

    def test_invalid_screening_issues(self):
        with self.assertRaisesRegex(PaymentValidationServiceError, "screening issues must be a list"):
            evaluate_payment_validation(valid_request(), valid_screening(issues={}))

    def test_invalid_screening_issue_item(self):
        for issue in ("bad", {}, {"field": 1, "code": "x"}, {"field": "x", "code": 1}):
            with self.subTest(issue=issue), self.assertRaisesRegex(
                    PaymentValidationServiceError, "invalid screening issue"):
                evaluate_payment_validation(valid_request(), valid_screening(issues=[issue]))

    def test_result_values_are_json_compatible(self):
        def assert_json_value(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertIsInstance(key, str); assert_json_value(child)
            elif isinstance(value, list):
                for child in value: assert_json_value(child)
            else:
                self.assertTrue(value is None or isinstance(value, (str, int, float, bool)))
        assert_json_value(self.evaluate())


if __name__ == "__main__":
    unittest.main()
