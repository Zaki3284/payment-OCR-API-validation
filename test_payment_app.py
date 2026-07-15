import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from payment_app import create_app


def image_base64(size=600, noisy=False):
    if noisy:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    else:
        image = np.zeros((size, size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def gemini_result():
    fields = ("provider", "reference", "amount", "paid_at", "phone")
    return {
        "provider": "Bankily",
        "reference": "PAY-ABC123",
        "amount": "1250",
        "paid_at": "2026-07-14T20:30:00+00:00",
        "phone": "22123456",
        "confidence": {field: 0.95 for field in fields},
    }


class PaymentAppTests(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health(self):
        with patch.dict("os.environ", {}, clear=True):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "gemini_configured": False})

    def test_payment_ocr_without_image(self):
        response = self.client.post("/payments/ocr", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing image_base64")

    def test_payment_ocr_with_invalid_base64(self):
        response = self.client.post("/payments/ocr", json={"image_base64": "%%%"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid base64 image")

    def test_payment_ocr_with_invalid_max_age_hours(self):
        response = self.client.post("/payments/ocr", json={
            "image_base64": image_base64(), "max_age_hours": 0.1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("between 0.5 and 72", response.get_json()["error"])

    def test_payment_ocr_with_low_quality_image(self):
        response = self.client.post("/payments/ocr", json={"image_base64": image_base64()})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.get_json()["quality"]["acceptable"])

    @patch("payment_app.extract_with_gemini")
    @patch("payment_app.image_quality")
    def test_valid_payment_response(self, mock_quality, mock_extract):
        mock_quality.return_value = {
            "blur_score": 500.0, "width": 600, "height": 600, "acceptable": True}
        mock_extract.return_value = gemini_result()
        response = self.client.post("/payments/ocr", json={
            "image_base64": image_base64(noisy=True),
            "expected_payment_code": "ABC123",
            "max_age_hours": 72,
        })
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["ocr"], gemini_result())
        mock_extract.assert_called_once()

    @patch("payment_app.extract_with_gemini", side_effect=RuntimeError("Gemini unavailable"))
    @patch("payment_app.image_quality")
    def test_payment_ocr_when_gemini_fails(self, mock_quality, mock_extract):
        mock_quality.return_value = {
            "blur_score": 500.0, "width": 600, "height": 600, "acceptable": True}
        response = self.client.post("/payments/ocr", json={"image_base64": image_base64(noisy=True)})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"success": False, "error": "Gemini unavailable"})
        mock_extract.assert_called_once()


from datetime import datetime, timezone


def v1_body(**overrides):
    body = {"image_base64": image_base64(noisy=True), "expected_provider": "bankily",
            "expected_amount": 1250, "expected_currency": "MRU", "max_age_hours": 6}
    body.update(overrides)
    return body


def v1_ocr_result(**overrides):
    fields = ("provider", "reference", "amount", "paid_at", "phone", "recipient", "status")
    result = {"provider": "Bankily", "reference": "PAY-ABC123", "amount": "1250",
              "currency": "MRU", "paid_at": datetime.now(timezone.utc).isoformat(),
              "phone": "22123456", "recipient": "ACPEC", "status_text": "successful",
              "confidence": {field: 0.95 for field in fields}}
    result.update(overrides)
    return result


class PaymentProofV1Tests(unittest.TestCase):
    def setUp(self):
        app = create_app(); app.config.update(TESTING=True); self.client = app.test_client()
        self.good_quality = {"blur_score": 500.0, "width": 600, "height": 600, "acceptable": True}

    def post_completed(self, body=None, ocr=None):
        with patch("payment_app.image_quality", return_value=self.good_quality), \
                patch("payment_app.extract_with_gemini", return_value=ocr or v1_ocr_result()):
            return self.client.post("/api/v1/payment-proofs/validate", json=body or v1_body())

    def assert_contract_error(self, body, field, code):
        with patch("payment_app.extract_with_gemini") as gemini:
            response = self.client.post("/api/v1/payment-proofs/validate", json=body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["field"], field)
        self.assertEqual(response.get_json()["error"]["code"], code)
        gemini.assert_not_called()

    def assert_decision(self, decision, body=None, ocr=None):
        response = self.post_completed(body, ocr); payload = response.get_json()
        self.assertEqual(response.status_code, 200); self.assertTrue(payload["success"])
        self.assertEqual(payload["api_version"], "v1"); self.assertEqual(payload["decision"], decision)
        self.assertEqual(payload["decision"], payload["validation"]["decision"])
        return payload

    def test_v1_validation_rejects_non_json_body(self):
        with patch("payment_app.extract_with_gemini") as gemini:
            response = self.client.post("/api/v1/payment-proofs/validate", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400); self.assertEqual(response.get_json()["error"]["field"], "body"); gemini.assert_not_called()
    def test_v1_validation_rejects_empty_body(self): self.assert_contract_error({}, "image_base64", "missing")
    def test_v1_validation_requires_image(self):
        body = v1_body(); del body["image_base64"]; self.assert_contract_error(body, "image_base64", "missing")
    def test_v1_validation_requires_provider(self):
        body = v1_body(); del body["expected_provider"]; self.assert_contract_error(body, "expected_provider", "missing")
    def test_v1_validation_requires_amount(self):
        body = v1_body(); del body["expected_amount"]; self.assert_contract_error(body, "expected_amount", "missing")
    def test_v1_validation_rejects_unsupported_provider(self): self.assert_contract_error(v1_body(expected_provider="other"), "expected_provider", "unsupported")
    def test_v1_validation_rejects_invalid_amount(self): self.assert_contract_error(v1_body(expected_amount="money"), "expected_amount", "invalid_format")
    def test_v1_validation_rejects_invalid_max_age(self): self.assert_contract_error(v1_body(max_age_hours=100), "max_age_hours", "out_of_range")

    def test_v1_validation_rejects_invalid_base64(self):
        response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body(image_base64="%%%"))
        self.assertEqual(response.status_code, 400); self.assertEqual(response.get_json()["error"]["code"], "invalid_image")
    def test_v1_validation_rejects_corrupt_image(self):
        corrupt = base64.b64encode(b"not an image").decode("ascii")
        response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body(image_base64=corrupt))
        self.assertEqual(response.status_code, 400); self.assertEqual(response.get_json()["error"]["code"], "invalid_image")
    def test_v1_validation_rejects_low_quality_image(self):
        with patch("payment_app.extract_with_gemini") as gemini:
            response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body(image_base64=image_base64()))
        self.assertEqual(response.status_code, 422); self.assertEqual(response.get_json()["error"]["code"], "quality_too_low"); gemini.assert_not_called()
    def test_v1_low_quality_does_not_call_gemini(self):
        with patch("payment_app.image_quality", return_value={**self.good_quality, "acceptable": False}), patch("payment_app.extract_with_gemini") as gemini:
            response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body())
        self.assertEqual(response.status_code, 422); gemini.assert_not_called()

    def test_v1_validation_returns_503_when_gemini_fails(self):
        with patch("payment_app.image_quality", return_value=self.good_quality), patch("payment_app.extract_with_gemini", side_effect=RuntimeError("Gemini unavailable")):
            response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], {"field": "ocr", "code": "service_unavailable", "message": "Gemini unavailable"})
    def test_v1_validation_rejects_non_object_gemini_result(self):
        for value in (None, [], "text", 123, True):
            with self.subTest(value=value), patch("payment_app.image_quality", return_value=self.good_quality), patch("payment_app.extract_with_gemini", return_value=value):
                response = self.client.post("/api/v1/payment-proofs/validate", json=v1_body())
                self.assertEqual(response.status_code, 502); self.assertEqual(response.get_json()["error"]["code"], "invalid_response")

    def test_v1_matching_successful_payment_is_accepted(self): self.assert_decision("accepted")
    def test_v1_provider_mismatch_is_rejected(self): self.assert_decision("rejected", v1_body(expected_provider="masrivi"))
    def test_v1_amount_mismatch_is_rejected(self): self.assert_decision("rejected", v1_body(expected_amount=1200))
    def test_v1_verified_failed_payment_is_rejected(self): self.assert_decision("rejected", ocr=v1_ocr_result(status_text="failed"))
    def test_v1_verified_cancelled_payment_is_rejected(self): self.assert_decision("rejected", ocr=v1_ocr_result(status_text="cancelled"))
    def test_v1_verified_rejected_payment_is_rejected(self): self.assert_decision("rejected", ocr=v1_ocr_result(status_text="rejected"))
    def test_v1_missing_status_requires_manual_review(self):
        ocr = v1_ocr_result(); del ocr["status_text"]; self.assert_decision("manual_review", ocr=ocr)
    def test_v1_unknown_status_requires_manual_review(self): self.assert_decision("manual_review", ocr=v1_ocr_result(status_text="pending"))
    def test_v1_low_confidence_status_requires_manual_review(self):
        ocr = v1_ocr_result(); ocr["confidence"]["status"] = 0.2; self.assert_decision("manual_review", ocr=ocr)
    def test_v1_low_confidence_recipient_requires_manual_review(self):
        ocr = v1_ocr_result(); ocr["confidence"]["recipient"] = 0.2; self.assert_decision("manual_review", v1_body(expected_recipient="ACPEC"), ocr)
    def test_v1_verified_recipient_can_match(self): self.assert_decision("accepted", v1_body(expected_recipient="ACPEC"))
    def test_v1_recipient_mismatch_is_rejected(self): self.assert_decision("rejected", v1_body(expected_recipient="OTHER"))
    def test_v1_sender_phone_match(self): self.assert_decision("accepted", v1_body(expected_sender_phone="+22222123456"))
    def test_v1_sender_phone_mismatch_is_rejected(self): self.assert_decision("rejected", v1_body(expected_sender_phone="+22233123456"))

    def test_v1_response_does_not_include_image_base64(self):
        body = v1_body(); payload = self.assert_decision("accepted", body)
        def contains(value):
            if isinstance(value, dict): return any(contains(v) for v in value.values())
            if isinstance(value, list): return any(contains(v) for v in value)
            return value == body["image_base64"]
        self.assertFalse(contains(payload))
    def test_v1_response_contains_quality(self): self.assertIn("quality", self.assert_decision("accepted"))
    def test_v1_response_contains_ocr(self): self.assertIn("ocr", self.assert_decision("accepted"))
    def test_v1_response_contains_validation(self): self.assertIn("validation", self.assert_decision("accepted"))
    def test_v1_response_contains_fingerprint(self): self.assertIsNotNone(self.assert_decision("accepted")["validation"]["fingerprint"])
    def test_v1_processing_time_is_non_negative_integer(self):
        value = self.assert_decision("accepted")["processing_time_ms"]; self.assertIsInstance(value, int); self.assertGreaterEqual(value, 0)

    def test_legacy_payment_ocr_route_still_exists(self): self.assertNotEqual(self.client.post("/payments/ocr", json={}).status_code, 404)
    def test_legacy_payment_ocr_response_shape_is_unchanged(self):
        with patch("payment_app.image_quality", return_value=self.good_quality), patch("payment_app.extract_with_gemini", return_value=gemini_result()):
            payload = self.client.post("/payments/ocr", json={"image_base64": image_base64(noisy=True), "max_age_hours": 72}).get_json()
        self.assertEqual(set(payload), {"success", "quality", "ocr", "validation", "processing_time_ms"})
    def test_v1_route_does_not_change_health_response(self):
        with patch.dict("os.environ", {}, clear=True): payload = self.client.get("/health").get_json()
        self.assertEqual(payload, {"status": "ok", "gemini_configured": False})
    def test_v1_route_only_accepts_post(self): self.assertEqual(self.client.get("/api/v1/payment-proofs/validate").status_code, 405)

    def test_payment_test_page_is_available(self):
        response = self.client.get("/static/payment_test.html")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/api/v1/payment-proofs/validate", response.data)
        response.close()

if __name__ == "__main__":
    unittest.main()
