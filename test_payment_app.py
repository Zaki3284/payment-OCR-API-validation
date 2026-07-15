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


if __name__ == "__main__":
    unittest.main()
