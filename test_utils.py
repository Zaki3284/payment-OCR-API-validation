import base64
import unittest

import cv2
import numpy as np

from utils import decode_base64_image, json_image


def png_base64():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :6] = (10, 20, 30)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ImageDecodingTests(unittest.TestCase):
    def test_missing_base64_image(self):
        with self.assertRaisesRegex(ValueError, "missing image_base64"):
            json_image({}, "image_base64")

    def test_empty_base64_image(self):
        with self.assertRaisesRegex(ValueError, "empty base64 image"):
            decode_base64_image("   ")

    def test_invalid_base64_image(self):
        with self.assertRaisesRegex(ValueError, "invalid base64 image"):
            decode_base64_image("%%%not-base64%%")

    def test_corrupt_image(self):
        corrupt = base64.b64encode(b"not an image").decode("ascii")
        with self.assertRaisesRegex(ValueError, "unsupported or corrupt image"):
            decode_base64_image(corrupt)

    def test_valid_base64_image(self):
        image = decode_base64_image(png_base64())
        self.assertEqual(image.shape, (8, 12, 3))

    def test_data_url_base64_image(self):
        image = decode_base64_image("data:image/png;base64," + png_base64())
        self.assertEqual(image.shape, (8, 12, 3))


if __name__ == "__main__":
    unittest.main()
