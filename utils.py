"""API helpers — JSON + base64 images (Flutter)."""

from __future__ import annotations

import base64
import re
import cv2
import numpy as np

_B64_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.I)


def decode_base64_image(b64: str) -> np.ndarray:
    if not b64 or not str(b64).strip():
        raise ValueError("empty base64 image")
    text = _B64_DATA_URL.sub("", str(b64).strip())
    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 image") from exc
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("unsupported or corrupt image")
    return img


def json_image(data: dict, key: str) -> np.ndarray:
    b64 = data.get(key)
    if not b64:
        raise ValueError(f"missing {key}")
    return decode_base64_image(b64)


def blur_metrics(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"score": round(score, 1)}

