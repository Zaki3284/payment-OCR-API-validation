"""CIN extraction for the HTTP API."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from api.utils import blur_metrics, fields_for_api
from kyc_demo import extract_fields, normalize_card_image

_VALID_ENGINES = frozenset({"auto", "gemini", "offline", "tesseract", "easyocr"})


def normalize_engine(engine: str) -> str:
    e = (engine or "auto").lower().strip()
    if e not in _VALID_ENGINES:
        raise ValueError(
            f"engine must be one of: {', '.join(sorted(_VALID_ENGINES))}"
        )
    return e


def extract_cin_from_image(
    img: np.ndarray,
    *,
    engine: str = "auto",
    use_ml: bool = False,
) -> dict[str, Any]:
    engine = normalize_engine(engine)
    t0 = time.time()
    work, card_detected = normalize_card_image(img)
    blur = blur_metrics(work)
    fields = extract_fields(work, engine=engine, use_ml=use_ml, verbose=False)
    fields.pop("nom", None)
    return {
        "data": fields_for_api(fields),
        "card_detected": card_detected,
        "blur_score": blur["score"],
        "engine": engine,
        "processing_time_ms": int((time.time() - t0) * 1000),
    }
