"""Apply the reviewed, size-based hand detection score correction."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MODEL = Path(__file__).resolve().parent / "calibration" / "confidence_model.json"


def load_model(path: Path = DEFAULT_MODEL) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corrected_score(raw_score: float, box: tuple[int, int, int, int],
                    image_size: tuple[int, int], model: dict) -> float:
    width, height = image_size
    left, top, right, bottom = box
    area_fraction = ((right - left) * (bottom - top) /
                     max(1, width * height))
    if area_fraction <= 0:
        return 0.0
    scale = min(1.0, (model["size_limit"] / area_fraction) ** model["power"])
    border_limit = model.get("border_margin_limit_px")
    if border_limit is not None:
        margin = min(left, top, width - right, height - bottom)
        if margin <= border_limit:
            scale *= model["border_score_scale"]
    return float(raw_score) * scale
