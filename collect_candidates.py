"""Collect low-threshold anime-hand candidates and review sheets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "calibration")
    parser.add_argument("--confidence", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from imgutils.detect.hand import detect_hands

    candidates = []
    for image_path in sorted(args.images.glob("comic_*.webp")):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        results = detect_hands(image, level="s", version="v1.0",
                               conf_threshold=args.confidence)
        for box, _, score in results:
            candidates.append({"id": len(candidates), "image": image_path.name,
                               "width": image.width, "height": image.height,
                               "box_xyxy": [int(value) for value in box],
                               "raw_confidence": round(float(score), 6)})
        print(f"{image_path.name}: {len(results)}", flush=True)

    (args.output / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total: {len(candidates)}")

    columns, rows, cell_w, cell_h = 5, 8, 200, 180
    per_sheet = columns * rows
    for sheet_number, start in enumerate(range(0, len(candidates), per_sheet), start=1):
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        for offset, candidate in enumerate(candidates[start:start + per_sheet]):
            image_path = args.images / candidate["image"]
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            left, top, right, bottom = candidate["box_xyxy"]
            span = max(right - left, bottom - top)
            pad = max(20, round(span * 0.55))
            crop_box = (max(0, left - pad), max(0, top - pad),
                        min(image.width, right + pad), min(image.height, bottom + pad))
            crop = image.crop(crop_box)
            crop.thumbnail((cell_w - 8, cell_h - 28))
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            sheet.paste(crop, (x + (cell_w - crop.width) // 2, y + 22))
            scale_x = crop.width / (crop_box[2] - crop_box[0])
            scale_y = crop.height / (crop_box[3] - crop_box[1])
            paste_x = x + (cell_w - crop.width) // 2
            paste_y = y + 22
            draw.rectangle((paste_x + (left - crop_box[0]) * scale_x,
                            paste_y + (top - crop_box[1]) * scale_y,
                            paste_x + (right - crop_box[0]) * scale_x,
                            paste_y + (bottom - crop_box[1]) * scale_y),
                           outline="#ffff00", width=2)
            draw.text((x + 5, y + 4),
                      f"#{candidate['id']} {candidate['image'][6:10]} "
                      f"{candidate['raw_confidence']:.2f}", fill="#111111")
        sheet.save(args.output / f"review_{sheet_number:02d}.jpg", quality=90)


if __name__ == "__main__":
    main()
