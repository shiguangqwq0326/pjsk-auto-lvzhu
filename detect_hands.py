"""Detect drawn hands in anime images and save boxes as an image and JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from score_correction import DEFAULT_MODEL, corrected_score, load_model


def detect(input_path: Path, output_path: Path | None, json_path: Path | None,
           level: str, confidence: float, offline: bool = False,
           calibration_path: Path | None = DEFAULT_MODEL) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到图片：{input_path}")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from imgutils.detect.hand import detect_hands

    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size

    # This model was trained for hands in anime art, including flat drawings.
    detections = detect_hands(image, level=level, version="v1.0",
                              conf_threshold=confidence)
    calibration = load_model(calibration_path) if calibration_path and level == "s" else None
    draw = ImageDraw.Draw(image) if output_path is not None else None
    hands = []
    for box, label, score in detections:
        left, top, right, bottom = (int(value) for value in box)
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if left >= right or top >= bottom:
            continue
        raw_score = float(score)
        score = (corrected_score(raw_score, (left, top, right, bottom),
                                 (width, height), calibration)
                 if calibration else raw_score)
        if score < confidence:
            continue
        index = len(hands) + 1
        if draw is not None:
            draw.rectangle((left, top, right - 1, bottom - 1),
                           outline=(255, 205, 0), width=3)
            draw.text((left, max(0, top - 16)), f"Hand {index}: {score:.2f}",
                      fill=(255, 205, 0), stroke_width=1, stroke_fill=(0, 0, 0))
        hands.append({"box_xyxy": [left, top, right, bottom],
                      "confidence": round(score, 4),
                      "raw_confidence": round(raw_score, 4), "label": label})

    data = {"image": str(input_path), "width": width, "height": height,
            "hand_count": len(hands), "hands": hands}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="识别动漫图片中的平面手部")
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument("--output", type=Path, help="标注图路径")
    parser.add_argument("--json", type=Path, help="坐标 JSON 路径")
    parser.add_argument("--level", choices=("s", "n"), default="s",
                        help="s 较准，n 较快；默认 s")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="检测阈值，范围 0 到 1；默认 0.35")
    parser.add_argument("--offline", action="store_true",
                        help="使用首次运行后缓存的模型，不访问网络")
    parser.add_argument("--raw", action="store_true",
                        help="关闭使用漫画样本训练的分数修正")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence 必须在 0 到 1 之间")
    output = args.output or args.image.with_name(args.image.stem + "_hands.png")
    json_path = args.json or args.image.with_name(args.image.stem + "_hands.json")
    try:
        data = detect(args.image, output, json_path, args.level,
                      args.confidence, args.offline,
                      None if args.raw or args.level == "n" else DEFAULT_MODEL)
    except Exception as exc:
        parser.exit(1, f"识别失败：{exc}\n")
    print(f"识别到 {data['hand_count']} 处动漫手部")
    for index, hand in enumerate(data["hands"], start=1):
        print(f"手 {index}: {hand['box_xyxy']} 置信度 {hand['confidence']:.2f}")
    print(f"标注图：{output}\n坐标文件：{json_path}")


if __name__ == "__main__":
    main()
