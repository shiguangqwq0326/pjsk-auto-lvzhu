"""Apply the corrected anime-hand detector to every comic image in a folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from detect_hands import detect


def main() -> None:
    parser = argparse.ArgumentParser(description="批量识别动漫手部")
    parser.add_argument("images", type=Path, help="包含原始漫画图片的目录")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "calibrated_outputs")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f"找不到图片目录：{args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in sorted(args.images.glob("comic_*.webp")):
        output_path = args.output / f"{image_path.stem}_hands.png"
        json_path = args.output / f"{image_path.stem}_hands.json"
        data = detect(image_path, output_path, json_path, "s",
                      args.confidence, offline=True)
        rows.append({"image": image_path.name, "hand_count": data["hand_count"],
                     "annotated_image": str(output_path), "json": str(json_path)})
        print(f"{image_path.name}: {data['hand_count']}", flush=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"处理完成：{len(rows)} 张图片")


if __name__ == "__main__":
    main()
