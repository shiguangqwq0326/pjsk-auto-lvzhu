"""Check frame timing and hand/GIF placement for the generated batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from compose_gifs import (DEFAULT_OVERLAY, center, center_distance,
                          center_region, load_overlay, visible_touches)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "hand_gifs"
DETECTIONS = ROOT / "calibrated_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="检查生成的手部 GIF")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = json.loads((args.output / "placements.json").read_text(encoding="utf-8"))
    issues = []
    singles = pairs = 0
    overlay_frames, _, _ = load_overlay(DEFAULT_OVERLAY)
    for result in results:
        source = Path(result["source"])
        output = Path(result["output"])
        detection = json.loads((DETECTIONS / f"{source.stem}_hands.json").read_text(
            encoding="utf-8"))
        boxes = [hand["box_xyxy"] for hand in detection["hands"]]
        covered = []
        with Image.open(source) as original, Image.open(output) as gif:
            if gif.size != original.size:
                issues.append(f"{source.name}: 输出尺寸不一致")
            expected_frames = 2 if boxes else 1
            if gif.n_frames != expected_frames:
                issues.append(f"{source.name}: 帧数 {gif.n_frames}，预期 {expected_frames}")
            if boxes:
                if [gif.seek(i) or gif.info.get("duration") for i in range(gif.n_frames)] != [100, 100]:
                    issues.append(f"{source.name}: 帧时长不一致")
        for placement in result["placements"]:
            indices = placement["hand_indices"]
            covered.extend(indices)
            overlay_box = placement["overlay_box_xyxy"]
            if placement["kind"] == "single":
                singles += 1
                box = boxes[indices[0] - 1]
                hand_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                gif_center = ((overlay_box[0] + overlay_box[2]) / 2,
                              (overlay_box[1] + overlay_box[3]) / 2)
                if any(abs(a - b) > 1 for a, b in zip(hand_center, gif_center)):
                    issues.append(f"{source.name}: 单手 GIF 未居中 {indices}")
                sprite = overlay_frames[0].resize(
                    (overlay_box[2] - overlay_box[0],
                     overlay_box[3] - overlay_box[1]), Image.Resampling.LANCZOS)
                if not visible_touches(sprite, overlay_box[0], overlay_box[1], box):
                    issues.append(f"{source.name}: 单手 GIF 未覆盖手部 {indices}")
            else:
                pairs += 1
                first, second = (boxes[index - 1] for index in indices)
                if center_distance(first, second) > 200:
                    issues.append(f"{source.name}: 双手超过 200 像素 {indices}")
                ca, cb = center(first), center(second)
                target_center = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)
                gif_center = ((overlay_box[0] + overlay_box[2]) / 2,
                              (overlay_box[1] + overlay_box[3]) / 2)
                if any(abs(a - b) > 1 for a, b in zip(target_center, gif_center)):
                    issues.append(f"{source.name}: 双手 GIF 未按手中心居中 {indices}")
                sprite = overlay_frames[0].resize(
                    (overlay_box[2] - overlay_box[0],
                     overlay_box[3] - overlay_box[1]), Image.Resampling.LANCZOS)
                for box in (first, second):
                    if not visible_touches(sprite, overlay_box[0], overlay_box[1],
                                           center_region(box)):
                        issues.append(f"{source.name}: GIF 未到达手中心附近 {indices}")
        if sorted(covered) != list(range(1, len(boxes) + 1)):
            issues.append(f"{source.name}: 手部分组重复或遗漏")
    print(f"GIF 数量：{len(results)}；单手：{singles}；双手组：{pairs}")
    if issues:
        raise SystemExit("\n".join(issues))
    print("尺寸、帧时长和手部中心位置检查全部通过")


if __name__ == "__main__":
    main()
