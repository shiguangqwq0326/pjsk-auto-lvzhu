"""Place an animated GIF on detected anime hands and export animated GIFs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps
from score_correction import DEFAULT_MODEL


ROOT = Path(__file__).resolve().parent
DEFAULT_OVERLAY = ROOT / "assets" / "hand_overlay.gif"
DEFAULT_DETECTIONS = ROOT / "calibrated_outputs"
DEFAULT_OUTPUT = ROOT / "hand_gifs"


def center(box: list[int]) -> tuple[float, float]:
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def center_distance(first: list[int], second: list[int]) -> float:
    ax, ay = center(first)
    bx, by = center(second)
    return math.hypot(ax - bx, ay - by)


def group_hands(boxes: list[list[int]], max_distance: float) -> list[tuple[int, ...]]:
    """Pair as many nearby hands as possible, choosing the nearest total pairing."""
    distances = {(i, j): center_distance(boxes[i], boxes[j])
                 for i in range(len(boxes)) for j in range(i + 1, len(boxes))}

    @lru_cache(None)
    def solve(remaining: tuple[int, ...]) -> tuple[int, float, tuple[tuple[int, ...], ...]]:
        if not remaining:
            return 0, 0.0, ()
        first, *others = remaining
        count, distance, groups = solve(tuple(others))
        best = count, distance, ((first,),) + groups
        for second in others:
            gap = distances[(min(first, second), max(first, second))]
            if gap > max_distance:
                continue
            rest = tuple(index for index in others if index != second)
            count, distance, groups = solve(rest)
            candidate = count + 1, distance + gap, ((first, second),) + groups
            if candidate[0] > best[0] or (candidate[0] == best[0]
                                             and candidate[1] < best[1]):
                best = candidate
        return best

    return list(solve(tuple(range(len(boxes))))[2])


def center_region(box: list[int], tolerance: int = 10) -> list[int]:
    """A small area around the detected hand center for visible contact."""
    cx, cy = center(box)
    return [max(box[0], math.floor(cx - tolerance)),
            max(box[1], math.floor(cy - tolerance)),
            min(box[2], math.ceil(cx + tolerance + 1)),
            min(box[3], math.ceil(cy + tolerance + 1))]


def load_overlay(path: Path) -> tuple[list[Image.Image], list[int], int]:
    with Image.open(path) as source:
        frames = []
        durations = []
        loop = source.info.get("loop", 0)
        for index in range(source.n_frames):
            source.seek(index)
            frames.append(source.convert("RGBA").copy())
            durations.append(max(20, int(source.info.get("duration", 100))))
    bounds = [frame.getchannel("A").getbbox() for frame in frames]
    bounds = [bbox for bbox in bounds if bbox]
    if not bounds:
        raise ValueError(f"GIF 没有可见内容：{path}")
    left = min(bbox[0] for bbox in bounds)
    top = min(bbox[1] for bbox in bounds)
    right = max(bbox[2] for bbox in bounds)
    bottom = max(bbox[3] for bbox in bounds)
    return [frame.crop((left, top, right, bottom)) for frame in frames], durations, loop


def visible_touches(sprite: Image.Image, left: int, top: int,
                    box: list[int]) -> bool:
    overlap = (max(0, box[0] - left), max(0, box[1] - top),
               min(sprite.width, box[2] - left),
               min(sprite.height, box[3] - top))
    if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
        return False
    alpha = sprite.getchannel("A").crop(overlap)
    return alpha.point(lambda value: 255 if value >= 128 else 0).getbbox() is not None


def placements(boxes: list[list[int]], sprite_frame: Image.Image,
               pair_distance: float, single_scale: float, pair_scale: float,
               min_pair_size: int) -> list[dict]:
    sprite_w, sprite_h = sprite_frame.size
    result = []
    for group in group_hands(boxes, pair_distance):
        if len(group) == 1:
            box = boxes[group[0]]
            middle_x, middle_y = center(box)
            hand_width, hand_height = box[2] - box[0], box[3] - box[1]
            # The diagonal also fits inside the opaque circular character,
            # including the rectangular hand box's corners.
            scale = max(hand_width / sprite_w, hand_height / sprite_h,
                        math.hypot(hand_width, hand_height) /
                        min(sprite_w, sprite_h)) * single_scale
            kind = "single"
            distance = None
        else:
            first, second = boxes[group[0]], boxes[group[1]]
            center_a, center_b = center(first), center(second)
            middle_x = (center_a[0] + center_b[0]) / 2
            middle_y = (center_a[1] + center_b[1]) / 2
            center_gap = center_distance(first, second)
            scale = max(center_gap, min_pair_size) / min(sprite_w, sprite_h)
            scale *= pair_scale
            kind = "pair"
            distance = round(center_gap, 2)
        width = max(1, math.ceil(sprite_w * scale))
        height = max(1, math.ceil(sprite_h * scale))
        left = round(middle_x - width / 2)
        top = round(middle_y - height / 2)
        if kind == "pair":
            for _ in range(30):
                sprite = sprite_frame.resize((width, height), Image.Resampling.LANCZOS)
                if (visible_touches(sprite, left, top, center_region(first)) and
                        visible_touches(sprite, left, top, center_region(second))):
                    break
                scale *= 1.02
                width = max(1, math.ceil(sprite_w * scale))
                height = max(1, math.ceil(sprite_h * scale))
                left = round(middle_x - width / 2)
                top = round(middle_y - height / 2)
            else:
                raise RuntimeError(f"无法让 GIF 主体接触双手中心附近：{group}")
        result.append({"kind": kind, "hand_indices": [index + 1 for index in group],
                       "center_distance": distance,
                       "overlay_box_xyxy": [left, top, left + width, top + height]})
    return result


def paste_clipped(canvas: Image.Image, sprite: Image.Image,
                  left: int, top: int) -> None:
    right = min(canvas.width, left + sprite.width)
    bottom = min(canvas.height, top + sprite.height)
    dest_left, dest_top = max(0, left), max(0, top)
    if dest_left < right and dest_top < bottom:
        clipped = sprite.crop((dest_left - left, dest_top - top,
                               right - left, bottom - top))
        canvas.alpha_composite(clipped, (dest_left, dest_top))


def get_detections(image_path: Path, detections_dir: Path,
                   output_dir: Path, confidence: float) -> dict:
    cached = detections_dir / f"{image_path.stem}_hands.json"
    current_inputs_time = max(image_path.stat().st_mtime, DEFAULT_MODEL.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime >= current_inputs_time:
        data = json.loads(cached.read_text(encoding="utf-8"))
        if Path(data["image"]).resolve() == image_path.resolve():
            return data
    from detect_hands import detect
    return detect(image_path,
                  detections_dir / f"{image_path.stem}_hands.png",
                  cached,
                  "s", confidence, offline=True)


def compose(image_path: Path, overlay_frames: list[Image.Image],
            durations: list[int], loop: int, detections_dir: Path,
            output_dir: Path, pair_distance: float, single_scale: float,
            pair_scale: float, min_pair_size: int, confidence: float,
            detections: dict | None = None) -> dict:
    data = (detections if detections is not None else
            get_detections(image_path, detections_dir, output_dir, confidence))
    with Image.open(image_path) as source:
        background = ImageOps.exif_transpose(source).convert("RGBA")
    boxes = [hand["box_xyxy"] for hand in data["hands"]]
    positions = placements(boxes, overlay_frames[0], pair_distance,
                           single_scale, pair_scale, min_pair_size)
    resized: dict[tuple[int, int, int], Image.Image] = {}
    output_frames = []
    for frame_index, overlay_frame in enumerate(overlay_frames):
        canvas = background.copy()
        for item in positions:
            left, top, right, bottom = item["overlay_box_xyxy"]
            key = frame_index, right - left, bottom - top
            if key not in resized:
                resized[key] = overlay_frame.resize((key[1], key[2]),
                                                     Image.Resampling.LANCZOS)
            paste_clipped(canvas, resized[key], left, top)
        output_frames.append(canvas)
    output_path = output_dir / f"{image_path.stem}_overlay.gif"
    if not positions:
        output_frames = output_frames[:1]
        durations = [sum(durations)]
    # A shared palette keeps unchanged comic pixels identical across frames.
    palette = output_frames[0].convert("RGB").quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    indexed_frames = [frame.convert("RGB").quantize(
        palette=palette, dither=Image.Dither.NONE) for frame in output_frames]
    indexed_frames[0].save(output_path, save_all=True,
                           append_images=indexed_frames[1:],
                           duration=durations, loop=loop, disposal=2,
                           optimize=False)
    return {"source": str(image_path), "output": str(output_path),
            "hand_count": len(boxes), "single_count": sum(p["kind"] == "single"
                                                     for p in positions),
            "pair_count": sum(p["kind"] == "pair" for p in positions),
            "placements": positions}


def main() -> None:
    parser = argparse.ArgumentParser(description="在识别出的动漫手部叠加动态 GIF")
    parser.add_argument("images", type=Path, help="一张图片，或包含图片的文件夹")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--detections-dir", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-distance", type=float, default=200,
                        help="双手中心距离上限，默认 200 像素")
    parser.add_argument("--single-scale", type=float, default=0.88,
                        help="单手 GIF 相对手部框对角线的放大倍数，默认 0.88")
    parser.add_argument("--pair-scale", type=float, default=0.9,
                        help="双手 GIF 相对两手中心距离的倍率，默认 0.9")
    parser.add_argument("--min-pair-size", type=int, default=24,
                        help="双手相邻或重叠时 GIF 的最小尺寸，默认 24")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="没有缓存检测结果时使用的阈值")
    args = parser.parse_args()
    if (args.pair_distance < 0 or args.single_scale <= 0 or
            args.pair_scale <= 0 or args.min_pair_size < 1 or
            not 0 <= args.confidence <= 1):
        parser.error("距离不能为负；尺寸倍率与最小尺寸必须为正；置信度须在 0 到 1 之间")
    if not args.overlay.is_file():
        parser.error(f"找不到 GIF：{args.overlay}")
    if args.images.is_dir():
        image_paths = sorted(args.images.glob("comic_*.webp"))
    elif args.images.is_file():
        image_paths = [args.images]
    else:
        parser.error(f"找不到图片：{args.images}")
    if not image_paths:
        parser.error("没有找到可处理的图片")
    args.output.mkdir(parents=True, exist_ok=True)
    overlay_frames, durations, loop = load_overlay(args.overlay)
    results = []
    for image_path in image_paths:
        result = compose(image_path, overlay_frames, durations, loop,
                         args.detections_dir, args.output, args.pair_distance,
                         args.single_scale, args.pair_scale,
                         args.min_pair_size, args.confidence)
        results.append(result)
        print(f"{image_path.name}: 单手 {result['single_count']}，"
              f"双手 {result['pair_count']}，{result['output']}", flush=True)
    (args.output / "placements.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("source", "output", "hand_count",
                                                  "single_count", "pair_count"))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    print(f"完成：{len(results)} 张图片，保存至 {args.output}")


if __name__ == "__main__":
    main()
