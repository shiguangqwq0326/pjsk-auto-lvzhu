"""Detect anime hands in one image and save the animated overlay beside it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from compose_gifs import DEFAULT_OVERLAY, compose, load_overlay
from detect_hands import detect


SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def model_is_cached() -> bool:
    cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is None:
            hf_home = str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                          / "huggingface")
        cache_root = str(Path(hf_home) / "hub")
    repo_cache = Path(cache_root) / "models--deepghs--anime_hand_detection"
    return any(repo_cache.glob("snapshots/*/hand_detect_v1.0_s/model.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser(description="识别动漫手部，并在原图片文件夹生成动态 GIF")
    parser.add_argument("image", nargs="?", help="原图片路径；省略时在窗口中输入")
    args = parser.parse_args()

    supplied = (args.image if args.image is not None else
                input("请输入图片完整路径（可将图片拖入窗口）："))
    cleaned_path = supplied.strip().strip('"')
    if not cleaned_path:
        parser.error("未输入图片路径")
    image_path = Path(cleaned_path).expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"找不到图片：{image_path}")
    if image_path.suffix.lower() not in SUPPORTED:
        parser.error("支持 PNG、JPG、JPEG、WEBP 和 BMP 图片")

    frames, durations, loop = load_overlay(DEFAULT_OVERLAY)
    cached = model_is_cached()
    if not cached:
        print("未找到本机模型缓存，首次运行需要联网下载动漫手检测模型。", flush=True)
    detections = detect(image_path, None, None, "s", 0.35, offline=cached)
    result = compose(image_path, frames, durations, loop,
                     image_path.parent, image_path.parent,
                     pair_distance=200, single_scale=0.88,
                     pair_scale=0.9, min_pair_size=24, confidence=0.35,
                     detections=detections)

    print(f"识别到 {result['hand_count']} 只手："
          f"单手 {result['single_count']}，双手组 {result['pair_count']}")
    print(f"成品 GIF：{result['output']}")


if __name__ == "__main__":
    main()
