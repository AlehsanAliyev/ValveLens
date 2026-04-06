from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "detection" / "combined"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "detection_preview"
SPLITS = ("train", "valid", "test")
CLASS_NAMES = {0: "valve", 1: "gauge"}
COLORS = {0: "#f97316", 1: "#0ea5e9"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save preview images with YOLO boxes drawn.")
    parser.add_argument("--per-split", type=int, default=6, help="Number of images to preview per split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def iter_images(split: str) -> List[Path]:
    images_dir = DATASET_ROOT / split / "images"
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def label_path_for(split: str, image_path: Path) -> Path:
    return DATASET_ROOT / split / "labels" / f"{image_path.stem}.txt"


def load_boxes(label_path: Path, image_size: Tuple[int, int]) -> Iterable[Tuple[int, Tuple[int, int, int, int]]]:
    width, height = image_size
    if not label_path.exists():
        return []

    boxes = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])
        x1 = int((x_center - box_w / 2.0) * width)
        y1 = int((y_center - box_h / 2.0) * height)
        x2 = int((x_center + box_w / 2.0) * width)
        y2 = int((y_center + box_h / 2.0) * height)
        boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def draw_preview(image_path: Path, split: str, output_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, (x1, y1, x2, y2) in load_boxes(label_path_for(split, image_path), image.size):
            color = COLORS.get(class_id, "#ffffff")
            label = CLASS_NAMES.get(class_id, str(class_id))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((max(0, x1 + 2), max(0, y1 + 2)), label, fill=color)
        image.save(output_path)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_out = OUTPUT_ROOT / split
        split_out.mkdir(parents=True, exist_ok=True)
        images = iter_images(split)
        if not images:
            continue
        sample_count = min(args.per_split, len(images))
        sampled = sorted(rng.sample(images, sample_count))
        for image_path in sampled:
            draw_preview(image_path, split, split_out / image_path.name)
        print(f"{split}: saved {sample_count} preview images to {split_out}")


if __name__ == "__main__":
    main()
