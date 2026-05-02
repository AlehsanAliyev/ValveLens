from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

from robustness_utils import (
    REPO_ROOT,
    blur_score,
    copy_matching_label,
    ensure_dir,
    glare_percent,
    image_dir_for,
    label_dir_for_image_dir,
    list_images,
    read_image,
    write_image,
    write_json,
)


def low_light(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _ = rng
    dark = image.astype(np.float32) * 0.35
    return np.clip(dark, 0, 255).astype(np.uint8)


def blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _ = rng
    return cv2.GaussianBlur(image, (9, 9), 0)


def noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = image.astype(np.float32)
    noisy += rng.normal(0.0, 18.0, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def glare(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    overlay = image.copy().astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (
        int(rng.uniform(0.25, 0.75) * w),
        int(rng.uniform(0.20, 0.70) * h),
    )
    axes = (
        max(8, int(rng.uniform(0.10, 0.22) * w)),
        max(8, int(rng.uniform(0.05, 0.16) * h)),
    )
    angle = float(rng.uniform(-25, 25))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(10, w * 0.04))
    alpha = (glow.astype(np.float32) / 255.0)[:, :, None] * 0.75
    overlay = overlay * (1.0 - alpha) + 255.0 * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def low_contrast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _ = rng
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    adjusted = mean + (image.astype(np.float32) - mean) * 0.45
    return np.clip(adjusted, 0, 255).astype(np.uint8)


CORRUPTIONS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "low_light": low_light,
    "blur": blur,
    "noise": noise,
    "glare": glare,
    "low_contrast": low_contrast,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic robustness corruptions."
    )
    parser.add_argument(
        "--source",
        default="data/detection/combined/test/images",
        help="Source image folder.",
    )
    parser.add_argument(
        "--out",
        default="data/robustness/synthetic",
        help="Output root for corruption folders.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max images to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    source_images_dir = image_dir_for(args.source)
    source_label_dir = label_dir_for_image_dir(source_images_dir)
    images = list_images(source_images_dir, limit=args.limit if args.limit > 0 else None)
    if not images:
        raise SystemExit(f"No source images found in {source_images_dir}")

    out_root = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)
    summary = {
        "source": str(source_images_dir.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "limit": args.limit,
        "corruptions": {},
    }

    for name, transform in CORRUPTIONS.items():
        image_out = ensure_dir(out_root / name / "images")
        label_out = ensure_dir(out_root / name / "labels")
        records = []
        copied_labels = 0
        for image_path in images:
            image = read_image(image_path)
            corrupted = transform(image, rng)
            target = image_out / image_path.name
            write_image(target, corrupted)
            if copy_matching_label(image_path, source_label_dir, label_out):
                copied_labels += 1
            records.append(
                {
                    "file": image_path.name,
                    "blur_score": blur_score(corrupted),
                    "glare_percent": glare_percent(corrupted),
                }
            )
        write_json(out_root / name / "manifest.json", {"records": records})
        summary["corruptions"][name] = {
            "images": len(records),
            "labels_copied": copied_labels,
            "image_dir": str((out_root / name / "images").relative_to(REPO_ROOT)),
            "label_dir": str((out_root / name / "labels").relative_to(REPO_ROOT)),
        }

    summary_path = write_json("artifacts/robustness/synthetic_summary.json", summary)
    print("Synthetic corruption generation complete.")
    for name, item in summary["corruptions"].items():
        print(f"  - {name}: {item['images']} images, {item['labels_copied']} labels")
    print(f"Summary: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
