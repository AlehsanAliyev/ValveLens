from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from robustness_utils import REPO_ROOT, ensure_dir, image_dir_for, list_images, read_image, repo_path, write_image


def _resize_to_height(image: np.ndarray, height: int = 220) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / max(1, h)
    width = max(1, int(w * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    pad = 32
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3), 245, dtype=np.uint8)
    canvas[pad:, :, :] = image
    cv2.putText(
        canvas,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _make_row(items: List[tuple[str, np.ndarray]]) -> np.ndarray:
    captioned = [_caption(_resize_to_height(image), title) for title, image in items]
    height = max(item.shape[0] for item in captioned)
    padded = []
    for item in captioned:
        if item.shape[0] < height:
            bottom = np.full((height - item.shape[0], item.shape[1], 3), 245, dtype=np.uint8)
            item = np.vstack([item, bottom])
        padded.append(item)
    gap = np.full((height, 12, 3), 255, dtype=np.uint8)
    row = padded[0]
    for item in padded[1:]:
        row = np.hstack([row, gap, item])
    return row


def _matching_preprocessed(condition: str, explicit: Optional[str]) -> List[Path]:
    if explicit:
        return [repo_path(explicit)]
    root = repo_path("data/robustness/preprocessed")
    if not root.exists():
        return []
    return [
        child
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir() and child.name.startswith(f"{condition}_")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create side-by-side robustness preprocessing examples."
    )
    parser.add_argument(
        "--original",
        default="data/detection/combined/test/images",
        help="Original image folder.",
    )
    parser.add_argument(
        "--corrupted",
        default="data/robustness/synthetic",
        help="Synthetic corruption root.",
    )
    parser.add_argument(
        "--preprocessed",
        default=None,
        help="Optional specific preprocessed folder.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/robustness/preprocessing_preview",
        help="Output preview folder.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Max examples per condition.")
    args = parser.parse_args()

    original_dir = image_dir_for(args.original)
    corrupted_root = repo_path(args.corrupted)
    out_dir = ensure_dir(args.out)
    if not corrupted_root.exists():
        raise SystemExit(f"Synthetic corruption root not found: {corrupted_root}")

    saved = 0
    for condition_dir in sorted(corrupted_root.iterdir(), key=lambda item: item.name.lower()):
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        corrupted_images = list_images(condition_dir, limit=args.limit)
        if not corrupted_images:
            continue
        preprocessed_dirs = _matching_preprocessed(condition, args.preprocessed)
        if not preprocessed_dirs:
            preprocessed_dirs = [None]

        for pre_dir in preprocessed_dirs:
            variant_name = pre_dir.name if pre_dir is not None else "no_preprocess"
            for corrupted_path in corrupted_images:
                original_path = original_dir / corrupted_path.name
                if not original_path.exists():
                    continue
                items = [
                    ("original", read_image(original_path)),
                    (condition, read_image(corrupted_path)),
                ]
                if pre_dir is not None:
                    processed_path = image_dir_for(pre_dir) / corrupted_path.name
                    if processed_path.exists():
                        items.append((variant_name, read_image(processed_path)))
                row = _make_row(items)
                out_name = f"{condition}_{variant_name}_{corrupted_path.stem}.jpg"
                write_image(out_dir / out_name, row)
                saved += 1

    print(f"Saved {saved} preview images.")
    print(f"Preview folder: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
