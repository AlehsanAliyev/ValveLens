from __future__ import annotations

import argparse
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


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    gamma = max(0.05, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((idx / 255.0) ** inv_gamma) * 255 for idx in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=7, sigmaColor=55, sigmaSpace=55)


def apply_unsharp(image: np.ndarray, alpha: float = 1.35) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def identity(image: np.ndarray, gamma: float) -> np.ndarray:
    _ = gamma
    return image


def clahe(image: np.ndarray, gamma: float) -> np.ndarray:
    _ = gamma
    return apply_clahe(image)


def gamma_variant(image: np.ndarray, gamma: float) -> np.ndarray:
    return apply_gamma(image, gamma=gamma)


def denoise_clahe(image: np.ndarray, gamma: float) -> np.ndarray:
    _ = gamma
    return apply_clahe(apply_denoise(image))


def sharpen_clahe(image: np.ndarray, gamma: float) -> np.ndarray:
    _ = gamma
    return apply_unsharp(apply_clahe(image))


VARIANTS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "none": identity,
    "clahe": clahe,
    "gamma": gamma_variant,
    "denoise_clahe": denoise_clahe,
    "sharpen_clahe": sharpen_clahe,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply classical preprocessing variants to a folder of images."
    )
    parser.add_argument("--source", required=True, help="Input folder or folder with images/.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANTS.keys()),
        help="Preprocessing variant.",
    )
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.65,
        help="Gamma value for the gamma variant. Values below 1 brighten.",
    )
    args = parser.parse_args()

    source_image_dir = image_dir_for(args.source)
    source_label_dir = label_dir_for_image_dir(source_image_dir)
    images = list_images(source_image_dir, limit=args.limit if args.limit > 0 else None)
    if not images:
        raise SystemExit(f"No source images found in {source_image_dir}")

    out_root = ensure_dir(args.out)
    out_images = ensure_dir(out_root / "images")
    out_labels = ensure_dir(out_root / "labels")
    transform = VARIANTS[args.variant]

    records = []
    copied_labels = 0
    for image_path in images:
        image = read_image(image_path)
        processed = transform(image, args.gamma)
        target = out_images / image_path.name
        write_image(target, processed)
        if copy_matching_label(image_path, source_label_dir, out_labels):
            copied_labels += 1
        records.append(
            {
                "file": image_path.name,
                "variant": args.variant,
                "blur_score": blur_score(processed),
                "glare_percent": glare_percent(processed),
            }
        )

    manifest = {
        "source": str(source_image_dir.relative_to(REPO_ROOT)),
        "variant": args.variant,
        "out": str(out_root.relative_to(REPO_ROOT)),
        "images": len(records),
        "labels_copied": copied_labels,
        "records": records,
    }
    manifest_path = write_json(out_root / "manifest.json", manifest)
    print(
        f"Preprocessed {len(records)} images with {args.variant}; "
        f"copied {copied_labels} labels."
    )
    print(f"Output: {out_root.relative_to(REPO_ROOT)}")
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
