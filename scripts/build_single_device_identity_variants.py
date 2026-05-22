from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["clean", "low_light", "blur", "noise", "low_contrast", "glare", "occluded"]


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def resize_long_edge(image: np.ndarray, long_edge: int = 720) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def affine_variant(image: np.ndarray, rng: np.random.Generator, stronger: bool) -> np.ndarray:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-3.0, 3.0) if not stronger else rng.uniform(-6.0, 6.0))
    scale = float(rng.uniform(0.97, 1.03) if not stronger else rng.uniform(0.92, 1.06))
    tx = float(rng.uniform(-8, 8) if not stronger else rng.uniform(-18, 18))
    ty = float(rng.uniform(-8, 8) if not stronger else rng.uniform(-18, 18))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def low_light(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) * 0.55 - 10, 0, 255).astype(np.uint8)


def low_contrast(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    mean = np.mean(arr, axis=(0, 1), keepdims=True)
    return np.clip(mean + (arr - mean) * 0.48, 0, 255).astype(np.uint8)


def add_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = image.astype(np.float32) + rng.normal(0, 14, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_glare(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.astype(np.float32)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(rng.uniform(0.20, 0.80) * w), int(rng.uniform(0.18, 0.70) * h))
    axes = (int(rng.uniform(0.10, 0.22) * w), int(rng.uniform(0.05, 0.13) * h))
    cv2.ellipse(mask, center, axes, float(rng.uniform(-25, 25)), 0, 360, 255, -1)
    glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(10, w * 0.045))
    alpha = (glow.astype(np.float32) / 255.0)[:, :, None] * 0.55
    return np.clip(out * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)


def add_occlusion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    # Keep this modest so the real tag usually remains partly readable.
    x1 = int(rng.uniform(0.05, 0.72) * w)
    y1 = int(rng.uniform(0.05, 0.72) * h)
    x2 = min(w - 1, x1 + int(rng.uniform(0.08, 0.18) * w))
    y2 = min(h - 1, y1 + int(rng.uniform(0.06, 0.16) * h))
    cv2.rectangle(out, (x1, y1), (x2, y2), (45, 45, 45), -1)
    return out


def apply_condition(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return image.copy()
    if condition == "low_light":
        return low_light(image)
    if condition == "blur":
        return cv2.GaussianBlur(image, (7, 7), 0)
    if condition == "noise":
        return add_noise(image, rng)
    if condition == "low_contrast":
        return low_contrast(image)
    if condition == "glare":
        return add_glare(image, rng)
    if condition == "occluded":
        return add_occlusion(image, rng)
    return image.copy()


def write_devices_manifest(path: Path, device_id: str, device_type: str, zone_id: str) -> None:
    rows = [
        {
            "device_id": device_id,
            "type": device_type,
            "zone_id": zone_id,
            "description": f"Manual real-tag identity source for {device_id}",
            "has_visible_tag": "true",
        }
    ]
    write_csv(path, rows)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    source = repo_path(args.source)
    out = repo_path(args.out)
    if not source.exists():
        raise SystemExit(f"Source image not found: {source}")

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read source image: {source}")
    image = resize_long_edge(image, args.long_edge)
    rng = np.random.default_rng(args.seed)

    refs_dir = out / "refs" / args.device_id
    queries_dir = out / "queries" / args.device_id
    base_dir = out / "base"
    write_image(base_dir / "source_preserved_tag.jpg", image)

    for idx in range(1, args.refs + 1):
        variant = affine_variant(image, rng, stronger=False)
        variant = apply_condition(variant, "clean", rng)
        write_image(refs_dir / f"ref{idx:03d}.jpg", variant)

    query_rows: List[Dict[str, str]] = []
    query_index = 0
    for condition in CONDITIONS:
        for _ in range(args.queries_per_condition):
            query_index += 1
            variant = affine_variant(image, rng, stronger=True)
            variant = apply_condition(variant, condition, rng)
            path = queries_dir / condition / f"q{query_index:03d}.jpg"
            write_image(path, variant)
            query_rows.append(
                {
                    "image_path": str(path.relative_to(REPO_ROOT)),
                    "expected_device_id": args.device_id,
                    "expected_type": args.device_type,
                    "condition": condition,
                    "tag_visible": "true" if condition != "occluded" else "partial",
                    "expected_zone": args.zone_id,
                }
            )

    write_devices_manifest(out / "devices_manifest.csv", args.device_id, args.device_type, args.zone_id)
    write_csv(out / "queries_manifest.csv", query_rows)
    metadata = {
        "source": str(source),
        "device_id": args.device_id,
        "note": "Variants preserve the visible source tag; no synthetic tag was drawn.",
        "refs": args.refs,
        "queries": len(query_rows),
        "conditions": CONDITIONS,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Single-device identity variants generated.")
    print(f"  source: {source}")
    print(f"  output: {out}")
    print(f"  refs: {args.refs}")
    print(f"  queries: {len(query_rows)}")
    print("  note: preserved existing visible tag; no synthetic tag added")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference/query identity variants from one real tagged image.")
    parser.add_argument("--source", required=True, help="Path to the real source image with the visible tag already present.")
    parser.add_argument("--out", default="data/device_benchmark/manual_v1023")
    parser.add_argument("--device-id", default="V-1023")
    parser.add_argument("--device-type", default="valve")
    parser.add_argument("--zone-id", default="manual-zone")
    parser.add_argument("--refs", type=int, default=8)
    parser.add_argument("--queries-per-condition", type=int, default=2)
    parser.add_argument("--long-edge", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
