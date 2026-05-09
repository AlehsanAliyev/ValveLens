from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def list_images(path: Path, limit: int) -> List[Path]:
    if not path.exists():
        return []
    images = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda item: item.name.lower())[:limit]


def read_thumb(path: Path, size: int = 180) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((size, size, 3), 220, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def caption(image: np.ndarray, text: str) -> np.ndarray:
    pad = 34
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3), 250, dtype=np.uint8)
    canvas[pad:, :, :] = image
    cv2.putText(canvas, text[:28], (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def make_sheet(items: List[Tuple[str, Path]], columns: int = 4) -> np.ndarray:
    tiles = [caption(read_thumb(path), title) for title, path in items]
    if not tiles:
        return np.full((220, 220, 3), 245, dtype=np.uint8)
    gap = 12
    tile_h, tile_w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.full(
        (rows * tile_h + (rows - 1) * gap, columns * tile_w + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        y = row * (tile_h + gap)
        x = col * (tile_w + gap)
        canvas[y : y + tile_h, x : x + tile_w] = tile
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create proxy device benchmark contact sheets.")
    parser.add_argument("--root", default="data/device_benchmark")
    parser.add_argument("--out", default="artifacts/identity_benchmark/proxy_preview")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    root = repo_path(args.root)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_root = root / "refs"
    queries_root = root / "queries"
    base_root = root / "base_crops"

    if not refs_root.exists():
        raise SystemExit(f"Refs folder not found: {refs_root}")

    saved = 0
    for device_dir in sorted([item for item in refs_root.iterdir() if item.is_dir()], key=lambda item: item.name):
        device_id = device_dir.name
        items: List[Tuple[str, Path]] = []
        base = base_root / device_id / "base.jpg"
        if base.exists():
            items.append(("base crop", base))
        for ref in list_images(device_dir, args.limit):
            items.append((f"ref {ref.stem}", ref))
        query_device_root = queries_root / device_id
        if query_device_root.exists():
            for condition_dir in sorted([item for item in query_device_root.iterdir() if item.is_dir()], key=lambda item: item.name):
                for query in list_images(condition_dir, 1):
                    items.append((f"{condition_dir.name} {query.stem}", query))
        if not items:
            continue
        sheet = make_sheet(items)
        out_path = out_dir / f"{device_id}_proxy_preview.jpg"
        cv2.imwrite(str(out_path), sheet)
        saved += 1

    print(f"Saved {saved} proxy preview sheets.")
    print(f"Preview folder: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
