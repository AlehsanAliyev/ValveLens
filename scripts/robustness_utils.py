from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "valve", 1: "gauge"}


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def ensure_dir(path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def image_dir_for(source: str | Path) -> Path:
    source_path = repo_path(source)
    nested = source_path / "images"
    if nested.exists() and nested.is_dir():
        return nested
    return source_path


def label_dir_for_image_dir(image_dir: str | Path) -> Optional[Path]:
    image_path = repo_path(image_dir)
    if image_path.name.lower() == "images":
        sibling = image_path.parent / "labels"
        if sibling.exists():
            return sibling
    sibling = image_path / "labels"
    if sibling.exists():
        return sibling
    return None


def list_images(source: str | Path, limit: Optional[int] = None) -> List[Path]:
    image_dir = image_dir_for(source)
    if not image_dir.exists():
        return []
    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda item: item.name.lower())
    if limit is not None and limit > 0:
        return images[:limit]
    return images


def read_image(path: str | Path) -> np.ndarray:
    resolved = repo_path(path)
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {resolved}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(resolved), image)
    if not ok:
        raise ValueError(f"Could not write image: {resolved}")


def copy_matching_label(
    image_path: Path,
    source_label_dir: Optional[Path],
    target_label_dir: Path,
) -> bool:
    if source_label_dir is None:
        return False
    source_label = source_label_dir / f"{image_path.stem}.txt"
    if not source_label.exists():
        return False
    target_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_label, target_label_dir / source_label.name)
    return True


def blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_percent(image_bgr: np.ndarray, threshold: int = 245) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    glare_pixels = int(np.sum(gray >= threshold))
    return float(glare_pixels / max(1, gray.size))


def write_json(path: str | Path, payload: Dict) -> Path:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def write_csv(path: str | Path, rows: Iterable[Dict]) -> Path:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        resolved.write_text("", encoding="utf-8")
        return resolved
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def make_yolo_yaml(condition_root: Path, out_yaml: Path) -> Path:
    payload = (
        f"path: {condition_root.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "test: images\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: valve\n"
        "  1: gauge\n"
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(payload, encoding="utf-8")
    return out_yaml
