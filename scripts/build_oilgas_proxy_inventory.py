from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONDITIONS = ["clean", "low_light", "glare", "blur", "noise", "low_contrast", "occluded"]

DEFAULT_DATASETS = [
    "data/detection/oilgas_expanded/elementos_offshore",
    "data/detection/oilgas_expanded/oil_refinery",
    "data/detection/combined",
]

DEVICE_PLANS = [
    {"device_id": "V-1023", "type": "valve", "class_name": "valve", "description": "Proxy tagged valve"},
    {"device_id": "V-1212", "type": "valve", "class_name": "valve", "description": "Second proxy tagged valve"},
    {"device_id": "PG-45", "type": "gauge", "class_name": "gauge", "description": "Proxy pressure gauge"},
    {"device_id": "FL-101", "type": "flange", "class_name": "flange", "description": "Proxy flange"},
    {"device_id": "PT-101", "type": "instrument", "class_name": "instrument", "description": "Proxy instrument or transmitter"},
    {"device_id": "PV-201", "type": "vessel", "class_name": "vessel", "description": "Proxy pressure vessel"},
    {"device_id": "HX-301", "type": "heat_exchanger", "class_name": "heat_exchanger", "description": "Proxy heat exchanger"},
    {"device_id": "DS-101", "type": "desalter", "class_name": "desalter", "description": "Proxy desalter"},
    {"device_id": "HTR-101", "type": "heater", "class_name": "heater", "description": "Proxy heater"},
    {"device_id": "TK-101", "type": "tank_or_cylinder", "class_name": "tank_or_cylinder", "description": "Proxy tank or cylinder"},
]

SKIP_INVENTORY_CLASSES = {"pipe", "structure", "person", "support", "equipment", "ladder"}
CLASS_PRIORITY = {
    "valve": 100,
    "gauge": 95,
    "flange": 90,
    "instrument": 88,
    "vessel": 86,
    "heat_exchanger": 84,
    "desalter": 82,
    "heater": 80,
    "tank_or_cylinder": 78,
}


@dataclass
class CandidateCrop:
    class_id: int
    class_name: str
    dataset_root: str
    source_image: str
    bbox_xyxy: Tuple[int, int, int, int]
    box_area_ratio: float
    crop: np.ndarray
    score: float


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def read_class_names(dataset_root: Path) -> Dict[int, str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    names: Dict[int, str] = {}
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key.isdigit():
            names[int(key)] = value.strip().strip("'\"")
    return names


def iter_split_dirs(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for split in ("train", "valid", "test"):
        path = dataset_root / split
        if (path / "images").exists() and (path / "labels").exists():
            yield split, path


def iter_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: str(item).lower(),
    )


def parse_label_line(raw: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = raw.strip().split()
    if len(parts) != 5:
        return None
    try:
        return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def yolo_to_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1 = int(round((xc - bw / 2.0) * width))
    y1 = int(round((yc - bh / 2.0) * height))
    x2 = int(round((xc + bw / 2.0) * width))
    y2 = int(round((yc + bh / 2.0) * height))
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def resize_pad(image: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("invalid crop")
    scale = min((size * 0.82) / w, (size * 0.82) / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 224, dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def crop_score(class_name: str, box_w: int, box_h: int, image_area: int, dataset_root: Path) -> float:
    area_ratio = (box_w * box_h) / max(1, image_area)
    target = 0.18
    area_score = max(0.0, 1.0 - abs(area_ratio - target) / target)
    size_score = min(1.0, min(box_w, box_h) / 120.0)
    priority = CLASS_PRIORITY.get(class_name, 0) / 100.0
    dataset_bonus = 0.10 if "elementos_offshore" in str(dataset_root).lower() else 0.0
    return priority * 2.0 + area_score + size_score + dataset_bonus


def collect_candidates(
    dataset_roots: List[Path],
    wanted_classes: set[str],
    min_box_size: int,
    crop_expand: float,
) -> Dict[str, List[CandidateCrop]]:
    by_class: Dict[str, List[CandidateCrop]] = {name: [] for name in wanted_classes}
    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            print(f"WARNING: dataset missing: {dataset_root}")
            continue
        class_names = read_class_names(dataset_root)
        if not class_names:
            print(f"WARNING: data.yaml classes missing: {dataset_root}")
            continue
        for _, split_dir in iter_split_dirs(dataset_root):
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            for image_path in iter_images(images_dir):
                label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
                if not label_path.exists():
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                h, w = image.shape[:2]
                image_area = max(1, h * w)
                for raw in label_path.read_text(encoding="utf-8").splitlines():
                    parsed = parse_label_line(raw)
                    if parsed is None:
                        continue
                    cls_id, xc, yc, bw, bh = parsed
                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                    if class_name in SKIP_INVENTORY_CLASSES or class_name not in wanted_classes:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
                    box_w = x2 - x1 + 1
                    box_h = y2 - y1 + 1
                    if box_w < min_box_size or box_h < min_box_size:
                        continue
                    area_ratio = (box_w * box_h) / image_area
                    if area_ratio < 0.006 or area_ratio > 0.82:
                        continue
                    pad_x = int(round(box_w * max(0.0, crop_expand)))
                    pad_y = int(round(box_h * max(0.0, crop_expand)))
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w - 1, x2 + pad_x)
                    cy2 = min(h - 1, y2 + pad_y)
                    crop = image[cy1 : cy2 + 1, cx1 : cx2 + 1]
                    if crop.size == 0:
                        continue
                    try:
                        crop = resize_pad(crop)
                    except ValueError:
                        continue
                    try:
                        source_rel = str(image_path.relative_to(REPO_ROOT))
                        dataset_rel = str(dataset_root.relative_to(REPO_ROOT))
                    except ValueError:
                        source_rel = str(image_path)
                        dataset_rel = str(dataset_root)
                    by_class[class_name].append(
                        CandidateCrop(
                            class_id=cls_id,
                            class_name=class_name,
                            dataset_root=dataset_rel,
                            source_image=source_rel,
                            bbox_xyxy=(cx1, cy1, cx2, cy2),
                            box_area_ratio=area_ratio,
                            crop=crop,
                            score=crop_score(class_name, box_w, box_h, image_area, dataset_root),
                        )
                    )
    for class_name, candidates in by_class.items():
        candidates.sort(key=lambda item: item.score, reverse=True)
    return by_class


def affine_variant(image: np.ndarray, rng: np.random.Generator, stronger: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-7, 7) if stronger else rng.uniform(-3.5, 3.5))
    scale = float(rng.uniform(0.93, 1.08) if stronger else rng.uniform(0.96, 1.04))
    tx = float(rng.uniform(-12, 12) if stronger else rng.uniform(-5, 5))
    ty = float(rng.uniform(-12, 12) if stronger else rng.uniform(-5, 5))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def perspective_variant(image: np.ndarray, rng: np.random.Generator, max_shift: float = 0.03) -> np.ndarray:
    h, w = image.shape[:2]
    shift = max_shift * min(h, w)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.uniform(-shift, shift, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_color(image: np.ndarray, rng: np.random.Generator, condition: str = "clean") -> np.ndarray:
    out = image.astype(np.float32)
    if condition == "low_light":
        alpha = float(rng.uniform(0.66, 0.82))
        beta = float(rng.uniform(-8, 2))
    elif condition == "low_contrast":
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        return np.clip(mean + (out - mean) * float(rng.uniform(0.62, 0.78)), 0, 255).astype(np.uint8)
    else:
        alpha = float(rng.uniform(0.97, 1.05))
        beta = float(rng.uniform(-4, 4))
    return np.clip(out * alpha + beta, 0, 255).astype(np.uint8)


def apply_condition(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    if condition == "clean":
        return out
    if condition == "low_light":
        return out
    if condition == "glare":
        h, w = out.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(rng.uniform(0.22, 0.78) * w), int(rng.uniform(0.18, 0.78) * h))
        axes = (int(rng.uniform(0.07, 0.15) * w), int(rng.uniform(0.04, 0.11) * h))
        cv2.ellipse(mask, center, axes, float(rng.uniform(-25, 25)), 0, 360, 255, -1)
        glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, w * 0.04))
        alpha = (glow.astype(np.float32) / 255.0)[:, :, None] * 0.38
        return np.clip(out.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)
    if condition == "blur":
        return cv2.GaussianBlur(out, (3, 3), 0)
    if condition == "noise":
        noisy = out.astype(np.float32) + rng.normal(0, 7, out.shape).astype(np.float32)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if condition == "low_contrast":
        return adjust_color(out, rng, "low_contrast")
    if condition == "occluded":
        h, w = out.shape[:2]
        x1 = int(rng.uniform(0.18, 0.58) * w)
        y1 = int(rng.uniform(0.18, 0.58) * h)
        x2 = min(w - 1, x1 + int(rng.uniform(0.08, 0.16) * w))
        y2 = min(h - 1, y1 + int(rng.uniform(0.08, 0.16) * h))
        color = int(rng.uniform(55, 115))
        cv2.rectangle(out, (x1, y1), (x2, y2), (color, color, color), -1)
        return out
    return out


def make_tag_patch(device_id: str, scale: float = 1.0, thickness: int = 3) -> np.ndarray:
    font_scale = 1.16 * max(0.6, scale)
    thickness = max(2, int(thickness))
    (text_w, text_h), baseline = cv2.getTextSize(device_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad_x = 34
    pad_y = 24
    patch_w = max(250, text_w + pad_x * 2)
    patch_h = max(96, text_h + baseline + pad_y * 2)
    patch = np.full((patch_h, patch_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(patch, (0, 0), (patch_w - 1, patch_h - 1), (0, 0, 0), 3)
    x = (patch_w - text_w) // 2
    y = (patch_h + text_h) // 2 - baseline
    cv2.putText(patch, device_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return patch


def add_tag(image: np.ndarray, device_id: str, rng: np.random.Generator, scale: float, thickness: int) -> np.ndarray:
    out = image.copy()
    patch = make_tag_patch(device_id, scale=scale, thickness=thickness)
    h, w = out.shape[:2]
    ph, pw = patch.shape[:2]
    margin = 42
    positions = [
        (margin, margin),
        (max(margin, w - pw - margin), margin),
        (margin, max(margin, h - ph - margin)),
        (max(margin, w - pw - margin), max(margin, h - ph - margin)),
    ]
    x, y = positions[int(rng.integers(0, len(positions)))]
    x = max(0, min(w - pw, x))
    y = max(0, min(h - ph, y))
    out[y : y + ph, x : x + pw] = patch
    return out


def generate_variant(
    base: np.ndarray,
    device_id: str,
    condition: str,
    rng: np.random.Generator,
    tag_visible: bool,
    query: bool,
    tag_scale: float,
    tag_thickness: int,
) -> np.ndarray:
    out = base.copy()
    if condition == "clean":
        if not query or rng.random() < 0.25:
            out = affine_variant(out, rng, stronger=False)
    else:
        out = affine_variant(out, rng, stronger=False)
        if rng.random() < 0.35:
            out = perspective_variant(out, rng, max_shift=0.018)
    color_profile = condition if condition in {"low_light", "low_contrast"} else "clean"
    out = adjust_color(out, rng, color_profile)
    out = apply_condition(out, condition, rng)
    if tag_visible:
        out = add_tag(out, device_id, rng, scale=tag_scale, thickness=tag_thickness)
    return out


def distribute_conditions(total: int) -> List[str]:
    if total <= 0:
        return []
    return [CONDITIONS[idx % len(CONDITIONS)] for idx in range(total)]


def ensure_output(root: Path, overwrite: bool) -> None:
    if overwrite:
        for child in ["refs", "queries", "base_crops"]:
            target = root / child
            if target.exists():
                shutil.rmtree(target)
        for name in ["devices_manifest.csv", "queries_manifest.csv", "oilgas_proxy_inventory_metadata.json"]:
            path = root / name
            if path.exists():
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)
    if not overwrite and ((root / "refs").exists() or (root / "queries").exists()):
        print(f"WARNING: {root} already has benchmark folders; use --overwrite to rebuild cleanly.")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_devices(by_class: Dict[str, List[CandidateCrop]], rng: np.random.Generator) -> List[Tuple[Dict, CandidateCrop]]:
    selected: List[Tuple[Dict, CandidateCrop]] = []
    used_sources = set()
    for plan in DEVICE_PLANS:
        class_name = plan["class_name"]
        pool = [candidate for candidate in by_class.get(class_name, []) if candidate.source_image not in used_sources]
        if not pool:
            pool = by_class.get(class_name, [])
        if not pool:
            print(f"WARNING: no crop found for {plan['device_id']} ({class_name}); skipping.")
            continue
        top_pool = pool[: max(1, min(20, len(pool)))]
        crop = top_pool[int(rng.integers(0, len(top_pool)))]
        selected.append((plan, crop))
        used_sources.add(crop.source_image)
    return selected


def make_preview(records: List[Dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: List[Image.Image] = []
    for record in records:
        base_path = repo_path(record["base_crop_path"])
        if not base_path.exists():
            continue
        with Image.open(base_path) as image:
            image = image.convert("RGB")
            image.thumbnail((220, 180))
            tile = Image.new("RGB", (240, 230), (245, 245, 245))
            tile.paste(image, ((240 - image.width) // 2, 12))
            draw = ImageDraw.Draw(tile)
            draw.text((10, 195), record["device_id"], fill=(0, 0, 0))
            draw.text((10, 212), record["type"][:24], fill=(50, 50, 50))
            tiles.append(tile)
    cols = 4
    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 240, rows * 230), (255, 255, 255))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 240, (idx // cols) * 230))
    out_path = out_dir / "oilgas_proxy_inventory_contact_sheet.jpg"
    sheet.save(out_path)
    return out_path


def build_inventory(args: argparse.Namespace) -> Dict:
    rng = np.random.default_rng(args.seed)
    out_root = repo_path(args.out)
    ensure_output(out_root, args.overwrite)

    dataset_roots = [repo_path(path) for path in args.datasets]
    wanted_classes = {plan["class_name"] for plan in DEVICE_PLANS}
    by_class = collect_candidates(
        dataset_roots,
        wanted_classes=wanted_classes,
        min_box_size=args.min_box_size,
        crop_expand=args.crop_expand,
    )
    selected = select_devices(by_class, rng)
    if not selected:
        raise SystemExit("No usable inventory crops found. Check prepared oil/gas datasets first.")

    devices_manifest: List[Dict[str, str]] = []
    queries_manifest: List[Dict[str, str]] = []
    metadata_records: List[Dict] = []
    ref_total = 0
    query_total = 0
    tag_visible_queries = 0
    no_tag_queries = 0

    for plan, crop in selected:
        device_id = plan["device_id"]
        device_type = plan["type"]
        base_dir = out_root / "base_crops" / device_id
        base_path = base_dir / "base.jpg"
        write_image(base_path, crop.crop)

        devices_manifest.append(
            {
                "device_id": device_id,
                "type": device_type,
                "zone_id": args.zone_id,
                "description": f"{plan['description']} from {crop.dataset_root}",
                "has_visible_tag": "true",
            }
        )

        ref_dir = out_root / "refs" / device_id
        for idx in range(1, args.refs_per_device + 1):
            tag_visible = idx <= max(1, math.ceil(args.refs_per_device * 0.7))
            image = generate_variant(
                crop.crop,
                device_id,
                "clean",
                rng,
                tag_visible=tag_visible,
                query=False,
                tag_scale=args.tag_scale,
                tag_thickness=args.tag_thickness,
            )
            write_image(ref_dir / f"ref{idx:03d}.jpg", image)
            ref_total += 1

        per_condition_idx = {condition: 0 for condition in CONDITIONS}
        for condition in distribute_conditions(args.queries_per_device):
            per_condition_idx[condition] += 1
            if condition == "occluded":
                tag_visible = False
            elif condition in {"clean", "low_light"} and per_condition_idx[condition] == 1:
                tag_visible = True
            else:
                tag_visible = rng.random() < args.tagged_query_ratio
            image = generate_variant(
                crop.crop,
                device_id,
                condition,
                rng,
                tag_visible=tag_visible,
                query=True,
                tag_scale=args.tag_scale,
                tag_thickness=args.tag_thickness,
            )
            filename = f"q{per_condition_idx[condition]:03d}.jpg"
            query_path = out_root / "queries" / device_id / condition / filename
            write_image(query_path, image)
            queries_manifest.append(
                {
                    "image_path": str(query_path.relative_to(REPO_ROOT)),
                    "expected_device_id": device_id,
                    "expected_type": device_type,
                    "condition": condition,
                    "tag_visible": str(tag_visible).lower(),
                    "expected_zone": args.zone_id,
                }
            )
            query_total += 1
            if tag_visible:
                tag_visible_queries += 1
            else:
                no_tag_queries += 1

        metadata_records.append(
            {
                "device_id": device_id,
                "type": device_type,
                "class_name": crop.class_name,
                "dataset_root": crop.dataset_root,
                "source_image": crop.source_image,
                "source_bbox_xyxy": list(crop.bbox_xyxy),
                "box_area_ratio": crop.box_area_ratio,
                "selection_score": crop.score,
                "base_crop_path": str(base_path.relative_to(REPO_ROOT)),
            }
        )

    devices_path = out_root / "devices_manifest.csv"
    queries_path = out_root / "queries_manifest.csv"
    metadata_path = out_root / "oilgas_proxy_inventory_metadata.json"
    write_csv(devices_path, devices_manifest)
    write_csv(queries_path, queries_manifest)
    metadata = {
        "note": "Controlled proxy inventory benchmark for OCR/ReID/fusion mechanics. Clean samples use gentle augmentation; degraded samples use mild visual conditions. Not real industrial identity validation.",
        "zone_id": args.zone_id,
        "datasets": [str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path) for path in dataset_roots],
        "generated_devices": [record["device_id"] for record in metadata_records],
        "records": metadata_records,
        "counts": {
            "devices": len(metadata_records),
            "reference_images": ref_total,
            "query_images": query_total,
            "tag_visible_queries": tag_visible_queries,
            "no_tag_queries": no_tag_queries,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    preview_dir = repo_path(args.preview_out)
    preview_path = make_preview(metadata_records, preview_dir)

    return {
        "generated_devices": metadata["generated_devices"],
        "reference_images": ref_total,
        "query_images": query_total,
        "tag_visible_queries": tag_visible_queries,
        "no_tag_queries": no_tag_queries,
        "devices_manifest": str(devices_path.relative_to(REPO_ROOT)),
        "queries_manifest": str(queries_path.relative_to(REPO_ROOT)),
        "metadata": str(metadata_path.relative_to(REPO_ROOT)),
        "preview": str(preview_path.relative_to(REPO_ROOT)),
        "candidate_counts": {key: len(value) for key, value in by_class.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a controlled oil/gas proxy inventory benchmark.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out", default="data/device_benchmark")
    parser.add_argument("--preview-out", default="artifacts/identity_benchmark/oilgas_proxy_inventory_preview")
    parser.add_argument("--refs-per-device", type=int, default=8)
    parser.add_argument("--queries-per-device", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zone-id", default="<ZONE_ID>")
    parser.add_argument("--min-box-size", type=int, default=24)
    parser.add_argument("--crop-expand", type=float, default=0.45)
    parser.add_argument("--tagged-query-ratio", type=float, default=0.75)
    parser.add_argument("--tag-scale", type=float, default=1.15)
    parser.add_argument("--tag-thickness", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tagged_query_ratio = max(0.0, min(1.0, args.tagged_query_ratio))
    summary = build_inventory(args)

    print("Oil/gas proxy inventory benchmark built.")
    print(f"  generated devices: {', '.join(summary['generated_devices'])}")
    print(f"  reference images: {summary['reference_images']}")
    print(f"  query images: {summary['query_images']}")
    print(f"  tag-visible queries: {summary['tag_visible_queries']}")
    print(f"  no-tag queries: {summary['no_tag_queries']}")
    print(f"  devices manifest: {summary['devices_manifest']}")
    print(f"  queries manifest: {summary['queries_manifest']}")
    print(f"  metadata: {summary['metadata']}")
    print(f"  preview: {summary['preview']}")
    print("  candidate counts:")
    for class_name, count in sorted(summary["candidate_counts"].items()):
        print(f"    - {class_name}: {count}")

    print("\nNext commands:")
    print("  cd D:\\python_works\\ValveLens\\backend")
    print("  python -m app.cli.enroll_devices_from_manifest --manifest ..\\data\\device_benchmark\\devices_manifest.csv --refs-root ..\\data\\device_benchmark\\refs --force-add-refs")
    print("  python -m app.cli.rebuild_device_index")
    print("  python -m app.cli.validate_identity_benchmark --queries-manifest ..\\data\\device_benchmark\\queries_manifest.csv --topk 5 --out ..\\artifacts\\identity_benchmark")
    print("  python -m app.cli.smoke_reid --image \"..\\data\\device_benchmark\\queries\\V-1023\\clean\\q001.jpg\" --topk 5")
    print("  python -m app.cli.smoke_ocr --image \"..\\data\\device_benchmark\\queries\\V-1023\\clean\\q001.jpg\" --expected V-1023")

    print("\nClaim boundary:")
    print("  This validates a controlled proxy inventory path for OCR/ReID/fusion mechanics.")
    print("  It is not real industrial identity validation. Real physical device photos are still needed.")


if __name__ == "__main__":
    main()
