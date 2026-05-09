from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE_PLAN = [
    {"device_id": "V-1023", "type": "valve", "class_id": 0, "has_visible_tag": True},
    {"device_id": "V-2040", "type": "valve", "class_id": 0, "has_visible_tag": True},
    {"device_id": "PG-45", "type": "gauge", "class_id": 1, "has_visible_tag": True},
]
CONDITIONS = ["clean", "low_light", "glare", "blur", "noise", "low_contrast", "occluded"]


@dataclass
class CandidateCrop:
    class_id: int
    class_name: str
    source_image: str
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    crop: np.ndarray


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def read_class_names(dataset_root: Path) -> Dict[int, str]:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return {0: "valve", 1: "gauge"}
    names: Dict[int, str] = {}
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key.isdigit():
            continue
        names[int(key)] = value.strip().strip("\"'")
    return names or {0: "valve", 1: "gauge"}


def iter_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda item: item.name.lower(),
    )


def parse_yolo_label(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
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
        raise ValueError("invalid crop size")
    scale = min((size * 0.82) / w, (size * 0.82) / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 222, dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def collect_crops(
    images_dir: Path,
    labels_dir: Path,
    class_names: Dict[int, str],
    min_size: int,
    crop_expand: float,
) -> List[CandidateCrop]:
    crops: List[CandidateCrop] = []
    for image_path in iter_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_label(raw)
            if parsed is None:
                continue
            cls_id, xc, yc, bw, bh = parsed
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            if box_w < min_size or box_h < min_size:
                continue
            pad_x = int(round(box_w * max(0.0, crop_expand)))
            pad_y = int(round(box_h * max(0.0, crop_expand)))
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(width - 1, x2 + pad_x)
            crop_y2 = min(height - 1, y2 + pad_y)
            crop = image[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
            if crop.size == 0:
                continue
            crops.append(
                CandidateCrop(
                    class_id=cls_id,
                    class_name=class_names.get(cls_id, f"class_{cls_id}"),
                    source_image=str(image_path.relative_to(REPO_ROOT)),
                    bbox_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
                    area=int(box_w * box_h),
                    crop=resize_pad(crop),
                )
            )
    return crops


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_output(root: Path, overwrite: bool) -> None:
    if overwrite:
        for child in ["refs", "queries"]:
            target = root / child
            if target.exists():
                shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    for child in ["refs", "queries"]:
        target = root / child
        if target.exists() and any(target.rglob("*")) and not overwrite:
            print(f"WARNING: {target.relative_to(REPO_ROOT)} already contains files; generated files may be overwritten.")


def perspective_variant(image: np.ndarray, rng: np.random.Generator, max_shift: float = 0.035) -> np.ndarray:
    h, w = image.shape[:2]
    shift = max_shift * min(h, w)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.uniform(-shift, shift, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def affine_variant(image: np.ndarray, rng: np.random.Generator, stronger: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-8, 8) if stronger else rng.uniform(-4, 4))
    scale = float(rng.uniform(0.92, 1.08) if stronger else rng.uniform(0.96, 1.04))
    tx = float(rng.uniform(-12, 12) if stronger else rng.uniform(-6, 6))
    ty = float(rng.uniform(-12, 12) if stronger else rng.uniform(-6, 6))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def adjust_color(image: np.ndarray, rng: np.random.Generator, condition: str = "clean") -> np.ndarray:
    out = image.astype(np.float32)
    if condition == "low_light":
        alpha = float(rng.uniform(0.45, 0.65))
        beta = float(rng.uniform(-18, -5))
    elif condition == "low_contrast":
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        return np.clip(mean + (out - mean) * float(rng.uniform(0.42, 0.62)), 0, 255).astype(np.uint8)
    else:
        alpha = float(rng.uniform(0.86, 1.18))
        beta = float(rng.uniform(-14, 14))
    return np.clip(out * alpha + beta, 0, 255).astype(np.uint8)


def apply_condition(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    if condition == "clean":
        return adjust_color(out, rng, "clean")
    if condition == "low_light":
        return adjust_color(out, rng, "low_light")
    if condition == "glare":
        h, w = out.shape[:2]
        overlay = out.astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(rng.uniform(0.25, 0.75) * w), int(rng.uniform(0.20, 0.75) * h))
        axes = (int(rng.uniform(0.12, 0.22) * w), int(rng.uniform(0.06, 0.16) * h))
        cv2.ellipse(mask, center, axes, float(rng.uniform(-25, 25)), 0, 360, 255, -1)
        glow = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(10, w * 0.045))
        alpha = (glow.astype(np.float32) / 255.0)[:, :, None] * 0.70
        return np.clip(overlay * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)
    if condition == "blur":
        return cv2.GaussianBlur(out, (7, 7), 0)
    if condition == "noise":
        noisy = out.astype(np.float32) + rng.normal(0, 13, out.shape).astype(np.float32)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if condition == "low_contrast":
        return adjust_color(out, rng, "low_contrast")
    if condition == "occluded":
        h, w = out.shape[:2]
        x1 = int(rng.uniform(0.15, 0.55) * w)
        y1 = int(rng.uniform(0.15, 0.55) * h)
        x2 = min(w - 1, x1 + int(rng.uniform(0.14, 0.26) * w))
        y2 = min(h - 1, y1 + int(rng.uniform(0.12, 0.24) * h))
        color = int(rng.uniform(35, 95))
        cv2.rectangle(out, (x1, y1), (x2, y2), (color, color, color), -1)
        return out
    return out


def make_tag_patch(device_id: str, rng: np.random.Generator, degraded: bool = False) -> np.ndarray:
    patch = np.full((54, 154, 3), 250, dtype=np.uint8)
    cv2.rectangle(patch, (0, 0), (153, 53), (20, 20, 20), 2)
    cv2.putText(patch, device_id, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (15, 15, 15), 2, cv2.LINE_AA)
    if degraded:
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        patch = np.clip(patch.astype(np.float32) * float(rng.uniform(0.78, 0.92)), 0, 255).astype(np.uint8)
    angle = float(rng.uniform(-6, 6) if not degraded else rng.uniform(-12, 12))
    matrix = cv2.getRotationMatrix2D((patch.shape[1] / 2, patch.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]), borderValue=(245, 245, 245))


def add_tag(image: np.ndarray, device_id: str, rng: np.random.Generator, degraded: bool = False) -> np.ndarray:
    out = image.copy()
    patch = make_tag_patch(device_id, rng, degraded=degraded)
    h, w = out.shape[:2]
    ph, pw = patch.shape[:2]
    corners = [
        (18, 18),
        (w - pw - 18, 18),
        (18, h - ph - 18),
        (w - pw - 18, h - ph - 18),
    ]
    x, y = corners[int(rng.integers(0, len(corners)))]
    x = max(0, min(w - pw, x))
    y = max(0, min(h - ph, y))
    out[y : y + ph, x : x + pw] = patch
    return out


def generate_variant(
    base: np.ndarray,
    device_id: str,
    rng: np.random.Generator,
    condition: str,
    tag_visible: bool,
    query: bool,
) -> np.ndarray:
    out = affine_variant(base, rng, stronger=query)
    if query or rng.random() < 0.35:
        out = perspective_variant(out, rng, max_shift=0.045 if query else 0.025)
    out = adjust_color(out, rng, condition if condition in {"low_light", "low_contrast"} else "clean")
    out = apply_condition(out, condition, rng)
    if tag_visible:
        out = add_tag(out, device_id, rng, degraded=query and condition in {"low_light", "glare", "blur", "noise"})
    return out


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def select_devices(crops: List[CandidateCrop], count: int, rng: np.random.Generator) -> List[Tuple[Dict, CandidateCrop]]:
    selected: List[Tuple[Dict, CandidateCrop]] = []
    used_sources = set()
    for plan in DEVICE_PLAN[:count]:
        pool = [crop for crop in crops if crop.class_id == plan["class_id"] and crop.source_image not in used_sources]
        if not pool:
            pool = [crop for crop in crops if crop.class_id == plan["class_id"]]
        if not pool:
            raise SystemExit(f"No candidate crops found for class_id={plan['class_id']} ({plan['type']}).")
        pool = sorted(pool, key=lambda item: item.area, reverse=True)
        top_pool = pool[: max(1, min(25, len(pool)))]
        crop = top_pool[int(rng.integers(0, len(top_pool)))]
        selected.append((plan, crop))
        used_sources.add(crop.source_image)
    return selected


def distribute_conditions(total: int) -> List[str]:
    if total <= 0:
        return []
    conditions: List[str] = []
    idx = 0
    while len(conditions) < total:
        conditions.append(CONDITIONS[idx % len(CONDITIONS)])
        idx += 1
    return conditions


def build_benchmark(args: argparse.Namespace) -> Dict:
    images_dir = repo_path(args.images)
    labels_dir = repo_path(args.labels)
    out_root = repo_path(args.out)
    dataset_root = images_dir.parent.parent if images_dir.parent.name == "test" else repo_path("data/detection/combined")
    class_names = read_class_names(dataset_root)
    rng = np.random.default_rng(args.seed)

    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels folder not found: {labels_dir}")

    ensure_output(out_root, args.overwrite)
    crops = collect_crops(images_dir, labels_dir, class_names, args.min_crop_size, args.crop_expand)
    selected = select_devices(crops, args.devices, rng)

    devices_manifest = []
    queries_manifest = []
    generated_refs: Dict[str, int] = {}
    generated_queries: Dict[str, int] = {}
    base_records = []

    for plan, crop in selected:
        device_id = plan["device_id"]
        device_type = plan["type"]
        devices_manifest.append(
            {
                "device_id": device_id,
                "type": device_type,
                "zone_id": args.zone_id,
                "description": f"Proxy {device_type} identity crop from {crop.source_image}",
                "has_visible_tag": str(bool(plan["has_visible_tag"])).lower(),
            }
        )
        base_dir = out_root / "base_crops" / device_id
        write_image(base_dir / "base.jpg", crop.crop)
        base_records.append(
            {
                "device_id": device_id,
                "type": device_type,
                "source_image": crop.source_image,
                "source_bbox_xyxy": list(crop.bbox_xyxy),
                "class_id": crop.class_id,
            }
        )

        ref_dir = out_root / "refs" / device_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_count = 0
        for idx in range(1, args.refs_per_device + 1):
            tag_visible = idx <= max(1, math.ceil(args.refs_per_device * 0.65))
            image = generate_variant(
                crop.crop,
                device_id,
                rng,
                condition="clean",
                tag_visible=tag_visible,
                query=False,
            )
            write_image(ref_dir / f"ref{idx:03d}.jpg", image)
            ref_count += 1
        generated_refs[device_id] = ref_count

        query_count = 0
        condition_sequence = distribute_conditions(args.queries_per_device)
        per_condition_idx = {condition: 0 for condition in CONDITIONS}
        for condition in condition_sequence:
            per_condition_idx[condition] += 1
            query_count += 1
            tag_visible = bool(plan["has_visible_tag"]) and condition != "occluded" and (query_count % 4 != 0)
            image = generate_variant(
                crop.crop,
                device_id,
                rng,
                condition=condition,
                tag_visible=tag_visible,
                query=True,
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
        generated_queries[device_id] = query_count

    manifests = {
        "devices_manifest": out_root / "devices_manifest.csv",
        "queries_manifest": out_root / "queries_manifest.csv",
        "base_crops_manifest": out_root / "proxy_base_crops.json",
    }
    write_csv(manifests["devices_manifest"], devices_manifest)
    write_csv(manifests["queries_manifest"], queries_manifest)
    manifests["base_crops_manifest"].write_text(json.dumps(base_records, indent=2), encoding="utf-8")

    return {
        "source_images_scanned": len(list(iter_images(images_dir))),
        "labeled_objects_found": len(crops),
        "selected_pseudo_devices": base_records,
        "generated_refs": generated_refs,
        "generated_queries": generated_queries,
        "manifests": {key: str(path.relative_to(REPO_ROOT)) for key, path in manifests.items()},
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a proxy device identity benchmark from YOLO detection crops.")
    parser.add_argument("--images", default="data/detection/combined/test/images")
    parser.add_argument("--labels", default="data/detection/combined/test/labels")
    parser.add_argument("--out", default="data/device_benchmark")
    parser.add_argument("--devices", type=int, default=3)
    parser.add_argument("--refs-per-device", type=int, default=8)
    parser.add_argument("--queries-per-device", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zone-id", default="<ZONE_ID>")
    parser.add_argument("--min-crop-size", type=int, default=12)
    parser.add_argument("--crop-expand", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.devices > len(DEVICE_PLAN):
        raise SystemExit(f"This proxy builder supports up to {len(DEVICE_PLAN)} configured devices.")

    summary = build_benchmark(args)
    print("Proxy device benchmark built.")
    print(f"  source images scanned: {summary['source_images_scanned']}")
    print(f"  labeled objects found: {summary['labeled_objects_found']}")
    print("  selected pseudo-devices:")
    for item in summary["selected_pseudo_devices"]:
        print(f"    - {item['device_id']} ({item['type']}): {item['source_image']}")
    print("  reference images generated:")
    for device_id, count in summary["generated_refs"].items():
        print(f"    - {device_id}: {count}")
    print("  query images generated:")
    for device_id, count in summary["generated_queries"].items():
        print(f"    - {device_id}: {count}")
    print("  manifests written:")
    for path in summary["manifests"].values():
        print(f"    - {path}")
    print("\nNext commands:")
    print("  cd D:\\python_works\\ValveLens\\backend")
    print("  python -m app.cli.enroll_devices_from_manifest --manifest ..\\data\\device_benchmark\\devices_manifest.csv --refs-root ..\\data\\device_benchmark\\refs --force-add-refs")
    print("  python -m app.cli.rebuild_device_index")
    print("  python -m app.cli.validate_identity_benchmark --queries-manifest ..\\data\\device_benchmark\\queries_manifest.csv --topk 5 --out ..\\artifacts\\identity_benchmark")


if __name__ == "__main__":
    main()
