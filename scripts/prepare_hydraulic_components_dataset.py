from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = REPO_ROOT / "data_sources" / "downloads" / "roboflow" / "hydraulic_components"
EXTRACTED_DIR = REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "hydraulic_components"
OUTPUT_ROOT = REPO_ROOT / "data" / "detection" / "expanded_industrial" / "hydraulic_components"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "detection_expanded" / "hydraulic_components"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}

EXPANDED_CLASS_NAMES = [
    "fitting",
    "filter",
    "gauge",
    "hydraulic_cylinder",
    "motor",
    "oil_cooler",
    "oil_level",
    "power_unit",
    "pressure_switch",
    "pump",
    "solenoid_coil",
    "strainer",
    "valve",
]
EXPANDED_CLASS_TO_ID = {name: idx for idx, name in enumerate(EXPANDED_CLASS_NAMES)}

EXPLICIT_CLASS_MAP = {
    "Motor": "motor",
    "Fitting": "fitting",
    "Hydraulic Cylinder": "hydraulic_cylinder",
    "Oil Cooler": "oil_cooler",
    "Oil Level": "oil_level",
    "Power Unit": "power_unit",
    "Pressure Gauge": "gauge",
    "Pressure Switch": "pressure_switch",
    "ValvePressure Switch": "pressure_switch",
    "Pump": "pump",
    "Pump Catridge": "pump",
    "Pump Cartridge": "pump",
    "Pump Motor": "pump",
    "Filter": "filter",
    "Filter Element": "filter",
    "Strainer": "strainer",
    "Solenoid Coil": "solenoid_coil",
    "Valve": "valve",
    "Valve Check Valve": "valve",
    "Valve Directional": "valve",
    "Valve Feed Control": "valve",
    "Valve Flow Control": "valve",
    "Valve Flow Control and Check": "valve",
    "Valve Logic": "valve",
    "Valve Needle": "valve",
    "Valve Pressure Reducing": "valve",
    "Valve Relief": "valve",
    "Valve Relief and Flow Control": "valve",
    "Valve Throttle": "valve",
}


@dataclass
class SplitSummary:
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    copied_images: int = 0
    rewritten_annotations: int = 0
    dropped_annotations: int = 0
    skipped_images_no_kept_labels: int = 0


@dataclass
class DatasetSummary:
    dataset_root: Optional[Path] = None
    data_yaml: Optional[Path] = None
    class_names: List[str] = field(default_factory=list)
    source_class_counts: Counter = field(default_factory=Counter)
    target_class_counts: Counter = field(default_factory=Counter)
    split_summaries: Dict[str, SplitSummary] = field(default_factory=lambda: defaultdict(SplitSummary))
    size_bytes: int = 0
    preview_paths: List[Path] = field(default_factory=list)


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def ensure_dirs() -> None:
    for path in (DOWNLOAD_DIR, EXTRACTED_DIR, OUTPUT_ROOT.parent, ARTIFACT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def load_yaml(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return parse_yaml_fallback(raw, path)


def parse_yaml_fallback(raw: str, path: Path) -> dict:
    data: Dict[str, object] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            i += 1
            continue
        key, remainder = line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        if key == "names":
            if remainder:
                data["names"] = ast.literal_eval(remainder)
            else:
                mapping: Dict[int, str] = {}
                items: List[str] = []
                i += 1
                while i < len(lines):
                    child = lines[i]
                    if child and not child.startswith((" ", "\t", "-")):
                        i -= 1
                        break
                    child_stripped = child.strip()
                    if child_stripped.startswith("- "):
                        items.append(child_stripped[2:].strip().strip("'\""))
                    elif ":" in child_stripped:
                        idx_s, value = child_stripped.split(":", 1)
                        mapping[int(idx_s.strip())] = value.strip().strip("'\"")
                    i += 1
                data["names"] = mapping if mapping else items
        else:
            data[key] = remainder.strip().strip("'\"")
        i += 1
    if "names" not in data:
        raise SystemExit(f"Malformed data.yaml, no names found: {path}")
    return data


def parse_names(yaml_data: dict, yaml_path: Path) -> List[str]:
    names = yaml_data.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        ordered = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda item: item[0])
        return [value for _, value in ordered]
    raise SystemExit(f"Unsupported names format in {yaml_path}")


def candidate_score(path: Path) -> int:
    score = 0
    if (path / "data.yaml").exists():
        score += 5
    for split_name in ("train", "valid", "val", "test"):
        split = path / split_name
        if (split / "images").exists():
            score += 2
        if (split / "labels").exists():
            score += 1
    return score


def detect_dataset_root(root: Path) -> Optional[Path]:
    candidates = set()
    if (root / "data.yaml").exists():
        candidates.add(root)
    if root.exists():
        for yaml_path in root.rglob("data.yaml"):
            candidates.add(yaml_path.parent)
    scored = []
    for candidate in candidates:
        score = candidate_score(candidate)
        if score >= 8:
            depth = len(candidate.relative_to(root).parts) if candidate != root else 0
            scored.append((score, depth, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], str(item[2]).lower()))
    return scored[0][2]


def iter_split_dirs(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for child in sorted(dataset_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        split = SPLIT_ALIASES.get(child.name.lower())
        if split:
            yield split, child


def iter_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: str(item).lower(),
    )


def label_for_image(split_dir: Path, image_path: Path) -> Path:
    images_dir = split_dir / "images"
    rel = image_path.relative_to(images_dir)
    return split_dir / "labels" / rel.with_suffix(".txt")


def target_for_source_class(name: str) -> Optional[str]:
    if name in EXPLICIT_CLASS_MAP:
        return EXPLICIT_CLASS_MAP[name]
    normalized = name.strip().lower()
    if "valve" in normalized:
        return "valve"
    if "pump" in normalized:
        return "pump"
    if "gauge" in normalized:
        return "gauge"
    return None


def source_to_target_mapping(class_names: List[str]) -> Dict[int, Optional[int]]:
    mapping: Dict[int, Optional[int]] = {}
    for idx, name in enumerate(class_names):
        target = target_for_source_class(name)
        mapping[idx] = EXPANDED_CLASS_TO_ID[target] if target else None
    return mapping


def parse_yolo_line(line: str) -> Optional[Tuple[int, List[float]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return class_id, values


def normalize_box(values: List[float]) -> Optional[List[float]]:
    if len(values) == 4:
        x_center, y_center, width, height = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
    else:
        return None
    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(1.0, x_center + width / 2.0)
    y2 = min(1.0, y_center + height / 2.0)
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height]


def rewrite_labels(
    label_path: Path,
    mapping: Dict[int, Optional[int]],
    class_names: List[str],
    summary: DatasetSummary,
) -> List[str]:
    if not label_path.exists():
        return []
    rewritten: List[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_yolo_line(raw)
        if parsed is None:
            continue
        source_id, values = parsed
        if 0 <= source_id < len(class_names):
            summary.source_class_counts[class_names[source_id]] += 1
        target_id = mapping.get(source_id)
        if target_id is None:
            continue
        box = normalize_box(values)
        if box is None:
            continue
        rewritten.append(f"{target_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
        summary.target_class_counts[EXPANDED_CLASS_NAMES[target_id]] += 1
    return rewritten


def inspect_dataset(dataset_root: Path, class_names: List[str], mapping: Dict[int, Optional[int]]) -> DatasetSummary:
    summary = DatasetSummary(
        dataset_root=dataset_root,
        data_yaml=dataset_root / "data.yaml",
        class_names=class_names,
        size_bytes=folder_size(dataset_root),
    )
    for split, split_dir in iter_split_dirs(dataset_root):
        images = iter_images(split_dir / "images")
        labels_dir = split_dir / "labels"
        label_files = sorted(labels_dir.rglob("*.txt"), key=lambda item: str(item).lower()) if labels_dir.exists() else []
        split_summary = summary.split_summaries[split]
        split_summary.images = len(images)
        split_summary.labels = len(label_files)
        for image_path in images:
            label_path = label_for_image(split_dir, image_path)
            if not label_path.exists():
                split_summary.missing_labels += 1
                continue
            _ = rewrite_labels(label_path, mapping, class_names, summary)
    return summary


def prepare_dataset(
    dataset_root: Path,
    out_root: Path,
    class_names: List[str],
    mapping: Dict[int, Optional[int]],
    overwrite: bool,
    keep_empty: bool,
) -> DatasetSummary:
    if out_root.exists() and any(out_root.rglob("*")):
        if not overwrite:
            raise SystemExit(
                f"Output folder already contains files: {out_root}. "
                "Use --overwrite to rebuild the prepared dataset."
            )
        shutil.rmtree(out_root)
    for split in ("train", "valid", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    summary = DatasetSummary(dataset_root=dataset_root, data_yaml=dataset_root / "data.yaml", class_names=class_names)
    for split, split_dir in iter_split_dirs(dataset_root):
        split_summary = summary.split_summaries[split]
        images_dir = split_dir / "images"
        for image_path in iter_images(images_dir):
            split_summary.images += 1
            label_path = label_for_image(split_dir, image_path)
            if not label_path.exists():
                split_summary.missing_labels += 1
                if not keep_empty:
                    continue
                rewritten: List[str] = []
            else:
                split_summary.labels += 1
                rewritten = rewrite_labels(label_path, mapping, class_names, summary)
            if not rewritten and not keep_empty:
                split_summary.skipped_images_no_kept_labels += 1
                continue

            rel_name = image_path.relative_to(images_dir).as_posix().replace("/", "__")
            dest_image = out_root / split / "images" / rel_name
            dest_label = out_root / split / "labels" / f"{Path(rel_name).stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
            split_summary.copied_images += 1
            split_summary.rewritten_annotations += len(rewritten)

            source_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            split_summary.dropped_annotations += max(0, len([line for line in source_lines if line.strip()]) - len(rewritten))

    write_data_yaml(out_root)
    summary.size_bytes = folder_size(out_root)
    return summary


def write_data_yaml(out_root: Path) -> None:
    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(EXPANDED_CLASS_NAMES)}",
        "names:",
    ]
    for idx, name in enumerate(EXPANDED_CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_boxes(label_path: Path, names: List[str]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    boxes = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_yolo_line(raw)
        if parsed is None:
            continue
        class_id, values = parsed
        box = normalize_box(values)
        if box is None:
            continue
        label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        boxes.append((label, tuple(box)))
    return boxes


def draw_boxes(image_path: Path, label_path: Path, names: List[str], size: Tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail(size)
        canvas = Image.new("RGB", size, (245, 245, 245))
        offset_x = (size[0] - image.width) // 2
        offset_y = (size[1] - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        for label, (xc, yc, bw, bh) in load_boxes(label_path, names):
            x1 = offset_x + int((xc - bw / 2.0) * image.width)
            y1 = offset_y + int((yc - bh / 2.0) * image.height)
            x2 = offset_x + int((xc + bw / 2.0) * image.width)
            y2 = offset_y + int((yc + bh / 2.0) * image.height)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 88, 32), width=3)
            draw.rectangle((x1, max(0, y1 - 16), min(size[0], x1 + 7 * len(label) + 8), y1), fill=(255, 88, 32))
            draw.text((x1 + 3, max(0, y1 - 15)), label[:24], fill=(0, 0, 0))
        return canvas


def make_preview_sheet(
    dataset_root: Path,
    output_path: Path,
    names: List[str],
    sample_count: int,
) -> Optional[Path]:
    samples: List[Tuple[Path, Path]] = []
    for split, split_dir in iter_split_dirs(dataset_root):
        images_dir = split_dir / "images"
        for image_path in iter_images(images_dir):
            label_path = label_for_image(split_dir, image_path)
            if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                samples.append((image_path, label_path))
            if len(samples) >= sample_count:
                break
        if len(samples) >= sample_count:
            break
    if not samples:
        return None

    tile_size = (360, 260)
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), (255, 255, 255))
    for idx, (image_path, label_path) in enumerate(samples):
        tile = draw_boxes(image_path, label_path, names, tile_size)
        x = (idx % cols) * tile_size[0]
        y = (idx // cols) * tile_size[1]
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def print_mapping(class_names: List[str], mapping: Dict[int, Optional[int]]) -> None:
    print("\nClass mapping proposal:")
    for idx, name in enumerate(class_names):
        target_id = mapping.get(idx)
        target = EXPANDED_CLASS_NAMES[target_id] if target_id is not None else "IGNORE"
        print(f"  {idx}: {name} -> {target}")


def print_summary(title: str, summary: DatasetSummary) -> None:
    print(f"\n{title}")
    if summary.dataset_root:
        print(f"  dataset_root: {summary.dataset_root}")
    if summary.data_yaml:
        print(f"  data_yaml: {summary.data_yaml}")
    print(f"  size: {human_size(summary.size_bytes)}")
    for split in ("train", "valid", "test"):
        item = summary.split_summaries.get(split, SplitSummary())
        print(
            f"  {split}: images={item.images}, labels={item.labels}, "
            f"missing_labels={item.missing_labels}, copied={item.copied_images}"
        )
    if summary.target_class_counts:
        print("  target class counts:")
        for name in EXPANDED_CLASS_NAMES:
            print(f"    - {name}: {summary.target_class_counts.get(name, 0)}")


def print_manual_download_instructions() -> None:
    print("\nHydraulic Components Detection dataset is not staged locally.")
    print("Roboflow CLI/Python SDK is not configured, so automatic download was skipped.")
    print("Manual download:")
    print("  1. Open https://universe.roboflow.com/nattapat-kieuvongngam-rup1a/hydraulic-components-detection")
    print("  2. Click 'Use this Dataset'.")
    print("  3. Choose YOLOv8 format and download the zip.")
    print(f"  4. Save the zip under {DOWNLOAD_DIR}")
    print(f"  5. Extract it under {EXTRACTED_DIR}")
    print("  6. Rerun this script.")
    print("\nOptional SDK path if configured:")
    print("  pip install roboflow")
    print("  $env:ROBOFLOW_API_KEY='<your key>'")
    print("  python .\\scripts\\prepare_hydraulic_components_dataset.py --download --overwrite")


def try_download_with_sdk(version: int) -> bool:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set; skipping SDK download.")
        return False
    try:
        from roboflow import Roboflow  # type: ignore
    except Exception:
        print("Python roboflow package is not installed; skipping SDK download.")
        return False
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("nattapat-kieuvongngam-rup1a").project("hydraulic-components-detection")
        project.version(version).download("yolov8", location=str(EXTRACTED_DIR))
        return True
    except Exception as exc:
        print(f"Roboflow SDK download failed: {exc}")
        return False


def try_download_with_cli() -> bool:
    cli = shutil.which("roboflow")
    if not cli:
        return False
    try:
        result = subprocess.run(
            [
                cli,
                "download",
                "hydraulic-components-detection",
                "--format",
                "yolov8",
                "--location",
                str(EXTRACTED_DIR),
            ],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True
        print(result.stdout)
        print(result.stderr)
        return False
    except Exception as exc:
        print(f"Roboflow CLI download failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and remap the Roboflow Hydraulic Components Detection dataset."
    )
    parser.add_argument("--source", default=str(EXTRACTED_DIR))
    parser.add_argument("--out", default=str(OUTPUT_ROOT))
    parser.add_argument("--download", action="store_true", help="Try Roboflow CLI/SDK download if configured.")
    parser.add_argument("--version", type=int, default=2, help="Roboflow dataset version to request through SDK.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with no selected target labels.")
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    ensure_dirs()
    source_root = repo_path(args.source)
    out_root = repo_path(args.out)

    if args.download:
        downloaded = try_download_with_cli() or try_download_with_sdk(args.version)
        if downloaded:
            print(f"Dataset downloaded/staged under: {EXTRACTED_DIR}")
        else:
            print("Automatic download did not complete.")

    dataset_root = detect_dataset_root(source_root)
    if dataset_root is None:
        print_manual_download_instructions()
        return

    yaml_path = dataset_root / "data.yaml"
    yaml_data = load_yaml(yaml_path)
    class_names = parse_names(yaml_data, yaml_path)
    mapping = source_to_target_mapping(class_names)

    inspect_summary = inspect_dataset(dataset_root, class_names, mapping)
    source_preview = make_preview_sheet(
        dataset_root,
        ARTIFACT_ROOT / "hydraulic_components_source_preview.jpg",
        class_names,
        max(1, args.preview_count),
    )
    if source_preview:
        inspect_summary.preview_paths.append(source_preview)

    print_summary("Source dataset inspection:", inspect_summary)
    print("\nSource classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print_mapping(class_names, mapping)

    if args.inspect_only:
        if inspect_summary.preview_paths:
            print("\nPreview files:")
            for path in inspect_summary.preview_paths:
                print(f"  - {path}")
        return

    prepared_summary = prepare_dataset(
        dataset_root,
        out_root,
        class_names,
        mapping,
        overwrite=args.overwrite,
        keep_empty=args.keep_empty,
    )
    prepared_preview = make_preview_sheet(
        out_root,
        ARTIFACT_ROOT / "hydraulic_components_prepared_preview.jpg",
        EXPANDED_CLASS_NAMES,
        max(1, args.preview_count),
    )
    if prepared_preview:
        prepared_summary.preview_paths.append(prepared_preview)

    print_summary("Prepared expanded industrial dataset:", prepared_summary)
    print(f"\nPrepared data.yaml: {out_root / 'data.yaml'}")
    if inspect_summary.preview_paths or prepared_summary.preview_paths:
        print("Preview files:")
        for path in [*inspect_summary.preview_paths, *prepared_summary.preview_paths]:
            print(f"  - {path}")
    print("\nRecommendation:")
    print("  Keep mapped classes with enough labels: valve, pump, motor, fitting, gauge, pressure_switch,")
    print("  filter, strainer, solenoid_coil, oil_level, oil_cooler, hydraulic_cylinder, power_unit.")
    print("  Inspect preview sheets before merging into any larger detector dataset.")
    print("  Do not overwrite models/detector.pt or retrain until the class balance is reviewed.")


if __name__ == "__main__":
    main()
