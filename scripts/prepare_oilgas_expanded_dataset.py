from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}

TARGET_CLASSES = [
    "valve",
    "gauge",
    "wellhead",
    "flange",
    "instrument",
    "vessel",
    "pipe",
    "desalter",
    "heater",
    "heat_exchanger",
    "structure",
    "tank_or_cylinder",
    "control_panel",
    "relay",
    "pressure_vessel",
    "cooling_tower",
    "tee_connector",
    "turbine_generator",
    "warning_sign",
    "clamp",
    "equipment",
    "ladder",
    "person",
    "support",
]
TARGET_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    title: str
    source_url: str
    roboflow_workspace: Optional[str]
    roboflow_project: Optional[str]
    roboflow_version: Optional[int]
    roboflow_format: str
    download_dir: Path
    extracted_dir: Path
    output_dir: Path
    artifact_dir: Path
    expected_images: str
    license_name: str
    role: str
    limitation: str
    media_type: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "oil_refinery": DatasetConfig(
        key="oil_refinery",
        title="Oil Refinery",
        source_url="https://universe.roboflow.com/apparatusbeats-olbtp/oil-refinery",
        roboflow_workspace="apparatusbeats-olbtp",
        roboflow_project="oil-refinery",
        roboflow_version=3,
        roboflow_format="yolov5pytorch",
        download_dir=REPO_ROOT / "data_sources" / "downloads" / "roboflow" / "oil_refinery",
        extracted_dir=REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "oil_refinery",
        output_dir=REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "oil_refinery",
        artifact_dir=REPO_ROOT / "artifacts" / "detection_oilgas_expanded" / "oil_refinery",
        expected_images="350",
        license_name="CC BY 4.0",
        role="refinery-specific macro-equipment detection",
        limitation="341 raw classes must be collapsed to macro classes before any training.",
        media_type="unknown until inspection; Roboflow examples appear refinery/equipment oriented",
    ),
    "elementos_offshore": DatasetConfig(
        key="elementos_offshore",
        title="Elementos Offshore",
        source_url="https://universe.roboflow.com/dataset-offshore/elementos-offshore",
        roboflow_workspace="dataset-offshore",
        roboflow_project="elementos-offshore",
        roboflow_version=4,
        roboflow_format="yolov5pytorch",
        download_dir=REPO_ROOT / "data_sources" / "downloads" / "roboflow" / "elementos_offshore",
        extracted_dir=REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "elementos_offshore",
        output_dir=REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "elementos_offshore",
        artifact_dir=REPO_ROOT / "artifacts" / "detection_oilgas_expanded" / "elementos_offshore",
        expected_images="106",
        license_name="Public Domain",
        role="small offshore/industrial facility support dataset",
        limitation="Small dataset; useful for visual relevance but not enough alone.",
        media_type="real-image likely, verify after inspection",
    ),
    "wellhead_valve_gauge": DatasetConfig(
        key="wellhead_valve_gauge",
        title="Object_detection_dataset by Anto",
        source_url="Roboflow search result: Object_detection_dataset by Anto",
        roboflow_workspace=None,
        roboflow_project=None,
        roboflow_version=None,
        roboflow_format="yolov8",
        download_dir=REPO_ROOT / "data_sources" / "downloads" / "roboflow" / "wellhead_valve_gauge",
        extracted_dir=REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "wellhead_valve_gauge",
        output_dir=REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "wellhead_valve_gauge",
        artifact_dir=REPO_ROOT / "artifacts" / "detection_oilgas_expanded" / "wellhead_valve_gauge",
        expected_images="about 1.71k in gauge search result; about 1.25k in valve search result",
        license_name="unknown until exact Roboflow project page is identified",
        role="wellhead, valve, gauge, relay detection support",
        limitation="Exact project URL/slug must be confirmed before automatic download.",
        media_type="unknown until inspection",
    ),
    "industrial_multilabel": DatasetConfig(
        key="industrial_multilabel",
        title="industrial-multilabel",
        source_url="https://universe.roboflow.com/yolo-rovw9/industrial-multilabel",
        roboflow_workspace="yolo-rovw9",
        roboflow_project="industrial-multilabel",
        roboflow_version=0,
        roboflow_format="yolov8",
        download_dir=REPO_ROOT / "data_sources" / "downloads" / "roboflow" / "industrial_multilabel",
        extracted_dir=REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "industrial_multilabel",
        output_dir=REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "industrial_multilabel",
        artifact_dir=REPO_ROOT / "artifacts" / "detection_oilgas_expanded" / "industrial_multilabel",
        expected_images="970",
        license_name="MIT",
        role="optional broader industrial/factory support dataset",
        limitation="Contains many unrelated classes; only selected industrial classes should be kept.",
        media_type="unknown until inspection",
    ),
}


@dataclass
class SplitReport:
    source_images: int = 0
    source_labels: int = 0
    missing_labels: int = 0
    malformed_lines: int = 0
    invalid_boxes: int = 0
    copied_images: int = 0
    skipped_images: int = 0
    rewritten_annotations: int = 0
    dropped_annotations: int = 0


@dataclass
class PrepReport:
    dataset: str
    title: str
    downloaded: bool = False
    download_attempted: bool = False
    data_yaml_found: bool = False
    dataset_root: Optional[str] = None
    source_size_bytes: int = 0
    prepared_size_bytes: int = 0
    source_classes: List[str] = field(default_factory=list)
    mapped_classes: List[str] = field(default_factory=list)
    ignored_classes: List[str] = field(default_factory=list)
    split_reports: Dict[str, SplitReport] = field(default_factory=lambda: defaultdict(SplitReport))
    source_class_counts: Counter = field(default_factory=Counter)
    target_class_counts: Counter = field(default_factory=Counter)
    preview_paths: List[str] = field(default_factory=list)
    recommendation: str = "missing"


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def ensure_dirs(config: DatasetConfig) -> None:
    for path in (config.download_dir, config.extracted_dir, config.output_dir, config.artifact_dir):
        path.mkdir(parents=True, exist_ok=True)


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


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
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            yield SPLIT_ALIASES[child.name.lower()], child


def iter_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: str(item).lower(),
    )


def label_for_image(split_dir: Path, image_path: Path) -> Path:
    images_dir = split_dir / "images"
    return split_dir / "labels" / image_path.relative_to(images_dir).with_suffix(".txt")


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def map_class(dataset_key: str, source_name: str) -> Optional[str]:
    raw = source_name.strip()
    norm = normalize_name(raw)

    if dataset_key == "oil_refinery":
        compact = norm.replace(" ", "")
        if compact.startswith("desalter"):
            return "desalter"
        if "pipe" in compact:
            return "pipe"
        if "tank" in compact or "cylinder" in compact:
            return "tank_or_cylinder"
        if compact.startswith("heater"):
            return "heater"
        if compact.startswith("heatexchanger"):
            return "heat_exchanger"
        structure_terms = [
            "base",
            "beam",
            "chasis",
            "chassis",
            "frame",
            "ground",
            "ladder",
            "leg",
            "platform",
            "rail",
            "structure",
            "support",
            "walkway",
        ]
        if any(term in compact for term in structure_terms):
            return "structure" if "structure" in TARGET_TO_ID else None
        return None

    if dataset_key == "elementos_offshore":
        return {
            "equipamento": "equipment",
            "escada": "ladder",
            "flange": "flange",
            "instrumento": "instrument",
            "operador": "person",
            "suporte": "support",
            "valvula": "valve",
            "vaso": "vessel",
        }.get(norm)

    if dataset_key == "wellhead_valve_gauge":
        return {
            "wellhead": "wellhead",
            "well head": "wellhead",
            "gauge": "gauge",
            "relay": "relay",
            "valve": "valve",
        }.get(norm)

    if dataset_key == "industrial_multilabel":
        return {
            "control panel": "control_panel",
            "cooling towers": "cooling_tower",
            "pipes": "pipe",
            "pipe": "pipe",
            "pressure gauges": "gauge",
            "pressure gauge": "gauge",
            "pressure vessel": "pressure_vessel",
            "tee connector": "tee_connector",
            "turbine generator": "turbine_generator",
            "warning signs": "warning_sign",
            "warning sign": "warning_sign",
            "clamps": "clamp",
            "clamp": "clamp",
        }.get(norm)

    return None


def source_to_target_mapping(dataset_key: str, class_names: List[str]) -> Dict[int, Optional[int]]:
    mapping: Dict[int, Optional[int]] = {}
    for idx, name in enumerate(class_names):
        target = map_class(dataset_key, name)
        mapping[idx] = TARGET_TO_ID[target] if target in TARGET_TO_ID else None
    return mapping


def parse_yolo_line(line: str) -> Optional[Tuple[int, List[float]]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        return int(float(parts[0])), [float(value) for value in parts[1:]]
    except ValueError:
        return None


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


def rewrite_label_lines(
    dataset_key: str,
    label_path: Path,
    class_names: List[str],
    mapping: Dict[int, Optional[int]],
    report: PrepReport,
    split_report: SplitReport,
    max_images_per_class: Optional[int],
    image_class_tracker: Counter,
) -> List[str]:
    rewritten: List[str] = []
    kept_classes_in_image = set()
    if not label_path.exists():
        return rewritten

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parsed = parse_yolo_line(raw)
        if parsed is None:
            split_report.malformed_lines += 1
            continue
        source_id, values = parsed
        if 0 <= source_id < len(class_names):
            report.source_class_counts[class_names[source_id]] += 1
        target_id = mapping.get(source_id)
        if target_id is None:
            split_report.dropped_annotations += 1
            continue
        target_name = TARGET_CLASSES[target_id]
        if (
            max_images_per_class is not None
            and target_name not in kept_classes_in_image
            and image_class_tracker[target_name] >= max_images_per_class
        ):
            split_report.dropped_annotations += 1
            continue
        box = normalize_box(values)
        if box is None or any(value < 0.0 or value > 1.0 for value in box):
            split_report.invalid_boxes += 1
            continue
        rewritten.append(f"{target_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
        report.target_class_counts[target_name] += 1
        kept_classes_in_image.add(target_name)

    for target_name in kept_classes_in_image:
        image_class_tracker[target_name] += 1
    return rewritten


def clear_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.rglob("*")):
        if not overwrite:
            raise SystemExit(f"Output exists and is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    for split in ("train", "valid", "test"):
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_root: Path, active_class_ids: List[int]) -> None:
    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(TARGET_CLASSES)}",
        "names:",
    ]
    for idx, name in enumerate(TARGET_CLASSES):
        lines.append(f"  {idx}: {name}")
    lines.append("")
    lines.append("# Active classes in this prepared subset:")
    for idx in active_class_ids:
        lines.append(f"#   {idx}: {TARGET_CLASSES[idx]}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(
    config: DatasetConfig,
    source_root: Path,
    out_root: Path,
    dataset_root: Path,
    class_names: List[str],
    mapping: Dict[int, Optional[int]],
    overwrite: bool,
    max_images_per_class: Optional[int],
) -> PrepReport:
    report = PrepReport(dataset=config.key, title=config.title)
    report.data_yaml_found = True
    report.dataset_root = str(dataset_root)
    report.source_classes = class_names
    report.source_size_bytes = folder_size(dataset_root)
    report.mapped_classes = sorted(
        {TARGET_CLASSES[target_id] for target_id in mapping.values() if target_id is not None}
    )
    report.ignored_classes = [
        name for idx, name in enumerate(class_names) if mapping.get(idx) is None
    ]

    clear_output(out_root, overwrite=overwrite)
    image_class_tracker: Counter = Counter()

    for split, split_dir in iter_split_dirs(dataset_root):
        split_report = report.split_reports[split]
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images = iter_images(images_dir)
        label_files = sorted(labels_dir.rglob("*.txt"), key=lambda item: str(item).lower()) if labels_dir.exists() else []
        split_report.source_images = len(images)
        split_report.source_labels = len(label_files)

        for image_path in images:
            label_path = label_for_image(split_dir, image_path)
            if not label_path.exists():
                split_report.missing_labels += 1
                split_report.skipped_images += 1
                continue
            rewritten = rewrite_label_lines(
                config.key,
                label_path,
                class_names,
                mapping,
                report,
                split_report,
                max_images_per_class,
                image_class_tracker,
            )
            if not rewritten:
                split_report.skipped_images += 1
                continue
            rel_name = image_path.relative_to(images_dir).as_posix().replace("/", "__")
            dest_image = out_root / split / "images" / rel_name
            dest_label = out_root / split / "labels" / f"{Path(rel_name).stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
            split_report.copied_images += 1
            split_report.rewritten_annotations += len(rewritten)

    active_class_ids = sorted({TARGET_TO_ID[name] for name in report.target_class_counts})
    write_data_yaml(out_root, active_class_ids)
    report.prepared_size_bytes = folder_size(out_root)
    return report


def load_boxes(label_path: Path, names: List[str]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    boxes = []
    if not label_path.exists():
        return boxes
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


def draw_preview_tile(image_path: Path, label_path: Path, names: List[str], tile_size: Tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail(tile_size)
        canvas = Image.new("RGB", tile_size, (245, 245, 245))
        offset_x = (tile_size[0] - image.width) // 2
        offset_y = (tile_size[1] - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        for label, (xc, yc, bw, bh) in load_boxes(label_path, names):
            x1 = offset_x + int((xc - bw / 2.0) * image.width)
            y1 = offset_y + int((yc - bh / 2.0) * image.height)
            x2 = offset_x + int((xc + bw / 2.0) * image.width)
            y2 = offset_y + int((yc + bh / 2.0) * image.height)
            draw.rectangle((x1, y1, x2, y2), outline=(14, 165, 233), width=3)
            label_short = label[:28]
            draw.rectangle((x1, max(0, y1 - 16), min(tile_size[0], x1 + 7 * len(label_short) + 8), y1), fill=(14, 165, 233))
            draw.text((x1 + 3, max(0, y1 - 15)), label_short, fill=(0, 0, 0))
        return canvas


def make_preview(dataset_root: Path, output_path: Path, names: List[str], sample_count: int) -> Optional[str]:
    samples: List[Tuple[Path, Path]] = []
    for _, split_dir in iter_split_dirs(dataset_root):
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
        tile = draw_preview_tile(image_path, label_path, names, tile_size)
        sheet.paste(tile, ((idx % cols) * tile_size[0], (idx // cols) * tile_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def try_roboflow_download(config: DatasetConfig) -> bool:
    if not config.roboflow_workspace or not config.roboflow_project or config.roboflow_version is None:
        print(f"{config.key}: no exact Roboflow workspace/project/version configured; use manual staging.")
        return False
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(f"{config.key}: ROBOFLOW_API_KEY is not set in this process; automatic download skipped.")
        return False
    try:
        from roboflow import Roboflow  # type: ignore
    except Exception:
        print(f"{config.key}: Python roboflow package is not installed; automatic download skipped.")
        return False
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(config.roboflow_workspace).project(config.roboflow_project)
        project.version(config.roboflow_version).download(
            config.roboflow_format,
            location=str(config.extracted_dir),
            overwrite=True,
        )
        return True
    except Exception as exc:
        print(f"{config.key}: Roboflow download failed: {exc}")
        return False


def print_manual(config: DatasetConfig) -> None:
    print(f"\n{config.title} is not staged locally.")
    print(f"Source: {config.source_url}")
    print("Manual staging:")
    print("  1. Download/export the dataset in YOLOv8 format from Roboflow.")
    print(f"  2. Store the archive under {config.download_dir}")
    print(f"  3. Extract it under {config.extracted_dir}")
    print("  4. Rerun this script with --overwrite.")


def report_to_dict(report: PrepReport) -> Dict:
    return {
        "dataset": report.dataset,
        "title": report.title,
        "download_attempted": report.download_attempted,
        "downloaded": report.downloaded,
        "data_yaml_found": report.data_yaml_found,
        "dataset_root": report.dataset_root,
        "source_size": human_size(report.source_size_bytes),
        "prepared_size": human_size(report.prepared_size_bytes),
        "source_classes": report.source_classes,
        "mapped_classes": report.mapped_classes,
        "ignored_classes": report.ignored_classes,
        "splits": {
            split: vars(item)
            for split, item in sorted(report.split_reports.items(), key=lambda pair: pair[0])
        },
        "target_class_counts": dict(report.target_class_counts),
        "preview_paths": report.preview_paths,
        "recommendation": report.recommendation,
    }


def print_report(report: PrepReport) -> None:
    print("\nOil/gas expanded dataset report:")
    print(json.dumps(report_to_dict(report), indent=2))


def write_report(config: DatasetConfig, report: PrepReport) -> str:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = config.artifact_dir / "prep_report.json"
    path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    return str(path)


def recommendation_for(report: PrepReport) -> str:
    if not report.data_yaml_found:
        return "missing: stage/download first"
    total_annotations = sum(report.target_class_counts.values())
    copied = sum(item.copied_images for item in report.split_reports.values())
    if copied == 0 or total_annotations == 0:
        return "drop or remap: no selected labels survived remapping"
    if copied < 30:
        return "inspect more: very small after remapping"
    if len(report.target_class_counts) <= 2:
        return "inspect more: narrow class coverage"
    return "inspect previews before training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare focused oil/gas expanded detection datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--no-download", action="store_true", help="Do not try Roboflow SDK download when source is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    ensure_dirs(config)

    source_root = repo_path(args.source_root) if args.source_root else config.extracted_dir
    out_root = repo_path(args.out_root) if args.out_root else config.output_dir
    report = PrepReport(dataset=config.key, title=config.title)
    download_attempted = False
    downloaded = False

    dataset_root = detect_dataset_root(source_root)
    if dataset_root is None and not args.no_download:
        download_attempted = True
        downloaded = try_roboflow_download(config)
        report.download_attempted = download_attempted
        report.downloaded = downloaded
        dataset_root = detect_dataset_root(source_root)

    if dataset_root is None:
        print_manual(config)
        report.recommendation = recommendation_for(report)
        report.preview_paths = []
        report_path = write_report(config, report)
        print_report(report)
        print(f"Report JSON: {report_path}")
        return

    yaml_path = dataset_root / "data.yaml"
    yaml_data = load_yaml(yaml_path)
    class_names = parse_names(yaml_data, yaml_path)
    mapping = source_to_target_mapping(config.key, class_names)

    report = prepare_dataset(
        config=config,
        source_root=source_root,
        out_root=out_root,
        dataset_root=dataset_root,
        class_names=class_names,
        mapping=mapping,
        overwrite=args.overwrite,
        max_images_per_class=args.max_images_per_class,
    )
    report.download_attempted = download_attempted
    report.downloaded = downloaded

    source_preview = make_preview(
        dataset_root,
        config.artifact_dir / "source_preview.jpg",
        class_names,
        max(1, args.preview_count),
    )
    prepared_preview = make_preview(
        out_root,
        config.artifact_dir / "prepared_preview.jpg",
        TARGET_CLASSES,
        max(1, args.preview_count),
    )
    report.preview_paths = [path for path in (source_preview, prepared_preview) if path]
    report.recommendation = recommendation_for(report)

    report_path = write_report(config, report)
    print_report(report)
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
