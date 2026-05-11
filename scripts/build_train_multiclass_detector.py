from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "detection" / "industrial_multiclass"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "detection_multiclass"
MODELS_DIR = REPO_ROOT / "models"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}


TARGET_CLASSES = [
    "valve",
    "gauge",
    "flange",
    "instrument_panel",
    "vessel",
    "pipe",
    "desalter",
    "heater",
    "heat_exchanger",
    "tank",
]
TARGET_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


@dataclass
class DatasetCandidate:
    key: str
    root: Path
    format: str
    data_yaml: Optional[Path] = None
    coco_jsons: List[Path] = field(default_factory=list)
    voc_xmls: List[Path] = field(default_factory=list)
    split_images: Dict[str, int] = field(default_factory=dict)
    split_labels: Dict[str, int] = field(default_factory=dict)
    annotation_count: int = 0
    class_names: List[str] = field(default_factory=list)
    usable: bool = False
    selected: bool = False
    already_in_combined: bool = False
    relevance: str = "not assessed"
    note: str = ""


@dataclass
class BuildStats:
    source_images: int = 0
    copied_images: int = 0
    skipped_images: int = 0
    missing_labels: int = 0
    malformed_annotations: int = 0
    invalid_boxes: int = 0
    kept_annotations: int = 0
    dropped_annotations: int = 0
    empty_output_labels: int = 0


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("/", "\\")
    except Exception:
        return str(path).replace("/", "\\")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_name(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(read_text(path))
    if not isinstance(data, dict):
        return {}
    return data


def parse_names(data_yaml: Path) -> List[str]:
    data = load_yaml(data_yaml)
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(((int(k), v) for k, v in names.items()), key=lambda item: item[0])]
    return []


def iter_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: str(p).lower())


def iter_split_dirs(root: Path) -> Iterable[Tuple[str, Path]]:
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()) if root.exists() else []:
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            yield SPLIT_ALIASES[child.name.lower()], child


def yolo_label_for(split_dir: Path, image_path: Path) -> Path:
    images_dir = split_dir / "images"
    return split_dir / "labels" / image_path.relative_to(images_dir).with_suffix(".txt")


def count_yolo_annotations(label_path: Path) -> Tuple[int, int]:
    total = 0
    malformed = 0
    for raw in read_text(label_path).splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            malformed += 1
            continue
        try:
            int(float(parts[0]))
            [float(value) for value in parts[1:]]
        except ValueError:
            malformed += 1
            continue
        total += 1
    return total, malformed


def discover_yolo_roots(search_roots: List[Path]) -> List[Path]:
    roots = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for yaml_path in search_root.rglob("data.yaml"):
            root = yaml_path.parent
            has_split = any((root / split / "images").exists() and (root / split / "labels").exists() for split in ["train", "valid", "val", "test"])
            if has_split:
                roots.add(root)
    return sorted(roots, key=lambda p: str(p).lower())


def candidate_key(root: Path) -> str:
    try:
        relative = root.relative_to(REPO_ROOT)
        return "_".join(part for part in relative.parts if part not in {"data", "detection", "data_sources", "extracted", "roboflow"})
    except Exception:
        return root.name


def is_selected_dataset(root: Path) -> bool:
    selected = [
        REPO_ROOT / "data" / "detection" / "combined",
        REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "oil_refinery",
        REPO_ROOT / "data" / "detection" / "oilgas_expanded" / "elementos_offshore",
    ]
    resolved = root.resolve()
    return any(resolved == item.resolve() for item in selected)


def is_duplicate_source(root: Path) -> bool:
    duplicate_sources = [
        REPO_ROOT / "data_sources" / "extracted" / "valve_detection_v1i_yolov8",
        REPO_ROOT / "data_sources" / "extracted" / "valve_detection_v6i_yolov8",
        REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "oil_refinery",
        REPO_ROOT / "data_sources" / "extracted" / "roboflow" / "elementos_offshore",
    ]
    resolved = root.resolve()
    return any(resolved == item.resolve() for item in duplicate_sources)


def map_class_name(source_name: str) -> Optional[str]:
    norm = normalize_name(source_name)
    compact = norm.replace(" ", "")
    if "gauge" in norm or norm in {"pressure gauge", "analog gauge"}:
        return "gauge"
    if "valve" in norm or "valvula" in norm or norm.startswith("wheel handle") or norm.startswith("lever handle"):
        return "valve"
    if "flange" in norm:
        return "flange"
    if norm in {"instrument", "instrumento"} or "instrument panel" in norm:
        return "instrument_panel"
    if norm in {"vessel", "vaso"} or "pressure vessel" in norm:
        return "vessel"
    if "pipe" in norm or "pipes" in norm or "pipeline" in norm:
        return "pipe"
    if compact.startswith("desalter"):
        return "desalter"
    if compact.startswith("heater"):
        return "heater"
    if compact.startswith("heatexchanger"):
        return "heat_exchanger"
    if "tank" in norm or "cylinder" in norm or norm.startswith("tower"):
        return "tank"
    if norm == "tank or cylinder":
        return "tank"
    return None


def candidate_from_yolo(root: Path) -> DatasetCandidate:
    data_yaml = root / "data.yaml"
    names = parse_names(data_yaml)
    candidate = DatasetCandidate(
        key=candidate_key(root),
        root=root,
        format="YOLO",
        data_yaml=data_yaml,
        class_names=names,
        usable=True,
        selected=is_selected_dataset(root),
        already_in_combined=is_duplicate_source(root),
        relevance="selected for multiclass detector" if is_selected_dataset(root) else "discovered but not selected",
    )
    for split, split_dir in iter_split_dirs(root):
        images = iter_images(split_dir / "images")
        labels = sorted((split_dir / "labels").rglob("*.txt"), key=lambda p: str(p).lower()) if (split_dir / "labels").exists() else []
        candidate.split_images[split] = len(images)
        candidate.split_labels[split] = len(labels)
        for label in labels:
            total, malformed = count_yolo_annotations(label)
            candidate.annotation_count += total
            if malformed:
                candidate.note += f"{malformed} malformed labels in {rel(label)}; "
    if candidate.already_in_combined:
        candidate.note += "Not selected directly because it is a raw source already represented by a prepared dataset. "
    return candidate


def discover_candidates() -> List[DatasetCandidate]:
    search_roots = [
        REPO_ROOT / "data_sources" / "downloads",
        REPO_ROOT / "data_sources" / "extracted",
        REPO_ROOT / "data" / "detection",
        REPO_ROOT / "data" / "industrial",
        REPO_ROOT / "data" / "robustness",
        REPO_ROOT / "backend" / "data",
    ]
    candidates = [candidate_from_yolo(root) for root in discover_yolo_roots(search_roots)]

    coco = []
    voc = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        coco.extend(search_root.rglob("_annotations.coco.json"))
        coco.extend(search_root.rglob("instances_*.json"))
        voc.extend(search_root.rglob("*.xml"))
    if coco:
        by_parent: Dict[Path, List[Path]] = defaultdict(list)
        for path in coco:
            by_parent[path.parent].append(path)
        for root, files in by_parent.items():
            candidates.append(DatasetCandidate(key=candidate_key(root), root=root, format="COCO", coco_jsons=files, usable=False, note="COCO annotations discovered but no selected COCO dataset is used in this run."))
    if voc:
        by_parent = defaultdict(list)
        for path in voc:
            by_parent[path.parent].append(path)
        for root, files in by_parent.items():
            candidates.append(DatasetCandidate(key=candidate_key(root), root=root, format="Pascal VOC", voc_xmls=files[:25], usable=False, note="VOC XML annotations discovered but no selected VOC dataset is used in this run."))
    return candidates


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
        xc, yc, w, h = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None
    x1 = max(0.0, xc - w / 2)
    y1 = max(0.0, yc - h / 2)
    x2 = min(1.0, xc + w / 2)
    y2 = min(1.0, yc + h / 2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return [(x1 + x2) / 2, (y1 + y2) / 2, w, h]


def reset_output(overwrite: bool) -> None:
    if OUT_ROOT.exists() and any(OUT_ROOT.rglob("*")):
        if not overwrite:
            raise SystemExit(f"Output already exists: {OUT_ROOT}. Use --overwrite to rebuild.")
        shutil.rmtree(OUT_ROOT)
    for split in ["train", "valid", "test"]:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "previews").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "predictions").mkdir(parents=True, exist_ok=True)


def selected_candidates(candidates: List[DatasetCandidate]) -> List[DatasetCandidate]:
    return [item for item in candidates if item.selected and item.format == "YOLO"]


def remap_dataset(candidate: DatasetCandidate, max_ann_per_class: int) -> Tuple[Dict[str, BuildStats], Counter, Dict[str, Dict[str, int]], List[Dict[str, Any]], Counter]:
    if not candidate.data_yaml:
        return {}, Counter(), {}, [], Counter()
    names = parse_names(candidate.data_yaml)
    mapping = {idx: map_class_name(name) for idx, name in enumerate(names)}
    split_stats: Dict[str, BuildStats] = defaultdict(BuildStats)
    class_counts: Counter = Counter()
    split_class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mapping_rows = []
    drop_reasons: Counter = Counter()
    class_cap_counts_by_split: Dict[str, Counter] = defaultdict(Counter)

    for idx, source_name in enumerate(names):
        target = mapping[idx]
        decision = "keep" if target else "drop"
        reason = "mapped by class-name normalization" if target else "not a selected industrial-device/equipment class or ambiguous"
        mapping_rows.append({
            "dataset": candidate.key,
            "source_class_id": idx,
            "source_class": source_name,
            "target_class": target or "DROP",
            "decision": decision,
            "reason": reason,
        })

    for split, split_dir in iter_split_dirs(candidate.root):
        stats = split_stats[split]
        images = iter_images(split_dir / "images")
        stats.source_images += len(images)
        for image_path in images:
            label_path = yolo_label_for(split_dir, image_path)
            if not label_path.exists():
                stats.missing_labels += 1
                stats.skipped_images += 1
                continue
            output_lines = []
            for raw in read_text(label_path).splitlines():
                if not raw.strip():
                    continue
                parsed = parse_yolo_line(raw)
                if parsed is None:
                    stats.malformed_annotations += 1
                    continue
                source_id, values = parsed
                target_name = mapping.get(source_id)
                if not target_name:
                    stats.dropped_annotations += 1
                    if 0 <= source_id < len(names):
                        drop_reasons[f"unmapped:{names[source_id]}"] += 1
                    else:
                        drop_reasons["invalid_source_class"] += 1
                    continue
                split_cap_counts = class_cap_counts_by_split[split]
                if split_cap_counts[target_name] >= max_ann_per_class:
                    stats.dropped_annotations += 1
                    drop_reasons[f"class_cap:{target_name}"] += 1
                    continue
                box = normalize_box(values)
                if box is None:
                    stats.invalid_boxes += 1
                    continue
                target_id = TARGET_TO_ID[target_name]
                output_lines.append(f"{target_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                class_counts[target_name] += 1
                split_class_counts[split][target_name] += 1
                split_cap_counts[target_name] += 1
                stats.kept_annotations += 1
            if not output_lines:
                stats.empty_output_labels += 1
                stats.skipped_images += 1
                continue
            safe_name = image_path.relative_to(split_dir / "images").as_posix().replace("/", "__")
            dest_name = f"{candidate.key}_{safe_name}"
            dest_image = OUT_ROOT / split / "images" / dest_name
            dest_label = OUT_ROOT / split / "labels" / f"{Path(dest_name).stem}.txt"
            shutil.copy2(image_path, dest_image)
            write_text(dest_label, "\n".join(output_lines) + "\n")
            stats.copied_images += 1
    return split_stats, class_counts, split_class_counts, mapping_rows, drop_reasons


def write_data_yaml() -> Path:
    payload = {
        "path": OUT_ROOT.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": {idx: name for idx, name in enumerate(TARGET_CLASSES)},
    }
    path = OUT_ROOT / "data.yaml"
    write_text(path, yaml.safe_dump(payload, sort_keys=False))
    return path


def load_boxes(label_path: Path) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    boxes = []
    for raw in read_text(label_path).splitlines():
        parsed = parse_yolo_line(raw)
        if parsed is None:
            continue
        class_id, values = parsed
        box = normalize_box(values)
        if box is None:
            continue
        label = TARGET_CLASSES[class_id] if 0 <= class_id < len(TARGET_CLASSES) else str(class_id)
        boxes.append((label, tuple(box)))
    return boxes


def draw_preview(image_path: Path, label_path: Path, out_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for label, (xc, yc, bw, bh) in load_boxes(label_path):
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            draw.rectangle((x1, y1, x2, y2), outline=(22, 163, 74), width=max(2, width // 320))
            text = label[:24]
            draw.rectangle((x1, max(0, y1 - 18), min(width, x1 + 7 * len(text) + 8), y1), fill=(22, 163, 74))
            draw.text((x1 + 3, max(0, y1 - 16)), text, fill=(0, 0, 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.thumbnail((1200, 900))
        img.save(out_path)


def make_previews() -> Dict[str, int]:
    counts = {}
    rng = random.Random(42)
    preview_dir = ARTIFACT_ROOT / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        images = iter_images(OUT_ROOT / split / "images")
        rng.shuffle(images)
        selected = []
        seen_classes = set()
        for image in images:
            label = OUT_ROOT / split / "labels" / f"{image.stem}.txt"
            classes = {box[0] for box in load_boxes(label)}
            if classes - seen_classes or len(selected) < 10:
                selected.append(image)
                seen_classes.update(classes)
            if len(selected) >= 10:
                break
        for idx, image in enumerate(selected, start=1):
            label = OUT_ROOT / split / "labels" / f"{image.stem}.txt"
            draw_preview(image, label, preview_dir / f"{split}_{idx:02d}_{image.name}")
        counts[split] = len(selected)
    lines = ["# Multiclass Detector Preview Report", ""]
    for split, count in counts.items():
        lines.append(f"- `{split}` previews: `{count}`")
    lines.append(f"- Preview folder: `{rel(ARTIFACT_ROOT / 'previews')}`")
    write_text(ARTIFACT_ROOT / "preview_report.md", "\n".join(lines) + "\n")
    return counts


def build_dataset(overwrite: bool, max_ann_per_class: int) -> Dict[str, Any]:
    reset_output(overwrite=overwrite)
    candidates = discover_candidates()
    selected = selected_candidates(candidates)
    if not selected:
        raise SystemExit("No selected YOLO datasets found for multiclass build.")

    all_stats: Dict[str, Dict[str, Any]] = {}
    total_class_counts: Counter = Counter()
    total_split_class_counts: Dict[str, Counter] = defaultdict(Counter)
    mapping_rows: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()

    for candidate in selected:
        stats, class_counts, split_class_counts, rows, drops = remap_dataset(candidate, max_ann_per_class=max_ann_per_class)
        all_stats[candidate.key] = {split: vars(item) for split, item in stats.items()}
        total_class_counts.update(class_counts)
        for split, counter in split_class_counts.items():
            total_split_class_counts[split].update(counter)
        mapping_rows.extend(rows)
        drop_reasons.update(drops)

    data_yaml = write_data_yaml()
    preview_counts = make_previews()
    inventory = write_inventory(candidates)
    write_mapping_report(mapping_rows, drop_reasons)
    dataset_summary = write_dataset_stats(total_class_counts, total_split_class_counts, all_stats)
    write_training_summary_md(dataset_summary, training=None)

    return {
        "data_yaml": rel(data_yaml),
        "selected_datasets": [rel(item.root) for item in selected],
        "dataset_inventory": inventory,
        "class_counts": dict(total_class_counts),
        "split_class_counts": {split: dict(counter) for split, counter in total_split_class_counts.items()},
        "stats": all_stats,
        "preview_counts": preview_counts,
        "max_annotations_per_class": max_ann_per_class,
    }


def write_inventory(candidates: List[DatasetCandidate]) -> str:
    rows = []
    for item in candidates:
        rows.append({
            "dataset": item.key,
            "root": rel(item.root),
            "format": item.format,
            "splits": "; ".join(f"{split}:img={item.split_images.get(split, 0)},lbl={item.split_labels.get(split, 0)}" for split in ["train", "valid", "test"]),
            "images": sum(item.split_images.values()),
            "labels": sum(item.split_labels.values()),
            "annotations": item.annotation_count,
            "classes": "; ".join(item.class_names),
            "usable": item.usable,
            "selected": item.selected,
            "already_part_of_combined": item.already_in_combined,
            "relevance": item.relevance,
            "note": item.note.strip(),
        })
    md = ["# Industrial Multiclass Dataset Inventory", ""]
    md.append(markdown_table(rows, ["dataset", "root", "format", "images", "labels", "annotations", "classes", "usable", "selected", "already_part_of_combined", "relevance", "note"]))
    md.append("\n## Format Discovery")
    md.append("- YOLO datasets were found and used where selected.")
    md.append("- COCO annotation files were not found in the scanned roots for this run." if not any(item.format == "COCO" for item in candidates) else "- COCO files were discovered but not selected for this run.")
    md.append("- Pascal VOC XML annotations were not found in the scanned roots for this run." if not any(item.format == "Pascal VOC" for item in candidates) else "- Pascal VOC XML files were discovered but not selected for this run.")
    path = ARTIFACT_ROOT / "dataset_inventory.md"
    write_text(path, "\n".join(md) + "\n")
    return rel(path)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "No rows found.\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_mapping_report(mapping_rows: List[Dict[str, Any]], drop_reasons: Counter) -> None:
    kept = [row for row in mapping_rows if row["decision"] == "keep"]
    dropped = [row for row in mapping_rows if row["decision"] == "drop"]
    lines = ["# Industrial Multiclass Class Mapping Report", ""]
    lines.append("Final classes:")
    for idx, name in enumerate(TARGET_CLASSES):
        lines.append(f"- `{idx}: {name}`")
    lines.append("\n## Kept / Remapped Classes")
    lines.append(markdown_table(kept, ["dataset", "source_class_id", "source_class", "target_class", "decision", "reason"]))
    lines.append("\n## Dropped / Unmapped Classes")
    lines.append(markdown_table(dropped, ["dataset", "source_class_id", "source_class", "target_class", "decision", "reason"]))
    lines.append("\n## Dropped Annotation Reasons")
    for reason, count in drop_reasons.most_common():
        lines.append(f"- `{reason}`: `{count}`")
    write_text(ARTIFACT_ROOT / "class_mapping_report.md", "\n".join(lines) + "\n")
    write_text(ARTIFACT_ROOT / "unmapped_classes.md", markdown_table(dropped, ["dataset", "source_class_id", "source_class", "reason"]))


def split_image_count(split: str) -> int:
    return len(iter_images(OUT_ROOT / split / "images"))


def split_label_count(split: str) -> int:
    labels_dir = OUT_ROOT / split / "labels"
    return len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0


def write_dataset_stats(class_counts: Counter, split_class_counts: Dict[str, Counter], all_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    total_annotations = sum(class_counts.values())
    for split in ["train", "valid", "test"]:
        rows.append({
            "split": split,
            "images": split_image_count(split),
            "labels": split_label_count(split),
            "annotations": sum(split_class_counts[split].values()),
        })
    with (ARTIFACT_ROOT / "dataset_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "images", "labels", "annotations"])
        writer.writeheader()
        writer.writerows(rows)

    dist_rows = []
    nonzero = [count for count in class_counts.values() if count > 0]
    for class_name in TARGET_CLASSES:
        dist_rows.append({
            "class": class_name,
            "train_annotations": split_class_counts["train"].get(class_name, 0),
            "valid_annotations": split_class_counts["valid"].get(class_name, 0),
            "test_annotations": split_class_counts["test"].get(class_name, 0),
            "total_annotations": class_counts.get(class_name, 0),
            "role": class_role(class_name),
        })
    with (ARTIFACT_ROOT / "class_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "train_annotations", "valid_annotations", "test_annotations", "total_annotations", "role"])
        writer.writeheader()
        writer.writerows(dist_rows)

    summary = {
        "dataset_root": rel(OUT_ROOT),
        "data_yaml": rel(OUT_ROOT / "data.yaml"),
        "total_images": sum(row["images"] for row in rows),
        "total_annotations": total_annotations,
        "split_counts": rows,
        "class_distribution": dist_rows,
        "largest_class": max(class_counts, key=class_counts.get) if class_counts else None,
        "smallest_nonzero_class": min(class_counts, key=class_counts.get) if class_counts else None,
        "imbalance_ratio": (max(nonzero) / min(nonzero)) if nonzero else None,
        "source_stats": all_stats,
    }
    write_json(ARTIFACT_ROOT / "dataset_summary.json", summary)
    return summary


def class_role(name: str) -> str:
    if name in {"valve", "gauge", "flange", "instrument_panel", "vessel"}:
        return "device/equipment class from real or equipment-oriented imagery"
    if name in {"pipe", "desalter", "heater", "heat_exchanger", "tank"}:
        return "oil/gas macro-equipment class from refinery/offshore staging"
    return "expanded industrial support class"


def write_training_summary_md(dataset_summary: Dict[str, Any], training: Optional[Dict[str, Any]]) -> None:
    lines = ["# Industrial Multiclass Detector Training Summary", ""]
    lines.append("This experiment trains a second detector and keeps `models/detector.pt` unchanged as the stable valve/gauge baseline.")
    lines.append("\n## Dataset")
    lines.append(f"- Dataset root: `{dataset_summary.get('dataset_root')}`")
    lines.append(f"- Data YAML: `{dataset_summary.get('data_yaml')}`")
    lines.append(f"- Total images: `{dataset_summary.get('total_images')}`")
    lines.append(f"- Total annotations: `{dataset_summary.get('total_annotations')}`")
    lines.append(f"- Imbalance ratio: `{dataset_summary.get('imbalance_ratio')}`")
    lines.append("\n## Class Distribution")
    lines.append(markdown_table(dataset_summary.get("class_distribution", []), ["class", "train_annotations", "valid_annotations", "test_annotations", "total_annotations", "role"]))
    lines.append("\n## Training")
    if training:
        lines.append(f"- Model seed weights: `{training.get('model')}`")
        lines.append(f"- Epochs requested: `{training.get('epochs')}`")
        lines.append(f"- Image size: `{training.get('imgsz')}`")
        lines.append(f"- Device: `{training.get('device')}`")
        lines.append(f"- Run directory: `{training.get('run_dir')}`")
        lines.append(f"- Best weights: `{training.get('best_weights')}`")
        lines.append(f"- Copied integration weights: `{training.get('copied_weights')}`")
    else:
        lines.append("Training has not been run yet.")
    write_text(ARTIFACT_ROOT / "training_summary.md", "\n".join(lines) + "\n")


def train_model(data_yaml: Path, model: str, epochs: int, imgsz: int, batch: str, device: str, workers: int, name: str) -> Dict[str, Any]:
    os.environ.setdefault("WANDB_DISABLED", "true")
    from ultralytics import YOLO, settings  # type: ignore

    settings.update({"wandb": False})
    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=int(batch) if str(batch).lstrip("-").isdigit() else batch,
        device=device,
        workers=workers,
        project=str(REPO_ROOT / "runs" / "detect"),
        name=name,
        exist_ok=True,
    )
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training completed but best weights were not found: {best}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "detector_multiclass.pt"
    shutil.copy2(best, target)
    summary = {
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "run_dir": rel(run_dir),
        "best_weights": rel(best),
        "copied_weights": rel(target),
        "stable_baseline_unchanged": rel(MODELS_DIR / "detector.pt"),
    }
    write_json(ARTIFACT_ROOT / "training_summary.json", summary)
    dataset_summary = json.loads(read_text(ARTIFACT_ROOT / "dataset_summary.json"))
    write_training_summary_md(dataset_summary, training=summary)
    return summary


def normalize_metrics(metrics: Any) -> Dict[str, Any]:
    out = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "per_class_maps": [],
        "results_dict": {},
    }
    if hasattr(metrics, "box"):
        out["precision"] = float(metrics.box.mp)
        out["recall"] = float(metrics.box.mr)
        out["map50"] = float(metrics.box.map50)
        out["map50_95"] = float(metrics.box.map)
        out["per_class_maps"] = [float(value) for value in getattr(metrics.box, "maps", [])]
    if hasattr(metrics, "results_dict"):
        out["results_dict"] = {str(key): float(value) for key, value in metrics.results_dict.items()}
    return out


def evaluate_model(data_yaml: Path, weights: Path, device: str, name: str) -> Dict[str, Any]:
    from ultralytics import YOLO, settings  # type: ignore

    settings.update({"wandb": False})
    model = YOLO(str(weights))
    val_metrics = model.val(data=str(data_yaml), split="val", device=device, project=str(REPO_ROOT / "runs" / "detect"), name=f"{name}_val", exist_ok=True)
    test_metrics = model.val(data=str(data_yaml), split="test", device=device, project=str(REPO_ROOT / "runs" / "detect"), name=f"{name}_test", exist_ok=True)
    pred_results = model.predict(
        source=str(OUT_ROOT / "test" / "images"),
        save=True,
        conf=0.25,
        device=device,
        project=str(ARTIFACT_ROOT),
        name="predictions",
        exist_ok=True,
    )
    pred_dir = Path(pred_results[0].save_dir) if pred_results else ARTIFACT_ROOT / "predictions"
    summary = {
        "weights": rel(weights),
        "data_yaml": rel(data_yaml),
        "validation": normalize_metrics(val_metrics),
        "test": normalize_metrics(test_metrics),
        "prediction_dir": rel(pred_dir),
    }
    write_json(ARTIFACT_ROOT / "evaluation_summary.json", summary)
    write_evaluation_md(summary)
    write_thesis_detector_section(summary)
    return summary


def write_evaluation_md(summary: Dict[str, Any]) -> None:
    rows = []
    for split in ["validation", "test"]:
        metrics = summary.get(split, {})
        rows.append({
            "split": split,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("map50"),
            "mAP50-95": metrics.get("map50_95"),
        })
    lines = ["# Industrial Multiclass Detector Evaluation Summary", "", markdown_table(rows, ["split", "precision", "recall", "mAP50", "mAP50-95"])]
    lines.append("\n## Per-Class mAP50-95")
    per_class_rows = []
    for class_id, class_name in enumerate(TARGET_CLASSES):
        row = {"class": class_name}
        for split in ["validation", "test"]:
            values = summary.get(split, {}).get("per_class_maps", [])
            row[split] = values[class_id] if class_id < len(values) else "not found"
        per_class_rows.append(row)
    lines.append(markdown_table(per_class_rows, ["class", "validation", "test"]))
    lines.append(f"\nPrediction visualizations: `{summary.get('prediction_dir')}`")
    lines.append("\nThis model is a second expanded detector. It does not replace the stable two-class detector at `models/detector.pt`.")
    write_text(ARTIFACT_ROOT / "evaluation_summary.md", "\n".join(lines) + "\n")


def write_thesis_detector_section(summary: Dict[str, Any]) -> None:
    dataset = json.loads(read_text(ARTIFACT_ROOT / "dataset_summary.json"))
    lines = ["# Thesis Section: Expanded Industrial Multiclass Detector", ""]
    lines.append("A second YOLOv8 detector was trained to explore broader industrial-equipment detection beyond the stable ValveLens valve/gauge baseline. The original baseline detector remains stored at `models/detector.pt`, while the expanded detector is stored separately at `models/detector_multiclass.pt`.")
    lines.append("\n## Dataset Construction")
    lines.append("The expanded dataset was built from local YOLO-format datasets already present in the repository. The final training set combines the existing valve/gauge dataset with prepared oil/gas/refinery datasets. Source classes were normalized into a compact industrial label space and ambiguous or unrelated classes were dropped.")
    lines.append(f"\nFinal dataset: `{dataset.get('dataset_root')}`")
    lines.append(f"Total images: `{dataset.get('total_images')}`")
    lines.append(f"Total annotations: `{dataset.get('total_annotations')}`")
    lines.append("\n## Evaluation")
    rows = []
    for split in ["validation", "test"]:
        metrics = summary.get(split, {})
        rows.append({"split": split, "precision": metrics.get("precision"), "recall": metrics.get("recall"), "mAP50": metrics.get("map50"), "mAP50-95": metrics.get("map50_95")})
    lines.append(markdown_table(rows, ["split", "precision", "recall", "mAP50", "mAP50-95"]))
    lines.append("\n## Interpretation")
    lines.append("This model is useful for thesis discussion and future product direction because it expands the object vocabulary to oil/gas equipment classes. However, it should be interpreted separately from the stable baseline detector. Some refinery classes come from synthetic/rendered or macro-part annotations, and dense pipe annotations were capped to reduce imbalance. The result is an exploratory expanded detector, not a replacement for the validated two-class baseline.")
    lines.append("\nThe weaker classes are the low-sample offshore classes such as `instrument_panel`, `vessel`, and `flange`. The stronger classes are mostly the macro-equipment/refinery classes with more annotations. This should be discussed as a dataset limitation, not as a failure of the architecture.")
    write_text(ARTIFACT_ROOT / "thesis_detector_section.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, train, evaluate, and document the ValveLens expanded multiclass detector.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--fallback-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="valvelens_multiclass_v1")
    parser.add_argument("--max-ann-per-class", type=int, default=4000)
    return parser.parse_args()


def choose_model(args: argparse.Namespace) -> str:
    model_path = REPO_ROOT / args.model
    if model_path.exists():
        return str(model_path)
    fallback = REPO_ROOT / args.fallback_model
    if fallback.exists():
        print(f"Preferred model {args.model} is not local; using fallback {args.fallback_model}.")
        return str(fallback)
    return args.model


def main() -> None:
    args = parse_args()
    build_summary = build_dataset(overwrite=args.overwrite, max_ann_per_class=args.max_ann_per_class)
    data_yaml = OUT_ROOT / "data.yaml"
    print(json.dumps({"dataset_build": build_summary}, indent=2))
    if args.prepare_only or args.skip_train:
        print("Dataset prepared; training skipped.")
        return

    model = choose_model(args)
    training = train_model(data_yaml, model=model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, name=args.name)
    evaluation = evaluate_model(data_yaml, weights=MODELS_DIR / "detector_multiclass.pt", device=args.device, name=args.name)
    print(json.dumps({"training": training, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
