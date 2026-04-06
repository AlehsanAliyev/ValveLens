from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required to inspect the dataset: {exc}")


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "detection" / "combined"
DATA_YAML = DATASET_ROOT / "data.yaml"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise SystemExit(f"Missing data.yaml: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML object in {path}")
    return data


def split_dirs(split: str) -> Tuple[Path, Path]:
    return DATASET_ROOT / split / "images", DATASET_ROOT / split / "labels"


def paired_label_path(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def inspect_label_file(label_path: Path) -> Dict:
    summary = {
        "annotation_count": 0,
        "class_counts": Counter(),
        "malformed_lines": [],
        "invalid_class_ids": [],
        "invalid_boxes": [],
        "extreme_boxes": [],
        "empty": False,
    }

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        summary["empty"] = True
        return summary

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            summary["malformed_lines"].append(
                {"line_no": line_no, "reason": "expected 5 columns", "line": line}
            )
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            summary["malformed_lines"].append(
                {"line_no": line_no, "reason": "non-numeric value", "line": line}
            )
            continue

        if class_id not in (0, 1):
            summary["invalid_class_ids"].append({"line_no": line_no, "class_id": class_id, "line": line})
            continue

        x_center, y_center, width, height = coords
        if width <= 0 or height <= 0:
            summary["invalid_boxes"].append({"line_no": line_no, "reason": "non-positive area", "line": line})
            continue

        if not all(0.0 <= value <= 1.0 for value in coords):
            summary["invalid_boxes"].append({"line_no": line_no, "reason": "coordinate outside [0,1]", "line": line})
            continue

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
            summary["invalid_boxes"].append({"line_no": line_no, "reason": "box extends outside image", "line": line})
            continue

        if width > 0.95 or height > 0.95 or width < 0.01 or height < 0.01:
            summary["extreme_boxes"].append({"line_no": line_no, "line": line})

        summary["annotation_count"] += 1
        summary["class_counts"][class_id] += 1

    return summary


def inspect_split(split: str) -> Dict:
    images_dir, labels_dir = split_dirs(split)
    issues: Dict[str, List] = defaultdict(list)
    class_counts = Counter()
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    label_files = sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    image_basenames = {path.stem for path in image_files}
    label_basenames = {path.stem for path in label_files}

    missing_labels = []
    for image_path in image_files:
        label_path = paired_label_path(labels_dir, image_path)
        if not label_path.exists():
            missing_labels.append(str(image_path))
    if missing_labels:
        issues["missing_labels"].extend(missing_labels)

    orphan_labels = sorted(str(path) for path in label_files if path.stem not in image_basenames)
    if orphan_labels:
        issues["orphan_labels"].extend(orphan_labels)

    empty_labels = 0
    malformed_count = 0
    invalid_class_count = 0
    invalid_box_count = 0
    extreme_box_count = 0
    annotated_images = 0

    for label_path in label_files:
        result = inspect_label_file(label_path)
        if result["empty"]:
            empty_labels += 1
            continue
        if result["annotation_count"] > 0:
            annotated_images += 1
        class_counts.update(result["class_counts"])
        malformed_count += len(result["malformed_lines"])
        invalid_class_count += len(result["invalid_class_ids"])
        invalid_box_count += len(result["invalid_boxes"])
        extreme_box_count += len(result["extreme_boxes"])
        if result["malformed_lines"]:
            issues["malformed_lines"].append({"file": str(label_path), "items": result["malformed_lines"][:5]})
        if result["invalid_class_ids"]:
            issues["invalid_class_ids"].append({"file": str(label_path), "items": result["invalid_class_ids"][:5]})
        if result["invalid_boxes"]:
            issues["invalid_boxes"].append({"file": str(label_path), "items": result["invalid_boxes"][:5]})
        if result["extreme_boxes"]:
            issues["extreme_boxes"].append({"file": str(label_path), "items": result["extreme_boxes"][:5]})

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(image_files),
        "label_count": len(label_files),
        "paired_images_with_labels": len(image_basenames & label_basenames),
        "annotated_images": annotated_images,
        "empty_label_files": empty_labels,
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "issue_counts": {
            "missing_labels": len(issues["missing_labels"]),
            "orphan_labels": len(issues["orphan_labels"]),
            "malformed_lines": malformed_count,
            "invalid_class_ids": invalid_class_count,
            "invalid_boxes": invalid_box_count,
            "extreme_boxes": extreme_box_count,
        },
        "issues": issues,
    }


def main() -> None:
    config = load_config(DATA_YAML)
    names = config.get("names", {})
    expected_names = {0: "valve", 1: "gauge"}
    if isinstance(names, list):
        names = {idx: value for idx, value in enumerate(names)}
    resolved_names = {int(k): str(v) for k, v in names.items()}

    split_reports = {}
    for split in SPLITS:
        images_dir, labels_dir = split_dirs(split)
        if not images_dir.exists() or not labels_dir.exists():
            raise SystemExit(f"Missing split directories for '{split}': {images_dir} / {labels_dir}")
        split_reports[split] = inspect_split(split)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "data_yaml": str(DATA_YAML),
        "config_valid": {
            "nc": config.get("nc"),
            "names": resolved_names,
            "matches_expected": config.get("nc") == 2 and resolved_names == expected_names,
        },
        "splits": split_reports,
    }

    print(json.dumps(summary, indent=2))

    total_invalid = 0
    for report in split_reports.values():
        total_invalid += sum(
            report["issue_counts"][key]
            for key in ("missing_labels", "orphan_labels", "malformed_lines", "invalid_class_ids", "invalid_boxes")
        )
    if total_invalid > 0 or not summary["config_valid"]["matches_expected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
