from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_CANDIDATES = ("train", "valid", "val", "test")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise ValueError("Source data.yaml does not define class names.")


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "item"


def _split_dirs(source: Path) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    for split in SPLIT_CANDIDATES:
        root = source / split
        images = root / "images"
        labels = root / "labels"
        if images.exists() and labels.exists():
            found.append((split, images, labels))
    return found


def _validate_label_file(label_path: Path, class_count: int) -> tuple[int, Counter[int], list[str]]:
    annotations = 0
    class_counts: Counter[int] = Counter()
    errors: list[str] = []
    if not label_path.exists():
        errors.append(f"missing label file: {label_path}")
        return annotations, class_counts, errors

    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(item) for item in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no}: non-numeric YOLO label")
            continue
        if cls < 0 or cls >= class_count:
            errors.append(f"{label_path}:{line_no}: class {cls} outside 0..{class_count - 1}")
            continue
        if any(item < 0.0 or item > 1.0 for item in coords):
            errors.append(f"{label_path}:{line_no}: normalized coordinates outside 0..1")
            continue
        annotations += 1
        class_counts[cls] += 1
    return annotations, class_counts, errors


def build_fullfit(source: Path, out_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    source = source.resolve()
    out_dir = out_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    data_yaml = source / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing source data.yaml: {data_yaml}")

    data = _load_yaml(data_yaml)
    names = _class_names(data)
    splits = _split_dirs(source)
    if not splits:
        raise FileNotFoundError(f"No train/valid/val/test images+labels folders found under {source}")

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_files_count = 0
    annotation_count = 0
    class_counts: Counter[int] = Counter()
    errors: list[str] = []
    copied: list[dict[str, str]] = []

    for split_name, images_dir, labels_dir in splits:
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            ann, counts, label_errors = _validate_label_file(label_path, len(names))
            errors.extend(label_errors)
            if label_errors:
                continue

            prefix = _safe_stem(split_name)
            dest_stem = f"{prefix}__{image_path.stem}"
            dest_image = images_out / f"{dest_stem}{image_path.suffix.lower()}"
            dest_label = labels_out / f"{dest_stem}.txt"
            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)
            image_count += 1
            label_files_count += 1
            annotation_count += ann
            class_counts.update(counts)
            copied.append(
                {
                    "split": split_name,
                    "image": str(image_path),
                    "label": str(label_path),
                    "dest_image": str(dest_image),
                    "dest_label": str(dest_label),
                }
            )

    if errors:
        error_path = artifacts_dir / "fullfit_label_validation_errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise ValueError(
            f"Found {len(errors)} YOLO label validation error(s). See {error_path}"
        )

    full_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(names),
        "names": {idx: name for idx, name in enumerate(names)},
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(full_yaml, handle, sort_keys=False)

    split_counts: Counter[str] = Counter(item["split"] for item in copied)
    summary = {
        "source_dataset": str(source),
        "output_dataset": str(out_dir),
        "class_names": names,
        "total_images": image_count,
        "total_label_files": label_files_count,
        "total_annotations": annotation_count,
        "split_image_counts": dict(split_counts),
        "class_distribution": {
            names[idx]: int(class_counts.get(idx, 0)) for idx in range(len(names))
        },
        "data_yaml": str(out_dir / "data.yaml"),
    }
    (artifacts_dir / "fullfit_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Full-fit Detector Dataset Summary",
        "",
        "This dataset merges all available source splits for demo / in-sample verification.",
        "It is not a held-out train/validation/test split.",
        "",
        f"- Source dataset: `{source}`",
        f"- Output dataset: `{out_dir}`",
        f"- Data YAML: `{out_dir / 'data.yaml'}`",
        f"- Total images: `{image_count}`",
        f"- Total label files: `{label_files_count}`",
        f"- Total annotations: `{annotation_count}`",
        "",
        "## Splits merged",
        "",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"- `{split}`: `{count}` images")
    lines.extend(["", "## Class distribution", ""])
    for idx, name in enumerate(names):
        lines.append(f"- `{idx}` / `{name}`: `{class_counts.get(idx, 0)}`")
    lines.append("")
    (artifacts_dir / "fullfit_dataset_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Build the ValveLens full-fit YOLO dataset.")
    parser.add_argument(
        "--source",
        default=str(root / "data" / "detection" / "industrial_multiclass"),
    )
    parser.add_argument(
        "--fallback",
        default=str(root / "data" / "detection" / "combined"),
    )
    parser.add_argument(
        "--out",
        default=str(root / "data" / "detection" / "fullfit"),
    )
    parser.add_argument(
        "--artifacts",
        default=str(root / "artifacts" / "fullfit_detector"),
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        source = Path(args.fallback)
    summary = build_fullfit(source, Path(args.out), Path(args.artifacts))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
