from __future__ import annotations

import ast
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = REPO_ROOT / "data_sources" / "downloads"
EXTRACTED_DIR = REPO_ROOT / "data_sources" / "extracted"
COMBINED_DIR = REPO_ROOT / "data" / "detection" / "combined"

TARGET_ARCHIVES = {
    "Valve Detection.v1i.yolov8.zip": {
        "extract_dir": EXTRACTED_DIR / "valve_detection_v1i_yolov8",
        "prefix": "v1i",
    },
    "Valve Detection.v6i.yolov8.zip": {
        "extract_dir": EXTRACTED_DIR / "valve_detection_v6i_yolov8",
        "prefix": "v6i",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}
FINAL_CLASS_TO_ID = {"valve": 0, "gauge": 1}


@dataclass
class DatasetSummary:
    archive_name: str
    archive_path: Path
    extract_dir: Path
    dataset_root: Path
    prefix: str
    original_names: List[str]
    class_mapping: Dict[int, Optional[int]]
    split_image_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    split_label_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    annotations_rewritten: int = 0
    annotations_dropped: int = 0
    polygon_annotations_converted: int = 0
    empty_labels_created: int = 0
    missing_labels: int = 0
    copied_images: int = 0
    problematic_files: List[str] = field(default_factory=list)


def _find_archives() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not DOWNLOADS_DIR.exists():
        raise SystemExit(f"Downloads directory not found: {DOWNLOADS_DIR}")

    by_name = {path.name: path for path in DOWNLOADS_DIR.rglob("*.zip")}
    missing = []
    for name in TARGET_ARCHIVES:
        path = by_name.get(name)
        if path is None:
            missing.append(name)
        else:
            found[name] = path.resolve()
    if missing:
        raise SystemExit(
            "Missing required archive(s): " + ", ".join(missing)
        )
    return found


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if _looks_like_dataset_extract(extract_dir):
        return
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)


def _looks_like_dataset_extract(root: Path) -> bool:
    return _detect_dataset_root(root) is not None


def _candidate_score(path: Path) -> int:
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


def _detect_dataset_root(extract_dir: Path) -> Optional[Path]:
    candidates = set()
    if (extract_dir / "data.yaml").exists():
        candidates.add(extract_dir)
    for yaml_path in extract_dir.rglob("data.yaml"):
        candidates.add(yaml_path.parent)

    valid_candidates = []
    for candidate in candidates:
        score = _candidate_score(candidate)
        if score >= 8:
            valid_candidates.append((score, len(candidate.relative_to(extract_dir).parts), candidate))
    if not valid_candidates:
        return None
    valid_candidates.sort(key=lambda item: (-item[0], item[1], str(item[2]).lower()))
    return valid_candidates[0][2]


def _load_yaml(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError("YAML root is not a mapping")
        return data
    except Exception:
        return _parse_yaml_fallback(raw, path)


def _parse_yaml_fallback(raw: str, path: Path) -> dict:
    data: Dict[str, object] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, remainder = line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        if key == "names":
            if remainder:
                try:
                    data["names"] = ast.literal_eval(remainder)
                except Exception as exc:
                    raise SystemExit(f"Failed to parse names in {path}: {exc}") from exc
            else:
                mapping: Dict[int, str] = {}
                items: List[str] = []
                i += 1
                while i < len(lines):
                    child = lines[i]
                    if not child.startswith(" ") and not child.startswith("\t"):
                        i -= 1
                        break
                    child_stripped = child.strip()
                    if not child_stripped:
                        i += 1
                        continue
                    if child_stripped.startswith("- "):
                        items.append(child_stripped[2:].strip().strip("'\""))
                    elif ":" in child_stripped:
                        idx_s, value = child_stripped.split(":", 1)
                        mapping[int(idx_s.strip())] = value.strip().strip("'\"")
                    i += 1
                data["names"] = mapping if mapping else items
        else:
            value = remainder.strip().strip("'\"")
            data[key] = value
        i += 1
    if "names" not in data:
        raise SystemExit(f"Malformed data.yaml, no names found: {path}")
    return data


def _parse_names(yaml_data: dict, yaml_path: Path) -> List[str]:
    names = yaml_data.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        try:
            ordered = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda item: item[0])
        except Exception as exc:
            raise SystemExit(f"Malformed names mapping in {yaml_path}: {exc}") from exc
        return [value for _, value in ordered]
    raise SystemExit(f"Unsupported names format in {yaml_path}")


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _resolve_class_mapping(class_names: Iterable[str]) -> Dict[int, Optional[int]]:
    mapping: Dict[int, Optional[int]] = {}
    for idx, name in enumerate(class_names):
        normalized = _normalize_class_name(name)
        if "gauge" in normalized:
            mapping[idx] = FINAL_CLASS_TO_ID["gauge"]
        elif "valve" in normalized:
            mapping[idx] = FINAL_CLASS_TO_ID["valve"]
        else:
            mapping[idx] = None
    if all(value is None for value in mapping.values()):
        raise SystemExit(
            "Could not map any classes to final label space from class names: "
            + ", ".join(class_names)
        )
    return mapping


def _iter_split_dirs(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        canonical = SPLIT_ALIASES.get(child.name.lower())
        if canonical:
            yield canonical, child


def _clear_combined_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for split in ("train", "valid", "test"):
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)


def _safe_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _rewrite_label_lines(
    lines: Iterable[str],
    class_mapping: Dict[int, Optional[int]],
    source_label: Path,
    summary: DatasetSummary,
    final_class_counter: Counter,
) -> List[str]:
    rewritten: List[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            src_class = int(float(parts[0]))
            raw_values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(
                f"Invalid numeric YOLO label in {source_label} line {line_number}: {stripped}"
            ) from exc

        if len(parts) == 5:
            coords = raw_values
        elif len(raw_values) >= 6 and len(raw_values) % 2 == 0:
            xs = raw_values[0::2]
            ys = raw_values[1::2]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            coords = [
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                x_max - x_min,
                y_max - y_min,
            ]
            summary.polygon_annotations_converted += 1
        else:
            raise SystemExit(
                f"Invalid YOLO label format in {source_label} line {line_number}: {stripped}"
            )

        dst_class = class_mapping.get(src_class)
        if dst_class is None:
            summary.annotations_dropped += 1
            continue

        x_center, y_center, width, height = coords
        x1 = max(0.0, x_center - width / 2.0)
        y1 = max(0.0, y_center - height / 2.0)
        x2 = min(1.0, x_center + width / 2.0)
        y2 = min(1.0, y_center + height / 2.0)
        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            summary.annotations_dropped += 1
            continue
        coords = [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            width,
            height,
        ]

        rewritten.append(
            f"{dst_class} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
        )
        summary.annotations_rewritten += 1
        final_class_counter[dst_class] += 1
    return rewritten


def _copy_and_rewrite_dataset(
    summary: DatasetSummary,
    final_class_counter: Counter,
) -> None:
    for split_name, split_dir in _iter_split_dirs(summary.dataset_root):
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.rglob("*"), key=lambda p: str(p).lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel_name = image_path.relative_to(images_dir).as_posix().replace("/", "__")
            dest_stem = f"{summary.prefix}_{rel_name}"
            dest_image = COMBINED_DIR / split_name / "images" / dest_stem
            dest_label = (
                COMBINED_DIR / split_name / "labels" / f"{Path(dest_stem).stem}.txt"
            )

            if dest_image.exists() or dest_label.exists():
                raise SystemExit(f"Filename collision detected for {dest_image}")

            shutil.copy2(image_path, dest_image)
            summary.copied_images += 1
            summary.split_image_counts[split_name] += 1

            label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
            if label_path.exists():
                rewritten = _rewrite_label_lines(
                    _safe_lines(label_path),
                    summary.class_mapping,
                    label_path,
                    summary,
                    final_class_counter,
                )
                dest_label.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
                summary.split_label_counts[split_name] += 1
                if not rewritten:
                    summary.empty_labels_created += 1
            else:
                dest_label.write_text("", encoding="utf-8")
                summary.missing_labels += 1
                summary.empty_labels_created += 1
                summary.problematic_files.append(f"Missing label for image: {image_path}")


def _write_final_yaml() -> None:
    yaml_text = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: valve\n"
        "  1: gauge\n"
    )
    (COMBINED_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _validate_combined_dataset() -> Dict[str, int]:
    data_yaml = COMBINED_DIR / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing final data.yaml: {data_yaml}")

    yaml_data = _load_yaml(data_yaml)
    names = _parse_names(yaml_data, data_yaml)
    if names != ["valve", "gauge"]:
        raise SystemExit(f"Final data.yaml names mismatch: {names}")

    split_counts: Dict[str, int] = {}
    for split in ("train", "valid", "test"):
        images_dir = COMBINED_DIR / split / "images"
        labels_dir = COMBINED_DIR / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise SystemExit(f"Missing split directory: {split}")

        image_files = sorted(
            [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        split_counts[split] = len(image_files)
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise SystemExit(f"Missing paired label file: {label_path}")
            for line_number, raw_line in enumerate(_safe_lines(label_path), start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    raise SystemExit(
                        f"Invalid YOLO line in {label_path} line {line_number}: {stripped}"
                    )
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError as exc:
                    raise SystemExit(
                        f"Invalid numeric label in {label_path} line {line_number}: {stripped}"
                    ) from exc
                if class_id not in (0, 1):
                    raise SystemExit(
                        f"Unexpected class id {class_id} in {label_path} line {line_number}"
                    )
                if any(value < 0 or value > 1 for value in coords):
                    raise SystemExit(
                        f"Out-of-range bbox value in {label_path} line {line_number}: {stripped}"
                    )
                if coords[2] <= 0 or coords[3] <= 0:
                    raise SystemExit(
                        f"Non-positive bbox size in {label_path} line {line_number}: {stripped}"
                    )
    return split_counts


def _format_mapping(class_names: List[str], class_mapping: Dict[int, Optional[int]]) -> List[str]:
    label_by_id = {0: "valve", 1: "gauge"}
    lines = []
    for idx, name in enumerate(class_names):
        mapped = class_mapping.get(idx)
        mapped_name = label_by_id.get(mapped, "DROP")
        lines.append(f"{idx}: {name} -> {mapped_name}")
    return lines


def main() -> None:
    archives = _find_archives()
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Found archives:")
    for name, path in archives.items():
        print(f"  - {name}: {path}")

    dataset_summaries: List[DatasetSummary] = []
    for archive_name, archive_path in archives.items():
        extract_dir = TARGET_ARCHIVES[archive_name]["extract_dir"]
        prefix = TARGET_ARCHIVES[archive_name]["prefix"]

        _extract_archive(archive_path, extract_dir)
        dataset_root = _detect_dataset_root(extract_dir)
        if dataset_root is None:
            raise SystemExit(f"Could not detect YOLO dataset root under {extract_dir}")

        yaml_path = dataset_root / "data.yaml"
        yaml_data = _load_yaml(yaml_path)
        class_names = _parse_names(yaml_data, yaml_path)
        class_mapping = _resolve_class_mapping(class_names)

        summary = DatasetSummary(
            archive_name=archive_name,
            archive_path=archive_path,
            extract_dir=extract_dir,
            dataset_root=dataset_root,
            prefix=prefix,
            original_names=class_names,
            class_mapping=class_mapping,
        )
        dataset_summaries.append(summary)

    _clear_combined_dir(COMBINED_DIR)
    final_class_counter: Counter = Counter()
    for summary in dataset_summaries:
        _copy_and_rewrite_dataset(summary, final_class_counter)
    _write_final_yaml()
    final_split_counts = _validate_combined_dataset()

    print("\nDataset preparation summary:")
    for summary in dataset_summaries:
        print(f"\nArchive: {summary.archive_name}")
        print(f"  Extracted to: {summary.extract_dir}")
        print(f"  Dataset root: {summary.dataset_root}")
        print("  Original classes:")
        for class_name in summary.original_names:
            print(f"    - {class_name}")
        print("  Mapping:")
        for line in _format_mapping(summary.original_names, summary.class_mapping):
            print(f"    - {line}")
        print("  Split counts:")
        for split in ("train", "valid", "test"):
            print(
                f"    - {split}: images={summary.split_image_counts.get(split, 0)}, "
                f"labels={summary.split_label_counts.get(split, 0)}"
            )
        print(f"  Copied images: {summary.copied_images}")
        print(f"  Rewritten annotations: {summary.annotations_rewritten}")
        print(f"  Dropped annotations: {summary.annotations_dropped}")
        print(f"  Polygon annotations converted to boxes: {summary.polygon_annotations_converted}")
        print(f"  Empty labels created: {summary.empty_labels_created}")
        print(f"  Missing source labels: {summary.missing_labels}")
        if summary.problematic_files:
            print("  Problematic files:")
            for item in summary.problematic_files[:10]:
                print(f"    - {item}")
            if len(summary.problematic_files) > 10:
                print(f"    - ... {len(summary.problematic_files) - 10} more")

    print("\nFinal combined dataset:")
    print(f"  Output root: {COMBINED_DIR}")
    print(f"  data.yaml: {COMBINED_DIR / 'data.yaml'}")
    for split in ("train", "valid", "test"):
        print(f"  - {split}: {final_split_counts.get(split, 0)} images")
    print(f"  - valve annotations: {final_class_counter.get(0, 0)}")
    print(f"  - gauge annotations: {final_class_counter.get(1, 0)}")


if __name__ == "__main__":
    main()
