from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from robustness_utils import (
    CLASS_NAMES,
    REPO_ROOT,
    ensure_dir,
    image_dir_for,
    label_dir_for_image_dir,
    list_images,
    make_yolo_yaml,
    repo_path,
    write_csv,
    write_json,
)


def _condition_name(path: Path) -> str:
    if path == REPO_ROOT / "data" / "detection" / "combined" / "test":
        return "original_test"
    parent = path.parent.name
    if parent in {"synthetic", "preprocessed"}:
        return path.name
    if path.name == "images":
        return path.parent.name
    return path.name


def _labels_complete(images: List[Path], label_dir: Optional[Path]) -> bool:
    if label_dir is None:
        return False
    return all((label_dir / f"{image.stem}.txt").exists() for image in images)


def _prediction_summary(model, image_dir: Path, condition: str) -> Dict:
    images = list_images(image_dir)
    if not images:
        return {
            "condition": condition,
            "mode": "prediction_summary",
            "images": 0,
            "detections": 0,
            "mean_confidence": 0.0,
            "no_detection_count": 0,
            "class_distribution": {},
        }

    results = model.predict(source=str(image_dir), verbose=False, save=False)
    confidences: List[float] = []
    class_counts: Dict[str, int] = {}
    no_detection = 0
    detections = 0

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            no_detection += 1
            continue
        detections += len(boxes)
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            cls_name = str(model.names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown")))
            confidences.append(conf)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return {
        "condition": condition,
        "mode": "prediction_summary",
        "images": len(images),
        "detections": int(detections),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "no_detection_count": int(no_detection),
        "class_distribution": class_counts,
    }


def _labeled_eval(
    model,
    condition_root: Path,
    condition: str,
    yaml_dir: Path,
    workers: int,
) -> Dict:
    yaml_path = make_yolo_yaml(condition_root, yaml_dir / f"{condition}.yaml")
    result = model.val(data=str(yaml_path), split="val", verbose=False, workers=workers)
    metrics = {
        "condition": condition,
        "mode": "labeled",
        "dataset_yaml": str(yaml_path.relative_to(REPO_ROOT)),
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
    }
    if hasattr(result, "box"):
        metrics.update(
            {
                "precision": float(result.box.mp),
                "recall": float(result.box.mr),
                "map50": float(result.box.map50),
                "map50_95": float(result.box.map),
            }
        )
    if hasattr(result, "results_dict"):
        metrics["results_dict"] = {
            str(key): float(value) for key, value in result.results_dict.items()
        }
    return metrics


def _discover_conditions() -> List[Path]:
    paths: List[Path] = []
    original = REPO_ROOT / "data" / "detection" / "combined" / "test"
    if (original / "images").exists():
        paths.append(original)

    for root_name in ("data/robustness/synthetic", "data/robustness/preprocessed"):
        root = repo_path(root_name)
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir():
                continue
            if (child / "images").exists():
                paths.append(child)
    return paths


def _flat_row(item: Dict) -> Dict:
    mode = "mixed"
    if item.get("labeled") and not item.get("prediction"):
        mode = "labeled"
    elif item.get("prediction") and not item.get("labeled"):
        mode = "prediction_summary"
    return {
        "condition": item.get("condition"),
        "mode": mode,
        "images": item.get("prediction", item).get("images", ""),
        "detections": item.get("prediction", item).get("detections", ""),
        "mean_confidence": item.get("prediction", item).get("mean_confidence", ""),
        "no_detection_count": item.get("prediction", item).get("no_detection_count", ""),
        "precision": item.get("labeled", item).get("precision", ""),
        "recall": item.get("labeled", item).get("recall", ""),
        "map50": item.get("labeled", item).get("map50", ""),
        "map50_95": item.get("labeled", item).get("map50_95", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detector behavior on original, corrupted, and preprocessed images."
    )
    parser.add_argument("--model", default="models/detector.pt", help="YOLO model path.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Condition folder. May contain images/ and labels/. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "labeled", "predict"],
        default="auto",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/robustness/robustness_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--out-csv",
        default="artifacts/robustness/robustness_summary.csv",
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="YOLO validation workers. Keep 0 on Windows to avoid DataLoader spawn issues.",
    )
    args = parser.parse_args()

    model_path = repo_path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Detector weights not found: {model_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is required for evaluation: {exc}") from exc

    model = YOLO(str(model_path))
    yaml_dir = ensure_dir("artifacts/robustness/datasets")
    conditions = [repo_path(item) for item in args.source] if args.source else _discover_conditions()
    if not conditions:
        raise SystemExit("No evaluation conditions found.")

    results = []
    for condition_root in conditions:
        image_dir = image_dir_for(condition_root)
        images = list_images(image_dir)
        if not images:
            continue
        label_dir = label_dir_for_image_dir(image_dir)
        condition = _condition_name(condition_root)
        complete_labels = _labels_complete(images, label_dir)

        item = {"condition": condition, "path": str(condition_root.relative_to(REPO_ROOT))}
        if args.mode in {"auto", "labeled"} and complete_labels:
            labeled = _labeled_eval(
                model,
                condition_root,
                condition,
                yaml_dir,
                workers=args.workers,
            )
            item["labeled"] = labeled
        elif args.mode == "labeled":
            item["labeled"] = {
                "condition": condition,
                "mode": "labeled",
                "error": "matching labels are unavailable",
            }

        if args.mode in {"auto", "predict"} or not complete_labels:
            item["prediction"] = _prediction_summary(model, image_dir, condition)

        results.append(item)
        label_note = "labels" if complete_labels else "no labels"
        print(f"Evaluated {condition}: {len(images)} images, {label_note}")

    payload = {
        "model": str(model_path.relative_to(REPO_ROOT)),
        "results": results,
    }
    json_path = write_json(args.out_json, payload)
    csv_rows = [_flat_row(item) for item in results]
    csv_path = write_csv(args.out_csv, csv_rows)
    print(f"Summary JSON: {json_path.relative_to(REPO_ROOT)}")
    print(f"Summary CSV: {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
