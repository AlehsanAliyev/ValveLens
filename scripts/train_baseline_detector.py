from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "detection" / "combined"
CANONICAL_YAML = DATASET_ROOT / "data.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "detection_training"
MODELS_DIR = REPO_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, validate, and predict with a YOLOv8 baseline detector.")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="valvelens_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--copy-best", action="store_true", help="Copy best.pt to models/detector.pt after training.")
    return parser.parse_args()


def require_ultralytics():
    try:
        from ultralytics import YOLO, settings  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is not available in the current environment. "
            "Install backend requirements or activate the project venv first."
        ) from exc
    settings.update({"wandb": False})
    return YOLO


def build_absolute_yaml() -> Path:
    if not CANONICAL_YAML.exists():
        raise SystemExit(f"Missing dataset YAML: {CANONICAL_YAML}")
    data = yaml.safe_load(CANONICAL_YAML.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid dataset YAML: {CANONICAL_YAML}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute_yaml = ARTIFACTS_DIR / "combined_ultralytics.yaml"
    absolute_data = {
        "path": DATASET_ROOT.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": {0: "valve", 1: "gauge"},
    }
    absolute_yaml.write_text(yaml.safe_dump(absolute_data, sort_keys=False), encoding="utf-8")
    return absolute_yaml


def normalize_metrics(result) -> dict:
    metrics = {
        "results_dict": {},
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "per_class_maps": [],
    }

    if hasattr(result, "results_dict"):
        metrics["results_dict"] = {str(k): float(v) for k, v in result.results_dict.items()}
    if hasattr(result, "box"):
        metrics["precision"] = float(result.box.mp)
        metrics["recall"] = float(result.box.mr)
        metrics["map50"] = float(result.box.map50)
        metrics["map50_95"] = float(result.box.map)
        metrics["per_class_maps"] = [float(v) for v in getattr(result.box, "maps", [])]
    return metrics


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    args = parse_args()
    YOLO = require_ultralytics()
    absolute_yaml = build_absolute_yaml()

    model = YOLO(args.model)
    train_results = model.train(
        data=str(absolute_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
    )

    save_dir = Path(train_results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise SystemExit(f"Training completed but best weights were not found: {best_weights}")

    trained_model = YOLO(str(best_weights))
    val_results = trained_model.val(data=str(absolute_yaml))
    predict_results = trained_model.predict(
        source=str(DATASET_ROOT / "test" / "images"),
        save=True,
        project=args.project,
        name=f"{args.name}_predict",
        exist_ok=True,
        device=args.device,
    )
    predict_dir = Path(predict_results[0].save_dir) if predict_results else Path(args.project) / f"{args.name}_predict"

    copied_detector = None
    if args.copy_best:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target = MODELS_DIR / "detector.pt"
        if target.exists():
            raise SystemExit(
                f"Refusing to overwrite existing detector artifact: {target}. "
                "Move or remove it and rerun with --copy-best."
            )
        shutil.copy2(best_weights, target)
        copied_detector = target

    summary = {
        "dataset_yaml": str(absolute_yaml),
        "train_save_dir": str(save_dir),
        "best_weights": str(best_weights),
        "predict_dir": str(predict_dir),
        "validation": normalize_metrics(val_results),
        "copied_detector": str(copied_detector) if copied_detector else None,
    }

    summary_path = ARTIFACTS_DIR / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
