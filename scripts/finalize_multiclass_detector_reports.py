from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "detection_multiclass"
DATASET_ROOT = REPO_ROOT / "data" / "detection" / "industrial_multiclass"
RUN_ROOT = REPO_ROOT / "runs" / "detect" / "valvelens_multiclass_v1"
VAL_RUN = REPO_ROOT / "runs" / "detect" / "valvelens_multiclass_v1_val"
TEST_RUN = REPO_ROOT / "runs" / "detect" / "valvelens_multiclass_v1_test"
MODEL_PATH = REPO_ROOT / "models" / "detector_multiclass.pt"
BASELINE_PATH = REPO_ROOT / "models" / "detector.pt"


CLASS_NAMES = [
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


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("/", "\\")
    except Exception:
        return str(path).replace("/", "\\")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count())
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        env["torch_error"] = str(exc)
        env["cuda_available"] = False
    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        env["ultralytics_error"] = str(exc)
    return env


def result_artifacts(run_dir: Path) -> Dict[str, str]:
    names = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "results.csv",
        "results.png",
    ]
    return {name: rel(run_dir / name) for name in names if (run_dir / name).exists()}


def class_distribution_rows() -> List[Dict[str, Any]]:
    path = ARTIFACT_ROOT / "class_distribution.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dataset_counts_rows() -> List[Dict[str, Any]]:
    path = ARTIFACT_ROOT / "dataset_counts.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def per_class_rows(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    val_maps = evaluation.get("validation", {}).get("per_class_maps", [])
    test_maps = evaluation.get("test", {}).get("per_class_maps", [])
    for idx, name in enumerate(CLASS_NAMES):
        rows.append({
            "class": name,
            "validation_precision": "not exported",
            "validation_recall": "not exported",
            "validation_mAP50": "not exported",
            "validation_mAP50-95": val_maps[idx] if idx < len(val_maps) else "not found",
            "test_precision": "not exported",
            "test_recall": "not exported",
            "test_mAP50": "not exported",
            "test_mAP50-95": test_maps[idx] if idx < len(test_maps) else "not found",
        })
    return rows


def update_training_summary() -> Dict[str, Any]:
    path = ARTIFACT_ROOT / "training_summary.json"
    summary = read_json(path)
    summary.update({
        "command": (
            "python .\\scripts\\build_train_multiclass_detector.py --overwrite "
            "--max-ann-per-class 3000 --epochs 50 --imgsz 640 --batch -1 "
            "--device 0 --workers 0 --name valvelens_multiclass_v1"
        ),
        "environment": collect_environment(),
        "best_weights": rel(RUN_ROOT / "weights" / "best.pt"),
        "last_weights": rel(RUN_ROOT / "weights" / "last.pt"),
        "copied_weights": rel(MODEL_PATH),
        "baseline_detector_path": rel(BASELINE_PATH),
        "expanded_detector_path": rel(MODEL_PATH),
        "validation_command": (
            "yolo detect val model=runs\\detect\\valvelens_multiclass_v1\\weights\\best.pt "
            "data=data\\detection\\industrial_multiclass\\data.yaml split=val device=0"
        ),
        "test_command": (
            "yolo detect val model=runs\\detect\\valvelens_multiclass_v1\\weights\\best.pt "
            "data=data\\detection\\industrial_multiclass\\data.yaml split=test device=0"
        ),
        "prediction_command": (
            "yolo detect predict model=runs\\detect\\valvelens_multiclass_v1\\weights\\best.pt "
            "source=data\\detection\\industrial_multiclass\\test\\images save=True "
            "project=artifacts\\detection_multiclass name=predictions"
        ),
    })
    write_json(path, summary)
    return summary


def update_evaluation_summary() -> Dict[str, Any]:
    path = ARTIFACT_ROOT / "evaluation_summary.json"
    evaluation = read_json(path)
    evaluation["validation_artifacts"] = result_artifacts(VAL_RUN)
    evaluation["test_artifacts"] = result_artifacts(TEST_RUN)
    evaluation["per_class_table"] = per_class_rows(evaluation)
    evaluation["note"] = (
        "The saved Ultralytics summary contains aggregate precision/recall/mAP50 and "
        "per-class mAP50-95. Per-class precision, recall, and mAP50 were not exported "
        "in the stored JSON, so they are reported as not exported rather than inferred."
    )
    write_json(path, evaluation)
    return evaluation


def write_evaluation_md(evaluation: Dict[str, Any]) -> None:
    rows = []
    for split in ["validation", "test"]:
        metrics = evaluation.get(split, {})
        rows.append({
            "split": split,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("map50"),
            "mAP50-95": metrics.get("map50_95"),
        })
    lines = ["# Industrial Multiclass Detector Evaluation Summary", "", markdown_table(rows, ["split", "precision", "recall", "mAP50", "mAP50-95"])]
    lines.append("\n## Per-Class Metrics")
    lines.append(markdown_table(evaluation.get("per_class_table", []), [
        "class",
        "validation_precision",
        "validation_recall",
        "validation_mAP50",
        "validation_mAP50-95",
        "test_precision",
        "test_recall",
        "test_mAP50",
        "test_mAP50-95",
    ]))
    lines.append("\n## Evaluation Artifacts")
    lines.append("Validation artifacts:")
    for name, path in evaluation.get("validation_artifacts", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("\nTest artifacts:")
    for name, path in evaluation.get("test_artifacts", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append(f"\nPrediction visualizations: `{evaluation.get('prediction_dir')}`")
    lines.append("\nThis model is a second expanded detector. It does not replace the stable two-class detector at `models/detector.pt`.")
    lines.append("\nNote: per-class precision, recall, and mAP50 are marked `not exported` because they were not present in the saved Ultralytics result object. Per-class mAP50-95 is available.")
    write_text(ARTIFACT_ROOT / "evaluation_summary.md", "\n".join(lines) + "\n")


def write_prediction_observations(evaluation: Dict[str, Any]) -> None:
    per_class = evaluation.get("per_class_table", [])
    strong = []
    weak = []
    for row in per_class:
        value = row.get("test_mAP50-95")
        if isinstance(value, (int, float)) and value >= 0.45:
            strong.append(row["class"])
        if isinstance(value, (int, float)) and value < 0.05:
            weak.append(row["class"])
    prediction_count = len(list((ARTIFACT_ROOT / "predictions").glob("*"))) if (ARTIFACT_ROOT / "predictions").exists() else 0
    lines = ["# Prediction Observations", ""]
    lines.append(f"Prediction output folder: `{rel(ARTIFACT_ROOT / 'predictions')}`")
    lines.append(f"Saved prediction images/files: `{prediction_count}`")
    lines.append("\n## Good Examples")
    lines.append("The strongest test-set classes by per-class mAP50-95 are: " + (", ".join(f"`{item}`" for item in strong) if strong else "not found") + ".")
    lines.append("\n## Missed Detections and Weak Classes")
    lines.append("The weakest test-set classes by per-class mAP50-95 are: " + (", ".join(f"`{item}`" for item in weak) if weak else "not found") + ".")
    lines.append("\n## Likely Failure Modes")
    lines.extend([
        "- Low-sample classes such as `instrument_panel`, `vessel`, and `flange` have limited training evidence and weak per-class metrics.",
        "- Dense pipe annotations dominate the refinery dataset even after capping, which can make pipe behavior difficult to interpret.",
        "- The oil refinery source includes many rendered/macro-part annotations, so performance on real refinery images still requires external validation.",
        "- Small objects and cluttered equipment scenes are expected failure modes for the expanded model.",
        "- Low-light/glare failures were not isolated in this evaluation split; those remain covered by the separate robustness preprocessing experiment.",
    ])
    lines.append("\n## Class Confusions")
    lines.append("Confusion matrices were produced by the validation/test runs. Inspect:")
    lines.append(f"- `{rel(VAL_RUN / 'confusion_matrix.png')}`")
    lines.append(f"- `{rel(TEST_RUN / 'confusion_matrix.png')}`")
    write_text(ARTIFACT_ROOT / "prediction_observations.md", "\n".join(lines) + "\n")


def write_backend_readiness() -> None:
    detector_text = (REPO_ROOT / "backend" / "app" / "detector.py").read_text(encoding="utf-8", errors="replace")
    config_text = (REPO_ROOT / "backend" / "app" / "config.yaml").read_text(encoding="utf-8", errors="replace")
    assumes_two_class = "return \"unknown\"" in detector_text and "\"valve\"" in detector_text and "\"gauge\"" in detector_text
    lines = ["# Backend Integration Readiness: Expanded Multiclass Detector", ""]
    lines.append("No backend config was modified for this experiment.")
    lines.append("\n## Detector Paths")
    lines.append(f"- Stable baseline detector: `{rel(BASELINE_PATH)}`")
    lines.append(f"- Expanded detector: `{rel(MODEL_PATH)}`")
    lines.append("- Current configured detector path from `backend/app/config.yaml`: `models/detector.pt`" if "detector_model: models/detector.pt" in config_text else "- Current configured detector path: not parsed")
    lines.append("\n## Current Code Behavior")
    lines.append("`backend/app/detector.py` can technically load any Ultralytics YOLO model path, including `models/detector_multiclass.pt`, if configured.")
    lines.append("However, `_map_class()` currently maps model class names into a small semantic set: `valve`, `gauge`, `panel`, `tag`, or `unknown`.")
    lines.append(f"Code assumes narrow semantic mapping: `{assumes_two_class}`")
    lines.append("\n## What Must Change Before Runtime Integration")
    lines.extend([
        "- Add the expanded class names to detector semantic mapping instead of collapsing most classes to `unknown`.",
        "- Decide whether frontend colors/icons and side-panel labels should display all expanded classes.",
        "- Review downstream OCR/ReID logic: only some classes should trigger device identity OCR/ReID. For example, `valve`, `gauge`, `flange`, `instrument_panel`, and `vessel` may be device candidates; dense `pipe` regions may not need OCR.",
        "- Keep `models/detector.pt` as the default until the expanded model is intentionally selected in `backend/app/config.yaml`.",
        "- Add a config field or runtime mode for `detector_multiclass.pt` before switching demos.",
    ])
    lines.append("\n## Schema Compatibility")
    lines.append("The response schema stores detection class as a string (`DetectionInfo.cls`), so it can carry expanded class names. The main limitation is current detector class normalization, not the Pydantic schema.")
    lines.append("\n## Recommendation")
    lines.append("Backend is partially ready. The expanded model should remain an offline/thesis/future-product detector until class mapping, UI display, and OCR/ReID trigger rules are updated deliberately.")
    write_text(ARTIFACT_ROOT / "backend_integration_readiness.md", "\n".join(lines) + "\n")


def write_training_md(training: Dict[str, Any]) -> None:
    dataset = read_json(ARTIFACT_ROOT / "dataset_summary.json")
    rows = class_distribution_rows()
    env = training.get("environment", {})
    lines = ["# Industrial Multiclass Detector Training Summary", ""]
    lines.append("This experiment trains a second detector and keeps `models/detector.pt` unchanged as the stable valve/gauge baseline.")
    lines.append("\n## Dataset")
    lines.append(f"- Dataset root: `{dataset.get('dataset_root')}`")
    lines.append(f"- Data YAML: `{dataset.get('data_yaml')}`")
    lines.append(f"- Total images: `{dataset.get('total_images')}`")
    lines.append(f"- Total annotations: `{dataset.get('total_annotations')}`")
    lines.append(f"- Imbalance ratio: `{dataset.get('imbalance_ratio')}`")
    lines.append("\n## Class Distribution")
    lines.append(markdown_table(rows, ["class", "train_annotations", "valid_annotations", "test_annotations", "total_annotations", "role"]))
    lines.append("\n## Training Setup")
    for key in ["command", "model", "epochs", "imgsz", "batch", "device", "workers", "run_dir", "best_weights", "last_weights", "copied_weights"]:
        lines.append(f"- {key}: `{training.get(key)}`")
    lines.append("\n## Environment")
    for key in ["python", "torch", "ultralytics", "cuda_available", "cuda_device_count", "gpu_name"]:
        lines.append(f"- {key}: `{env.get(key)}`")
    lines.append("\nCPU note: CUDA was available for this run. If CUDA is unavailable, the same script can run on CPU, but training will be substantially slower.")
    write_text(ARTIFACT_ROOT / "training_summary.md", "\n".join(lines) + "\n")


def write_thesis_section(evaluation: Dict[str, Any], training: Dict[str, Any]) -> None:
    dataset = read_json(ARTIFACT_ROOT / "dataset_summary.json")
    split_rows = dataset_counts_rows()
    class_rows = class_distribution_rows()
    metric_rows = []
    for split in ["validation", "test"]:
        metrics = evaluation.get(split, {})
        metric_rows.append({
            "split": split,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("map50"),
            "mAP50-95": metrics.get("map50_95"),
        })
    lines = ["# Thesis Section: Expanded Industrial Multiclass Detector", ""]
    lines.append("A second YOLOv8 detector was trained to evaluate whether ValveLens can move beyond the original two-class valve/gauge detector toward broader oil/gas equipment recognition. The original baseline detector remains stored at `models/detector.pt`; the expanded detector is stored separately at `models/detector_multiclass.pt`.")
    lines.append("\n## Motivation")
    lines.append("The stable ValveLens detector is intentionally narrow and reliable for `valve` and `gauge`. For the thesis and future product direction, a broader detector was prepared to cover additional equipment categories such as flanges, vessels, pipes, desalters, heaters, heat exchangers, and tanks. This experiment demonstrates the feasibility and limitations of that expansion without changing the runtime backend configuration.")
    lines.append("\n## Dataset Sources and Class Normalization")
    lines.append("The final dataset was built only from locally available YOLO-format datasets. The selected sources were the existing combined valve/gauge dataset, the prepared Elementos Offshore dataset, and the prepared Oil Refinery dataset. Raw class names were normalized into a compact 10-class industrial label space. Ambiguous or unrelated classes were dropped, and very dense classes were capped per split to reduce imbalance.")
    lines.append("\nFinal classes: " + ", ".join(f"`{name}`" for name in CLASS_NAMES) + ".")
    lines.append("\n## Dataset Split Table")
    lines.append(markdown_table(split_rows, ["split", "images", "labels", "annotations"]))
    lines.append("\n## Class Distribution")
    lines.append(markdown_table(class_rows, ["class", "train_annotations", "valid_annotations", "test_annotations", "total_annotations", "role"]))
    lines.append("\n## Training Setup")
    lines.append(f"- Training command: `{training.get('command')}`")
    lines.append(f"- Seed weights: `{training.get('model')}`")
    lines.append(f"- Epochs: `{training.get('epochs')}`")
    lines.append(f"- Image size: `{training.get('imgsz')}`")
    lines.append(f"- Device: `{training.get('device')}`")
    lines.append(f"- CUDA available: `{training.get('environment', {}).get('cuda_available')}`")
    lines.append(f"- GPU: `{training.get('environment', {}).get('gpu_name')}`")
    lines.append(f"- Run directory: `{training.get('run_dir')}`")
    lines.append(f"- Best weights: `{training.get('best_weights')}`")
    lines.append(f"- Copied model: `{training.get('copied_weights')}`")
    lines.append("\n## Validation and Test Metrics")
    lines.append(markdown_table(metric_rows, ["split", "precision", "recall", "mAP50", "mAP50-95"]))
    lines.append("\n## Per-Class Discussion")
    lines.append("The strongest test classes by mAP50-95 were `tank`, `valve`, `desalter`, and `heater`. These classes have clearer visual structure or stronger representation in the prepared data. The weakest classes were `instrument_panel`, `vessel`, `flange`, and `pipe`. The low-sample offshore classes have limited examples, while the pipe class is dense and difficult because many annotations are small, repeated, and visually cluttered.")
    lines.append("\n## Failure Modes")
    lines.extend([
        "- Small-object failures are expected for flange and instrument-panel examples because the class counts are low.",
        "- Dense pipe scenes remain difficult because a single refinery image may contain many pipe annotations.",
        "- Some Oil Refinery data appears rendered or macro-part oriented, so real-image generalization must be validated separately.",
        "- Low-light and glare were not isolated in this detector experiment; those are handled by the separate robustness preprocessing experiment.",
    ])
    lines.append("\n## How This Extends the Original Detector")
    lines.append("The expanded detector shows how the ValveLens detection layer can grow from two classes to a broader oil/gas equipment vocabulary. It should be described as a second exploratory model rather than a replacement for the stable runtime detector. The backend can load the weights, but class mapping and UI semantics must be updated before using it as the default inference model.")
    lines.append("\n## Methodology / Results Wording")
    lines.append("Methodology: describe this as a multiclass dataset-normalization and YOLOv8 training experiment built from local oil/gas-related detection sources. Results: report the validation/test metrics, then discuss class imbalance and low-sample classes as the main limitations. Do not claim this model is deployed in the runtime pipeline yet.")
    write_text(ARTIFACT_ROOT / "thesis_detector_section.md", "\n".join(lines) + "\n")


def write_final_summary(evaluation: Dict[str, Any], training: Dict[str, Any]) -> None:
    dataset = read_json(ARTIFACT_ROOT / "dataset_summary.json")
    inventory = (ARTIFACT_ROOT / "dataset_inventory.md").read_text(encoding="utf-8", errors="replace") if (ARTIFACT_ROOT / "dataset_inventory.md").exists() else ""
    lines = ["# Final Multiclass Detector Summary", ""]
    lines.append("Datasets discovered: see `artifacts/detection_multiclass/dataset_inventory.md`.")
    lines.append("Datasets used: `data/detection/combined`, `data/detection/oilgas_expanded/elementos_offshore`, `data/detection/oilgas_expanded/oil_refinery`.")
    lines.append("Datasets excluded: raw source duplicates, empty staged folders, non-selected robustness/proxy identity folders, and any dataset without usable selected detection labels.")
    lines.append("Final class list: " + ", ".join(f"`{name}`" for name in CLASS_NAMES) + ".")
    lines.append(f"Total images: `{dataset.get('total_images')}`")
    lines.append(f"Total annotations: `{dataset.get('total_annotations')}`")
    lines.append(f"Training run path: `{training.get('run_dir')}`")
    lines.append(f"Best weights path: `{training.get('best_weights')}`")
    lines.append(f"Copied model path: `{training.get('copied_weights')}`")
    lines.append(f"Prediction output path: `{evaluation.get('prediction_dir')}`")
    lines.append("\nValidation metrics:")
    lines.append(f"- precision `{evaluation.get('validation', {}).get('precision')}`")
    lines.append(f"- recall `{evaluation.get('validation', {}).get('recall')}`")
    lines.append(f"- mAP50 `{evaluation.get('validation', {}).get('map50')}`")
    lines.append(f"- mAP50-95 `{evaluation.get('validation', {}).get('map50_95')}`")
    lines.append("\nTest metrics:")
    lines.append(f"- precision `{evaluation.get('test', {}).get('precision')}`")
    lines.append(f"- recall `{evaluation.get('test', {}).get('recall')}`")
    lines.append(f"- mAP50 `{evaluation.get('test', {}).get('map50')}`")
    lines.append(f"- mAP50-95 `{evaluation.get('test', {}).get('map50_95')}`")
    lines.append("\nRecommended thesis wording: describe this as a second exploratory expanded industrial detector that supports the future ValveLens product direction, while keeping the original valve/gauge detector as the stable baseline.")
    lines.append("Recommended backend integration step: add explicit expanded class mapping and OCR/ReID trigger rules before switching `backend/app/config.yaml` to `models/detector_multiclass.pt`.")
    write_text(ARTIFACT_ROOT / "final_summary.md", "\n".join(lines) + "\n")


def main() -> None:
    training = update_training_summary()
    evaluation = update_evaluation_summary()
    write_training_md(training)
    write_evaluation_md(evaluation)
    write_prediction_observations(evaluation)
    write_backend_readiness()
    write_thesis_section(evaluation, training)
    write_final_summary(evaluation, training)
    print("Multiclass detector reports finalized.")
    print(f"  thesis: {rel(ARTIFACT_ROOT / 'thesis_detector_section.md')}")
    print(f"  evaluation: {rel(ARTIFACT_ROOT / 'evaluation_summary.md')}")
    print(f"  class distribution: {rel(ARTIFACT_ROOT / 'class_distribution.csv')}")
    print(f"  backend readiness: {rel(ARTIFACT_ROOT / 'backend_integration_readiness.md')}")


if __name__ == "__main__":
    main()
