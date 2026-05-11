from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "thesis"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt"}


IMPORTANT_FILES = {
    "backend/app/pipeline.py": "main inference orchestration",
    "backend/app/detector.py": "YOLO detector wrapper",
    "backend/app/zone_localizer.py": "FAISS-based zone/VPR retrieval",
    "backend/app/quality.py": "frame quality diagnostics",
    "backend/app/ocr.py": "OCR reading, tag parsing, and device ID extraction",
    "backend/app/reid.py": "device reference retrieval",
    "backend/app/fusion.py": "zone/OCR/ReID fusion scoring",
    "backend/app/policy.py": "uncertainty-aware decision policy",
    "backend/app/tracker.py": "temporal object tracking and smoothing",
    "backend/app/db.py": "SQLite schema and persistence helpers",
    "backend/app/faiss_store.py": "FAISS/numpy index persistence",
    "backend/app/evidence.py": "structured evidence helper for interactive assistant",
    "backend/app/routes/infer.py": "image/video/webcam inference routes",
    "backend/app/routes/debug.py": "debug/status route",
    "backend/app/routes/devices.py": "device inventory routes",
    "backend/app/routes/feedback.py": "human feedback route",
    "backend/app/routes/zones.py": "zone routes",
    "backend/app/routes/ask.py": "rule-based text question route",
    "frontend/src/pages/Live.jsx": "main live interaction page",
    "frontend/src/components/OverlayCanvas.jsx": "visual overlay for detections",
    "frontend/src/components/SidePanel.jsx": "evidence, decision, feedback, and ask UI",
    "scripts/train_baseline_detector.py": "YOLO training entrypoint",
    "scripts/evaluate_detector.py": "detector evaluation entrypoint",
    "scripts/prepare_combined_detection_dataset.py": "valve/gauge dataset preparation",
    "scripts/build_oilgas_proxy_inventory.py": "controlled proxy identity benchmark generator",
    "scripts/evaluate_preprocessing_detector.py": "robustness/preprocessing evaluation",
}


MODULE_STATUS = [
    ("Frame quality estimation", "blur/low-light diagnostics", "backend/app/quality.py", "Methodology"),
    ("Zone localization / VPR", "retrieve likely zone candidates from keyframe embeddings", "backend/app/zone_localizer.py", "Methodology"),
    ("FAISS zone index", "persistent zone embedding index", "backend/app/faiss_store.py; backend/data/faiss/zones_meta.json", "Methodology"),
    ("YOLOv8 detector", "valve/gauge object detection", "backend/app/detector.py; models/detector.pt", "Methodology/Results"),
    ("Detector training/evaluation scripts", "train and evaluate detector", "scripts/train_baseline_detector.py; scripts/evaluate_detector.py", "Results"),
    ("OCR", "read device tags and parse IDs", "backend/app/ocr.py; backend/app/cli/smoke_ocr.py", "Methodology/Results"),
    ("ReID / embedding retrieval", "retrieve enrolled device references", "backend/app/reid.py; backend/app/cli/smoke_reid.py", "Methodology/Results"),
    ("Device database", "device/refs/observations/feedback storage", "backend/app/db.py; backend/data/valvelens.db", "Methodology"),
    ("Device FAISS index", "device reference embedding search", "backend/app/faiss_store.py; backend/data/faiss/devices_meta.json", "Methodology/Results"),
    ("Tracking / temporal smoothing", "track detections across frames", "backend/app/tracker.py; backend/app/tests/test_tracker.py", "Methodology"),
    ("Feedback / human-in-the-loop", "confirm/reject/tap feedback storage", "backend/app/routes/feedback.py; backend/app/tests/test_feedback_helpers.py", "Methodology"),
    ("Decision policy", "ACCEPTED/UNCERTAIN decision logic", "backend/app/policy.py; backend/app/tests/test_policy.py", "Methodology"),
    ("Metrics export", "export/summarize observations", "backend/app/cli/export_metrics.py; backend/app/cli/summarize_metrics.py", "Results"),
    ("Frontend Live UI", "upload/video/webcam UI and evidence display", "frontend/src/pages/Live.jsx; frontend/src/components/SidePanel.jsx", "Implementation"),
    ("Preprocessing / robustness scripts", "classical restoration experiment", "scripts/preprocess_images.py; scripts/evaluate_preprocessing_detector.py", "Results"),
    ("Synthetic corruptions", "low-light/blur/noise/glare/contrast degradation", "scripts/generate_synthetic_corruptions.py; data/robustness/synthetic", "Results"),
    ("ExDARK support", "optional low-light dataset support", "scripts/setup_robustness_datasets.py; docs/ROBUSTNESS_PREPROCESSING.md", "Supporting experiment"),
    ("OpenLORIS zone ingestion", "zone keyframe import", "backend/app/cli/import_openloris_zones.py", "Methodology"),
    ("NYC-Indoor-VPR exploration", "optional VPR dataset inspection", "backend/app/cli/inspect_nyc_indoor_vpr.py", "Exploration"),
    ("COLD optional dataset tools", "optional zone dataset download", "backend/app/cli/download_cold_subset.py; scripts/download_cold_subset.ps1", "Exploration"),
    ("SAM/SAM2 segmentation", "segmentation placeholder/optional component", "backend/app/segmenter.py", "Future work"),
    ("VLM/open-vocabulary components", "planned evidence-aware assistant, no VLM integration", "docs/INTERACTIVE_ASSISTANT_PLAN.md; backend/app/routes/ask.py", "Future work"),
]


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def status_for(path: str, validation_hint: Optional[bool] = None) -> str:
    if not file_exists(path.split(";")[0].strip()):
        return "Not found"
    if validation_hint is True:
        return "Implemented"
    if validation_hint is False:
        return "Implemented but not fully validated"
    return "Implemented"


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not path.exists():
        return data
    current_key: Optional[str] = None
    for raw in read_text(path).splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if re.match(r"^\s+\d+\s*:", line) and current_key:
            key, value = line.split(":", 1)
            data.setdefault(current_key, {})[int(key.strip())] = value.strip().strip("'\"")
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                if value.lower() in {"true", "false"}:
                    data[key] = value.lower() == "true"
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value.strip("'\"")
                current_key = None
            else:
                data[key] = {}
                current_key = key
    return data


def parse_class_names(dataset_root: Path) -> Dict[int, str]:
    yaml_path = dataset_root / "data.yaml"
    data = parse_simple_yaml(yaml_path)
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def list_files(root: Path, exts: Optional[set[str]] = None) -> List[Path]:
    if not root.exists():
        return []
    items = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if exts and path.suffix.lower() not in exts:
            continue
        items.append(path)
    return items


def image_count(path: Path) -> int:
    return len(list_files(path, IMAGE_EXTS))


def label_count(path: Path) -> int:
    return len(list_files(path, LABEL_EXTS))


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def human_size(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def yolo_dataset_summary(path: Path) -> Dict[str, Any]:
    classes = parse_class_names(path)
    split_data: Dict[str, Any] = {}
    total_labels = 0
    total_images = 0
    class_counts: Counter[int] = Counter()
    malformed = 0
    invalid_class_ids = 0
    empty_labels = 0
    for split in ["train", "valid", "val", "test"]:
        split_root = path / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.exists() and not labels_dir.exists():
            continue
        images = list_files(images_dir, IMAGE_EXTS)
        labels = list_files(labels_dir, LABEL_EXTS)
        split_counts: Counter[int] = Counter()
        split_malformed = 0
        split_invalid = 0
        split_empty = 0
        for label in labels:
            lines = [line.strip() for line in read_text(label).splitlines() if line.strip()]
            if not lines:
                split_empty += 1
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    split_malformed += 1
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    split_malformed += 1
                    continue
                split_counts[cls_id] += 1
                if classes and cls_id not in classes:
                    split_invalid += 1
        total_images += len(images)
        total_labels += len(labels)
        class_counts.update(split_counts)
        malformed += split_malformed
        invalid_class_ids += split_invalid
        empty_labels += split_empty
        split_data[split] = {
            "images": len(images),
            "labels": len(labels),
            "annotations": int(sum(split_counts.values())),
            "class_distribution": {classes.get(k, str(k)): int(v) for k, v in sorted(split_counts.items())},
            "empty_label_files": split_empty,
            "malformed_labels": split_malformed,
            "invalid_class_ids": split_invalid,
        }
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size": human_size(folder_size(path)),
        "classes": {str(k): v for k, v in classes.items()},
        "splits": split_data,
        "total_images": total_images,
        "total_label_files": total_labels,
        "total_annotations": int(sum(class_counts.values())),
        "class_distribution": {classes.get(k, str(k)): int(v) for k, v in sorted(class_counts.items())},
        "empty_label_files": empty_labels,
        "malformed_labels": malformed,
        "invalid_class_ids": invalid_class_ids,
    }


def collect_repo_structure() -> Dict[str, Any]:
    top = []
    for item in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if item.name in {".git", ".venv", "__pycache__"}:
            continue
        top.append({"path": rel(item), "type": "directory" if item.is_dir() else "file"})
    important = []
    for path, role in IMPORTANT_FILES.items():
        full = REPO_ROOT / path
        important.append({
            "path": path,
            "exists": full.exists(),
            "role": role,
            "size_bytes": full.stat().st_size if full.exists() and full.is_file() else None,
        })
    return {"top_level": top, "important_files": important}


def collect_config() -> Dict[str, Any]:
    return parse_simple_yaml(REPO_ROOT / "backend" / "app" / "config.yaml")


def collect_routes() -> Dict[str, Any]:
    routes: Dict[str, Any] = {}
    routes_dir = REPO_ROOT / "backend" / "app" / "routes"
    for path in sorted(routes_dir.glob("*.py")):
        text = read_text(path)
        endpoints = []
        for match in re.finditer(r"@router\.(get|post|put|delete)\(([^)]*)\)", text):
            endpoints.append({"method": match.group(1).upper(), "decorator": match.group(0)})
        routes[rel(path)] = endpoints
    return routes


def collect_frontend() -> Dict[str, Any]:
    root = REPO_ROOT / "frontend" / "src"
    files = []
    for path in sorted(list_files(root), key=lambda p: rel(p)):
        if path.suffix.lower() not in {".jsx", ".js", ".css"}:
            continue
        text = read_text(path)
        files.append({
            "path": rel(path),
            "lines": text.count("\n") + 1 if text else 0,
            "mentions_feedback": "feedback" in text.lower(),
            "mentions_ask": "ask" in text.lower(),
            "mentions_webcam": "webcam" in text.lower() or "camera" in text.lower(),
        })
    return {"files": files}


def collect_database_status() -> Dict[str, Any]:
    db_path = REPO_ROOT / "backend" / "data" / "valvelens.db"
    status: Dict[str, Any] = {"path": rel(db_path), "exists": db_path.exists()}
    if not db_path.exists():
        return status
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    counts = {}
    for table in tables:
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
        except Exception:
            counts[table] = "not verified"
    status["tables"] = tables
    status["counts"] = counts
    if "feedback" in tables:
        rows = conn.execute("SELECT feedback_type, COUNT(*) AS c FROM feedback GROUP BY feedback_type").fetchall()
        status["feedback_types"] = {row["feedback_type"] or "unknown": int(row["c"]) for row in rows}
    if "observations" in tables:
        accepted = conn.execute(
            "SELECT COUNT(*) AS c FROM observations WHERE final_device_id IS NOT NULL AND final_device_id != ''"
        ).fetchone()["c"]
        deferred = conn.execute(
            "SELECT COUNT(*) AS c FROM observations WHERE policy_action IS NOT NULL AND policy_action != ''"
        ).fetchone()["c"]
        status["accepted_observations_by_final_device"] = int(accepted)
        status["observations_with_policy_action"] = int(deferred)
    conn.close()
    return status


def collect_faiss_status() -> Dict[str, Any]:
    root = REPO_ROOT / "backend" / "data" / "faiss"
    status = {"path": rel(root), "exists": root.exists(), "indexes": {}}
    for name in ["zones", "devices"]:
        meta = root / f"{name}_meta.json"
        npy = root / f"{name}.npy"
        index = root / f"{name}.index"
        item: Dict[str, Any] = {
            "meta_path": rel(meta),
            "meta_exists": meta.exists(),
            "npy_path": rel(npy),
            "npy_exists": npy.exists(),
            "index_path": rel(index),
            "index_exists": index.exists(),
            "index_size": human_size(index.stat().st_size) if index.exists() else "not found",
        }
        if meta.exists():
            try:
                payload = json.loads(read_text(meta))
                items = payload.get("items", payload if isinstance(payload, list) else [])
                item["meta_items"] = len(items)
                item["built_at"] = payload.get("built_at") if isinstance(payload, dict) else None
            except Exception as exc:
                item["meta_error"] = str(exc)
        if npy.exists():
            try:
                import numpy as np

                arr = np.load(npy)
                item["npy_shape"] = list(arr.shape)
            except Exception as exc:
                item["npy_shape"] = f"not verified: {exc}"
        status["indexes"][name] = item
    return status


def collect_dataset_inventory() -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    known = {
        "combined_valve_gauge": ("data/detection/combined", "Main experiment dataset", "YOLO valve/gauge detector training/evaluation"),
        "source_valve_detection_v1i": ("data_sources/extracted/valve_detection_v1i_yolov8", "Source dataset", "Roboflow valve source normalized into the combined detector dataset"),
        "source_valve_detection_v6i": ("data_sources/extracted/valve_detection_v6i_yolov8", "Source dataset", "Roboflow valve/gauge source normalized into the combined detector dataset"),
        "nyc_indoor_vpr": ("data_sources/extracted/indoor_anony", "Explored/supporting dataset", "public indoor visual-place proxy for zone localization research"),
        "openloris_corridor": ("data_sources/extracted/corridor", "Supporting experiment dataset", "OpenLORIS-style corridor zones for VPR/zone retrieval"),
        "openloris_office": ("data_sources/extracted/office", "Supporting experiment dataset", "OpenLORIS-style office zones for VPR/zone retrieval"),
        "openloris_station": ("data_sources/extracted/station", "Supporting experiment dataset", "OpenLORIS-style station zones for VPR/zone retrieval"),
        "exdark": ("data_sources/extracted/exdark", "Supporting/explored dataset", "low-light image source for qualitative robustness/preprocessing discussion"),
        "hydraulic_components": ("data/detection/expanded_industrial/hydraulic_components", "Explored but not main", "earlier expanded industrial detection candidate, superseded by focused oil/gas staging"),
        "oil_refinery_expanded": ("data/detection/oilgas_expanded/oil_refinery", "Supporting experiment dataset", "expanded oil/gas detector staging"),
        "elementos_offshore_expanded": ("data/detection/oilgas_expanded/elementos_offshore", "Supporting experiment dataset", "expanded offshore detector staging"),
        "industrial_multilabel_expanded": ("data/detection/oilgas_expanded/industrial_multilabel", "Explored but not used", "optional expanded detector staging"),
        "wellhead_valve_gauge": ("data/detection/oilgas_expanded/wellhead_valve_gauge", "Explored but not used", "wellhead candidate dataset staging"),
        "device_benchmark": ("data/device_benchmark", "Supporting experiment dataset", "proxy identity benchmark"),
        "robustness_synthetic": ("data/robustness/synthetic", "Supporting experiment dataset", "synthetic corruptions for robustness"),
        "robustness_preprocessed": ("data/robustness/preprocessed", "Supporting experiment dataset", "restored/preprocessed robustness images"),
        "openloris_extracted": ("data_sources/extracted", "Supporting/explored dataset", "zone/VPR and optional low-light data sources"),
        "roboflow_downloads": ("data_sources/downloads/roboflow", "Exploration/staging", "downloaded Roboflow sources"),
    }
    for name, (path_str, role, thesis_role) in known.items():
        path = REPO_ROOT / path_str
        if (path / "data.yaml").exists():
            details = yolo_dataset_summary(path)
        else:
            details = {
                "path": rel(path),
                "exists": path.exists(),
                "size": human_size(folder_size(path)),
                "total_images": image_count(path),
                "total_label_files": label_count(path),
            }
        details["role"] = role
        details["thesis_role"] = thesis_role
        datasets[name] = details
    return datasets


def collect_development_timeline() -> List[Dict[str, Any]]:
    return [
        {
            "stage": "Problem framing and architecture",
            "role": "Defined ValveLens as an uncertainty-aware industrial vision assistant rather than a single detector.",
            "evidence": ["README.md", "docs/AGENT_CONTEXT_PACK.md", "docs/PROJECT_STATUS.md"],
            "thesis_use": "Use in introduction/methodology to explain why localization, detection, identity, and feedback are treated as separate evidence sources.",
        },
        {
            "stage": "Zone localization research",
            "role": "Used public indoor-place data as a proxy for plant zones because no dedicated refinery zone dataset was available.",
            "evidence": [
                "data_sources/extracted/indoor_anony",
                "data_sources/extracted/corridor",
                "data_sources/extracted/office",
                "data_sources/extracted/station",
                "backend/app/cli/import_openloris_zones.py",
                "backend/app/cli/import_zones_from_datasets.py",
            ],
            "thesis_use": "Report as supporting VPR/zone-localization methodology, not as final oil-facility deployment evidence.",
        },
        {
            "stage": "Valve/gauge detection dataset construction",
            "role": "Merged two Roboflow YOLO datasets into a two-class valve/gauge detector dataset.",
            "evidence": [
                "data_sources/extracted/valve_detection_v1i_yolov8",
                "data_sources/extracted/valve_detection_v6i_yolov8",
                "scripts/prepare_combined_detection_dataset.py",
                "data/detection/combined/data.yaml",
            ],
            "thesis_use": "Main object-detection experiment dataset.",
        },
        {
            "stage": "YOLOv8 detector training and integration",
            "role": "Trained and evaluated a YOLOv8 detector, then copied the selected weights into the backend runtime path.",
            "evidence": [
                "scripts/train_baseline_detector.py",
                "scripts/evaluate_detector.py",
                "artifacts/detection_training/valvelens_v1_cuda_summary.json",
                "artifacts/detection_training/valvelens_v1_cuda_testval_test_summary.json",
                "models/detector.pt",
            ],
            "thesis_use": "Main quantitative results for detection.",
        },
        {
            "stage": "Runtime evidence pipeline",
            "role": "Connected quality checks, zone candidates, detector ROIs, OCR, ReID, fusion, policy, tracking, and observation storage.",
            "evidence": [
                "backend/app/pipeline.py",
                "backend/app/quality.py",
                "backend/app/ocr.py",
                "backend/app/reid.py",
                "backend/app/fusion.py",
                "backend/app/policy.py",
                "backend/app/tracker.py",
                "backend/app/db.py",
            ],
            "thesis_use": "Core Methodology chapter material.",
        },
        {
            "stage": "Frontend and human feedback",
            "role": "Built a UI that displays detections/evidence and records human feedback instead of hiding uncertainty.",
            "evidence": [
                "frontend/src/pages/Live.jsx",
                "frontend/src/components/OverlayCanvas.jsx",
                "frontend/src/components/SidePanel.jsx",
                "backend/app/routes/feedback.py",
            ],
            "thesis_use": "Implementation and human-in-the-loop discussion.",
        },
        {
            "stage": "Robustness and restoration experiments",
            "role": "Generated low-light, blur, noise, glare, and contrast corruptions, then evaluated classical preprocessing as partial restoration.",
            "evidence": [
                "scripts/generate_synthetic_corruptions.py",
                "scripts/preprocess_images.py",
                "scripts/evaluate_preprocessing_detector.py",
                "scripts/select_best_preprocessing.py",
                "docs/ROBUSTNESS_PREPROCESSING.md",
                "data_sources/extracted/exdark",
                "artifacts/robustness/robustness_summary.json",
            ],
            "thesis_use": "Supporting robustness result; keep separate from runtime inference claims.",
        },
        {
            "stage": "Device identity benchmark",
            "role": "Addressed missing real device references by creating controlled proxy reference/query benchmarks with synthetic tags.",
            "evidence": [
                "docs/DEVICE_IDENTITY_BENCHMARK.md",
                "scripts/build_proxy_device_benchmark.py",
                "scripts/build_oilgas_proxy_inventory.py",
                "backend/app/cli/enroll_devices_from_manifest.py",
                "backend/app/cli/validate_identity_benchmark.py",
                "artifacts/identity_benchmark/identity_benchmark_summary.json",
            ],
            "thesis_use": "Use as mechanical validation of OCR/ReID/fusion plumbing; do not present as real industrial identity accuracy.",
        },
        {
            "stage": "Focused oil/gas dataset expansion",
            "role": "Staged refinery/offshore/wellhead/industrial datasets for a future expanded detector while keeping the active model unchanged.",
            "evidence": [
                "data_sources/DATASET_REGISTRY.md",
                "scripts/prepare_oilgas_expanded_dataset.py",
                "data/detection/oilgas_expanded/oil_refinery",
                "data/detection/oilgas_expanded/elementos_offshore",
            ],
            "thesis_use": "Mention as dataset expansion and future-work preparation, not as current runtime model performance.",
        },
        {
            "stage": "Interactive assistant planning",
            "role": "Designed a future text-question layer around structured ValveLens evidence instead of blind VLM image guessing.",
            "evidence": [
                "docs/INTERACTIVE_ASSISTANT_PLAN.md",
                "backend/app/evidence.py",
                "backend/app/routes/ask.py",
            ],
            "thesis_use": "Use as planned extension if files exist; state that no VLM is integrated yet.",
        },
    ]


def dataset_history_rows(datasets: Dict[str, Any]) -> List[Dict[str, Any]]:
    order = [
        ("NYC-Indoor-VPR", "nyc_indoor_vpr", "Zone/VPR proxy exploration", "Explored/supporting", "Used to test indoor place-recognition style zone retrieval when refinery zone data was unavailable."),
        ("OpenLORIS corridor", "openloris_corridor", "Zone/VPR keyframes", "Supporting", "Used as public proxy zones for spatial memory and FAISS retrieval."),
        ("OpenLORIS office", "openloris_office", "Zone/VPR keyframes", "Supporting", "Used as public proxy zones for spatial memory and FAISS retrieval."),
        ("OpenLORIS station", "openloris_station", "Zone/VPR keyframes", "Supporting", "Used as public proxy zones for spatial memory and FAISS retrieval."),
        ("Valve Detection v1i", "source_valve_detection_v1i", "Detector source", "Main source", "Raw valve classes collapsed into the project valve class."),
        ("Valve Detection v6i", "source_valve_detection_v6i", "Detector source", "Main source", "Raw gauge/valve classes collapsed into valve and gauge."),
        ("Combined valve/gauge", "combined_valve_gauge", "Detector training/evaluation", "Main experiment", "Final two-class YOLO dataset used for detector training and reporting."),
        ("ExDARK", "exdark", "Low-light robustness", "Supporting/explored", "Used for qualitative low-light discussion; not valve/gauge metric ground truth."),
        ("Synthetic corruptions", "robustness_synthetic", "Detector robustness", "Supporting experiment", "Generated from the test split to compare clean, degraded, and restored detector behavior."),
        ("Hydraulic Components", "hydraulic_components", "Expanded industrial detection", "Explored but not main", "Earlier industrial expansion candidate, later replaced by focused oil/gas datasets."),
        ("Oil Refinery", "oil_refinery_expanded", "Oil/gas expanded detection", "Staged/supporting", "Refinery-specific macro-equipment dataset staged for inspection."),
        ("Elementos Offshore", "elementos_offshore_expanded", "Oil/gas expanded detection", "Staged/supporting", "Small offshore dataset staged for visually relevant classes."),
        ("Wellhead candidate", "wellhead_valve_gauge", "Oil/gas expanded detection", "Planned/explored", "Useful target classes, but exact source/license still require verification if not present locally."),
        ("industrial-multilabel", "industrial_multilabel_expanded", "Optional expanded detection", "Optional/explored", "Broad industrial classes; only selected classes should be used after inspection."),
        ("Proxy device benchmark", "device_benchmark", "Identity validation", "Supporting experiment", "Controlled proxy inventory for OCR/ReID mechanics; not real industrial identity validation."),
    ]
    rows = []
    for label, key, role, status, note in order:
        item = datasets.get(key, {})
        rows.append({
            "Dataset": label,
            "Path": item.get("path", "not found"),
            "Exists": item.get("exists", False),
            "Images": item.get("total_images", "not counted"),
            "Labels": item.get("total_label_files", "not counted"),
            "Role": role,
            "Status": status,
            "Thesis wording": note,
        })
    return rows


def collect_detector_results() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "integration_weights": {
            "path": "models/detector.pt",
            "exists": (REPO_ROOT / "models" / "detector.pt").exists(),
            "size": human_size((REPO_ROOT / "models" / "detector.pt").stat().st_size)
            if (REPO_ROOT / "models" / "detector.pt").exists()
            else "not found",
        },
        "summaries": {},
    }
    for path in [
        REPO_ROOT / "artifacts" / "detection_training" / "valvelens_v1_cuda_summary.json",
        REPO_ROOT / "artifacts" / "detection_training" / "valvelens_v1_cuda_testval_test_summary.json",
    ]:
        if path.exists():
            try:
                result["summaries"][rel(path)] = json.loads(read_text(path))
            except Exception as exc:
                result["summaries"][rel(path)] = {"error": str(exc)}
    runs = []
    runs_root = REPO_ROOT / "runs" / "detect"
    if runs_root.exists():
        for run in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            entry = {"path": rel(run), "has_results_csv": (run / "results.csv").exists(), "has_best": (run / "weights" / "best.pt").exists()}
            if (run / "args.yaml").exists():
                entry["args_yaml"] = rel(run / "args.yaml")
            runs.append(entry)
    result["runs_detect"] = runs[:40]
    return result


def collect_robustness() -> Dict[str, Any]:
    root = REPO_ROOT / "artifacts" / "robustness"
    out: Dict[str, Any] = {"summary_json": rel(root / "robustness_summary.json"), "exists": (root / "robustness_summary.json").exists()}
    if (root / "robustness_summary.json").exists():
        try:
            out["summary"] = json.loads(read_text(root / "robustness_summary.json"))
        except Exception as exc:
            out["summary_error"] = str(exc)
    if (root / "synthetic_summary.json").exists():
        try:
            out["synthetic_summary"] = json.loads(read_text(root / "synthetic_summary.json"))
        except Exception as exc:
            out["synthetic_summary_error"] = str(exc)
    out["scripts"] = [rel(p) for p in [
        REPO_ROOT / "scripts" / "setup_robustness_datasets.py",
        REPO_ROOT / "scripts" / "generate_synthetic_corruptions.py",
        REPO_ROOT / "scripts" / "preprocess_images.py",
        REPO_ROOT / "scripts" / "evaluate_preprocessing_detector.py",
        REPO_ROOT / "scripts" / "preview_preprocessing_examples.py",
    ] if p.exists()]
    return out


def collect_identity() -> Dict[str, Any]:
    path = REPO_ROOT / "artifacts" / "identity_benchmark" / "identity_benchmark_summary.json"
    out: Dict[str, Any] = {"summary_json": rel(path), "exists": path.exists()}
    if path.exists():
        try:
            payload = json.loads(read_text(path))
            out.update(payload.get("summary", {}))
        except Exception as exc:
            out["error"] = str(exc)
    meta = REPO_ROOT / "data" / "device_benchmark" / "oilgas_proxy_inventory_metadata.json"
    if meta.exists():
        try:
            out["proxy_inventory_metadata"] = json.loads(read_text(meta))
        except Exception as exc:
            out["proxy_inventory_error"] = str(exc)
    return out


def collect_oilgas_reports() -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    root = REPO_ROOT / "artifacts" / "detection_oilgas_expanded"
    for report in sorted(root.glob("*/prep_report.json")):
        try:
            reports[rel(report)] = json.loads(read_text(report))
        except Exception as exc:
            reports[rel(report)] = {"error": str(exc)}
    return reports


def run_pytest() -> Dict[str, Any]:
    output_path = OUT_DIR / "pytest_output.txt"
    cmd = [sys.executable, "-m", "pytest", "app\\tests"]
    def write_pytest_output(text: str) -> Path:
        try:
            write_text(output_path, text)
            return output_path
        except PermissionError:
            fallback = OUT_DIR / f"pytest_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            write_text(fallback, text)
            return fallback

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT / "backend",
            text=True,
            capture_output=True,
            timeout=240,
        )
        text = proc.stdout + proc.stderr
        written_path = write_pytest_output(text)
        match = re.search(r"=+\s*(\d+) passed.*?in ([0-9.]+)s\s*=+", text)
        failed = re.search(r"=+\s*(\d+) failed, (\d+) passed.*?in ([0-9.]+)s\s*=+", text)
        return {
            "command": "cd backend; pytest app\\tests",
            "returncode": proc.returncode,
            "output": rel(written_path),
            "passed": int(match.group(1)) if match else (int(failed.group(2)) if failed else "not parsed"),
            "failed": int(failed.group(1)) if failed else (0 if proc.returncode == 0 else "not parsed"),
            "status": "passed" if proc.returncode == 0 else "failed",
        }
    except Exception as exc:
        written_path = write_pytest_output(str(exc))
        return {"command": "cd backend; pytest app\\tests", "status": "not run", "error": str(exc), "output": rel(written_path)}


def run_zone_smoke(database: Dict[str, Any]) -> Dict[str, Any]:
    out_json = OUT_DIR / "zone_smoke_results.json"
    out_md = OUT_DIR / "zone_smoke_results.md"
    db_path = REPO_ROOT / "backend" / "data" / "valvelens.db"
    results: List[Dict[str, Any]] = []
    if not db_path.exists() or int(database.get("counts", {}).get("zone_keyframes", 0) or 0) == 0:
        payload = {"status": "not run", "reason": "no zone keyframes found in database"}
        write_json(out_json, payload)
        write_text(out_md, "# Zone Smoke Results\n\nNot run: no zone keyframes found in database.\n")
        return payload
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT zone_id, image_path FROM zone_keyframes LIMIT 5").fetchall()
    conn.close()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "backend")
    for row in rows:
        img = Path(row["image_path"])
        if not img.is_absolute():
            img = (REPO_ROOT / img).resolve()
        item = {"query_path": rel(img), "expected_zone": row["zone_id"], "exists": img.exists()}
        if not img.exists():
            item["status"] = "missing image"
            results.append(item)
            continue
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "app.cli.smoke_zones_aggregate", "--image", str(img), "--topk", "5"],
                cwd=REPO_ROOT / "backend",
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            item["returncode"] = proc.returncode
            item["output"] = proc.stdout.strip()
            item["stderr"] = proc.stderr.strip()
            item["top1_correct"] = row["zone_id"] in proc.stdout.splitlines()[-1] if proc.stdout.strip() else False
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        results.append(item)
    payload = {"status": "completed", "results": results}
    write_json(out_json, payload)
    lines = ["# Zone Smoke Results", ""]
    for item in results:
        lines.append(f"- Query: `{item.get('query_path')}`")
        lines.append(f"  Expected zone: `{item.get('expected_zone')}`")
        lines.append(f"  Exists: `{item.get('exists')}`")
        lines.append(f"  Top-1 correct (parsed): `{item.get('top1_correct', 'not verified')}`")
        if item.get("output"):
            lines.append("  Output:")
            lines.append("  ```text")
            lines.extend("  " + line for line in str(item["output"]).splitlines())
            lines.append("  ```")
    write_text(out_md, "\n".join(lines) + "\n")
    return payload


def implementation_status(identity: Dict[str, Any], db_status: Dict[str, Any], datasets: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for module, purpose, evidence, category in MODULE_STATUS:
        first_path = evidence.split(";")[0].strip()
        status = "Implemented" if file_exists(first_path) else "Not found"
        if module in {"ReID / embedding retrieval", "Device FAISS index", "OCR"}:
            status = "Implemented but not fully validated"
            if module == "OCR" and identity.get("ocr_backend_available") and identity.get("ocr_exact_matches", 0):
                status = "Implemented and benchmarked on proxy data"
            if module == "ReID / embedding retrieval" and identity.get("reid_top1_accuracy") is not None:
                status = "Implemented and benchmarked on proxy data"
            if module == "Device FAISS index" and identity.get("device_faiss_size", 0):
                status = "Implemented"
        if module == "SAM/SAM2 segmentation":
            status = "Partially implemented" if file_exists(first_path) else "Not found"
        if module == "VLM/open-vocabulary components":
            status = "Planned / future work"
        rows.append({
            "Module": module,
            "Purpose": purpose,
            "Current status": status,
            "Evidence": evidence,
            "Thesis category": category,
        })
    return rows


def markdown_table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return "No rows found.\n"
    columns = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            vals.append(str(value).replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_repo_structure(summary: Dict[str, Any]) -> None:
    lines = ["# Repository Structure Summary", ""]
    lines.append("## Top-Level Layout")
    for item in summary["top_level"]:
        lines.append(f"- `{item['path']}` — {item['type']}")
    lines.append("\n## Thesis-Relevant Files")
    for item in summary["important_files"]:
        exists = "exists" if item["exists"] else "not found"
        lines.append(f"- `{item['path']}` — {item['role']} ({exists})")
    write_text(OUT_DIR / "repo_structure_summary.md", "\n".join(lines) + "\n")


def write_dataset_inventory(datasets: Dict[str, Any], oilgas: Dict[str, Any]) -> None:
    lines = ["# Dataset Inventory", ""]
    table = []
    for name, item in datasets.items():
        classes = item.get("classes") or {}
        table.append({
            "Name": name,
            "Path": item.get("path"),
            "Role": item.get("role"),
            "Images": item.get("total_images", "not counted"),
            "Labels": item.get("total_label_files", "not counted"),
            "Classes": ", ".join(classes.values()) if isinstance(classes, dict) else "not found",
            "Thesis use": item.get("thesis_role"),
        })
    lines.append(markdown_table(table, ["Name", "Path", "Role", "Images", "Labels", "Classes", "Thesis use"]))
    lines.append("\n## Oil/Gas Expanded Preparation Reports")
    if oilgas:
        for path, report in oilgas.items():
            lines.append(f"- `{path}`")
            if isinstance(report, dict):
                mapped = report.get("mapped_classes") or report.get("target_classes") or report.get("class_counts") or {}
                lines.append(f"  - status: `{report.get('status', 'not specified')}`")
                lines.append(f"  - source images: `{report.get('source_images', report.get('images', 'not found'))}`")
                lines.append(f"  - mapped/classes/counts: `{mapped}`")
    else:
        lines.append("No oil/gas preparation reports found.")
    write_text(OUT_DIR / "dataset_inventory.md", "\n".join(lines) + "\n")
    with (OUT_DIR / "dataset_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "path", "role", "images", "labels", "annotations", "classes"])
        writer.writeheader()
        for name, item in datasets.items():
            classes = item.get("classes") or {}
            writer.writerow({
                "name": name,
                "path": item.get("path"),
                "role": item.get("role"),
                "images": item.get("total_images", ""),
                "labels": item.get("total_label_files", ""),
                "annotations": item.get("total_annotations", ""),
                "classes": ";".join(classes.values()) if isinstance(classes, dict) else "",
            })


def write_development_timeline(timeline: List[Dict[str, Any]], datasets: Dict[str, Any]) -> None:
    lines = ["# ValveLens Development Timeline", ""]
    lines.append("This file turns the repository history into thesis-safe evidence. It explains why several datasets and experiments appear in the project even when they are not all part of the final detector result.")
    for index, item in enumerate(timeline, start=1):
        lines.append(f"\n## {index}. {item['stage']}")
        lines.append(item["role"])
        lines.append("")
        lines.append(f"Thesis use: {item['thesis_use']}")
        lines.append("")
        lines.append("Evidence:")
        for evidence in item["evidence"]:
            lines.append(f"- `{evidence}`")
    write_text(OUT_DIR / "development_timeline.md", "\n".join(lines) + "\n")

    rows = dataset_history_rows(datasets)
    dataset_lines = ["# Dataset Exploration Timeline", ""]
    dataset_lines.append("The project used different datasets for different research questions. Detection data, zone/VPR data, robustness data, and identity data must not be treated as the same benchmark.")
    dataset_lines.append("")
    dataset_lines.append(markdown_table(rows, ["Dataset", "Path", "Exists", "Images", "Labels", "Role", "Status", "Thesis wording"]))
    dataset_lines.append("\n## Thesis-Safe Interpretation")
    dataset_lines.extend([
        "- The combined valve/gauge YOLO dataset is the main quantitative detector dataset.",
        "- OpenLORIS and NYC-Indoor-VPR are proxy sources for zone localization/VPR, not oil-facility identity datasets.",
        "- ExDARK supports qualitative low-light robustness discussion only unless annotations are parsed and mapped later.",
        "- The oil/gas expanded datasets are staged for inspection and future detector expansion; they are not merged into the current runtime detector.",
        "- The proxy device benchmark validates enrollment, OCR/ReID, and FAISS mechanics under controlled conditions; real repeated photos of physical devices are still required for external identity validation.",
    ])
    write_text(OUT_DIR / "dataset_exploration_timeline.md", "\n".join(dataset_lines) + "\n")


def write_methodology(config: Dict[str, Any], routes: Dict[str, Any], frontend: Dict[str, Any]) -> None:
    lines = ["# Methodology Pipeline", ""]
    lines.append("## Runtime Flow Verified From Code")
    lines.extend([
        "1. Input is received through FastAPI inference routes in `backend/app/routes/infer.py`.",
        "2. `backend/app/pipeline.py` orchestrates quality estimation, zone localization, detection, OCR/ReID evidence extraction, fusion, policy, tracking, storage, and response formatting.",
        "3. Frame quality diagnostics are implemented in `backend/app/quality.py` and configured by `tau_blur` and `tau_low_light`.",
        "4. Zone candidates are retrieved using `backend/app/zone_localizer.py` over FAISS/numpy storage in `backend/app/faiss_store.py`.",
        "5. Valve/gauge detections are produced by `backend/app/detector.py` using the configured YOLO model.",
        "6. Candidate ROIs are passed to OCR (`backend/app/ocr.py`) and device retrieval (`backend/app/reid.py`).",
        "7. `backend/app/fusion.py` combines zone, OCR, and ReID evidence; `backend/app/policy.py` decides whether evidence is sufficient or uncertain.",
        "8. Observations and feedback are stored through `backend/app/db.py`.",
    ])
    lines.append("\n## Configuration Thresholds")
    lines.append(markdown_table([{"Parameter": k, "Value": v} for k, v in config.items()], ["Parameter", "Value"]))
    lines.append("\n## Backend Routes")
    for path, endpoints in routes.items():
        lines.append(f"- `{path}`")
        if endpoints:
            for ep in endpoints:
                lines.append(f"  - `{ep['method']}` {ep['decorator']}")
        else:
            lines.append("  - no route decorators parsed")
    lines.append("\n## Frontend Files")
    for item in frontend["files"]:
        lines.append(f"- `{item['path']}` — lines={item['lines']}, feedback={item['mentions_feedback']}, ask={item['mentions_ask']}, camera={item['mentions_webcam']}")
    write_text(OUT_DIR / "methodology_pipeline.md", "\n".join(lines) + "\n")


def write_results_tables(detector: Dict[str, Any], identity: Dict[str, Any], robustness: Dict[str, Any], datasets: Dict[str, Any]) -> None:
    lines = ["# Results Tables", ""]
    lines.append("## Detector Results")
    det_rows = []
    for path, payload in detector.get("summaries", {}).items():
        if "validation" in payload:
            val = payload["validation"]
            det_rows.append({"Artifact": path, "Split": "validation", "Precision": val.get("precision"), "Recall": val.get("recall"), "mAP50": val.get("map50"), "mAP50-95": val.get("map50_95")})
        else:
            det_rows.append({"Artifact": path, "Split": payload.get("split", "not specified"), "Precision": payload.get("precision"), "Recall": payload.get("recall"), "mAP50": payload.get("map50"), "mAP50-95": payload.get("map50_95")})
    lines.append(markdown_table(det_rows, ["Artifact", "Split", "Precision", "Recall", "mAP50", "mAP50-95"]))
    lines.append("\n## Combined Dataset Counts")
    combined = datasets.get("combined_valve_gauge", {})
    split_rows = []
    for split, item in combined.get("splits", {}).items():
        split_rows.append({"Split": split, "Images": item.get("images"), "Labels": item.get("labels"), "Annotations": item.get("annotations"), "Distribution": item.get("class_distribution")})
    lines.append(markdown_table(split_rows, ["Split", "Images", "Labels", "Annotations", "Distribution"]))
    lines.append("\n## Identity Benchmark Summary")
    id_keys = [
        "devices_count", "device_refs_count", "device_faiss_size", "total_query_images",
        "reid_top1_accuracy", "reid_topk_accuracy", "ocr_visible_tag_images",
        "ocr_exact_matches", "ocr_visible_tag_exact_match_rate", "api_tested",
        "at_least_one_accepted",
    ]
    lines.append(markdown_table([{"Metric": k, "Value": identity.get(k, "not found")} for k in id_keys], ["Metric", "Value"]))
    lines.append("\n## Robustness Detector Results")
    rob_rows = []
    for item in robustness.get("summary", {}).get("results", []):
        labeled = item.get("labeled") or {}
        pred = item.get("prediction") or {}
        rob_rows.append({
            "Condition": item.get("condition"),
            "mAP50": labeled.get("map50"),
            "mAP50-95": labeled.get("map50_95"),
            "Mean confidence": pred.get("mean_confidence"),
            "No detections": pred.get("no_detection_count"),
        })
    lines.append(markdown_table(rob_rows, ["Condition", "mAP50", "mAP50-95", "Mean confidence", "No detections"]))
    write_text(OUT_DIR / "results_tables.md", "\n".join(lines) + "\n")
    with (OUT_DIR / "detector_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "split", "precision", "recall", "map50", "map50_95"])
        writer.writeheader()
        for row in det_rows:
            writer.writerow({
                "artifact": row.get("Artifact"),
                "split": row.get("Split"),
                "precision": row.get("Precision"),
                "recall": row.get("Recall"),
                "map50": row.get("mAP50"),
                "map50_95": row.get("mAP50-95"),
            })


def write_experiment_inventory(detector: Dict[str, Any], robustness: Dict[str, Any], identity: Dict[str, Any], oilgas: Dict[str, Any]) -> None:
    lines = ["# Experiment Inventory", ""]
    lines.append("## Detector Training/Evaluation")
    lines.append("- Main training summary: `artifacts/detection_training/valvelens_v1_cuda_summary.json`")
    lines.append("- Test summary: `artifacts/detection_training/valvelens_v1_cuda_testval_test_summary.json`")
    lines.append("- Integrated model: `models/detector.pt`")
    lines.append("\n## Identity Benchmark")
    lines.append("- Summary: `artifacts/identity_benchmark/identity_benchmark_summary.json`")
    lines.append(f"- Devices: `{identity.get('devices_count', 'not found')}`")
    lines.append(f"- Device refs: `{identity.get('device_refs_count', 'not found')}`")
    lines.append(f"- Device FAISS size: `{identity.get('device_faiss_size', 'not found')}`")
    lines.append("\n## Robustness")
    lines.append("- Summary: `artifacts/robustness/robustness_summary.json`" if robustness.get("exists") else "- Robustness summary not found")
    lines.append("\n## Oil/Gas Expanded Dataset Staging")
    for path in oilgas:
        lines.append(f"- `{path}`")
    write_text(OUT_DIR / "experiment_inventory.md", "\n".join(lines) + "\n")


def write_status_table(rows: List[Dict[str, str]]) -> None:
    lines = ["# Implementation Status Table", "", markdown_table(rows, ["Module", "Purpose", "Current status", "Evidence", "Thesis category"])]
    write_text(OUT_DIR / "implementation_status_table.md", "\n".join(lines))


def write_figures() -> None:
    figures = [
        "System architecture diagram",
        "Backend inference pipeline diagram",
        "Dataset decomposition diagram",
        "OpenLORIS/zone keyframe sample",
        "Valve/gauge dataset samples",
        "Detector prediction success case",
        "Detector failure case",
        "Frontend Live UI screenshot",
        "Debug/status screenshot",
        "Decision policy/evidence diagram",
        "Robustness preprocessing before/after panel",
        "Oil/gas proxy identity benchmark contact sheet",
    ]
    write_text(OUT_DIR / "figures_to_capture.md", "# Figures To Capture\n\n" + "\n".join(f"- {item}" for item in figures) + "\n")
    mermaid = r"""# Thesis Figures Mermaid

## Overall System Pipeline
```mermaid
flowchart LR
  A[Input image/video/webcam frame] --> B[Quality diagnostics]
  B --> C[Zone localization / VPR]
  B --> D[YOLOv8 valve/gauge detector]
  D --> E[ROI extraction]
  E --> F[OCR device tag parsing]
  E --> G[ReID embedding search]
  C --> H[Fusion scoring]
  F --> H
  G --> H
  H --> I[Uncertainty policy]
  I --> J[Observation storage]
  I --> K[Frontend overlay + side panel]
  K --> L[Human feedback]
  L --> J
```

## Dataset Decomposition
```mermaid
flowchart TD
  A[ValveLens datasets] --> B[Main detection: combined valve/gauge]
  A --> C[Zone/VPR data: OpenLORIS-style folders]
  A --> D[Robustness: synthetic corruptions + preprocessing]
  A --> E[Identity: proxy inventory refs/queries]
  A --> F[Expanded oil/gas staging]
  F --> F1[Oil Refinery]
  F --> F2[Elementos Offshore]
  F --> F3[Wellhead/industrial optional]
```

## VPR Retrieval Flow
```mermaid
flowchart LR
  A[Zone keyframe images] --> B[Embedding model]
  B --> C[FAISS zone index]
  D[Query frame] --> E[Embedding model]
  E --> C
  C --> F[Top-k keyframes]
  F --> G[Aggregate by zone]
  G --> H[Zone candidates]
```

## Decision Policy
```mermaid
flowchart TD
  A[Fusion evidence] --> B{Scores exceed thresholds?}
  B -- no --> C[UNCERTAIN / defer]
  B -- yes --> D{Top candidate gap sufficient?}
  D -- no --> C
  D -- yes --> E[ACCEPTED]
  C --> F[Ask for feedback / more evidence]
  E --> G[Store final identity]
```

## v0.2 to v0.3 Roadmap
```mermaid
timeline
  title ValveLens Progression
  v0.2 : Zone retrieval : YOLO detection : UI evidence display
  v0.2+ : OCR/ReID wiring : feedback : tracking : robustness experiments
  v0.3 target : enrolled devices : validated identity : accepted decision path : metrics export
  Future : real device photos : runtime robustness policy : evidence-aware VLM explanations
```
"""
    write_text(OUT_DIR / "thesis_figures_mermaid.md", mermaid)


def safe_claims(identity: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    safe = [
        "The repository integrates a trained YOLOv8 valve/gauge detector through `models/detector.pt` and `backend/app/detector.py`.",
        "The backend implements FAISS/numpy based zone and device retrieval through `backend/app/faiss_store.py`.",
        "The inference pipeline combines quality diagnostics, zone localization, detection, OCR, ReID, fusion, decision policy, observation storage, and frontend evidence display.",
        "A controlled proxy identity benchmark exists and validates the mechanics of device enrollment, references, FAISS indexing, OCR, and ReID.",
        "Robustness preprocessing experiments are implemented as offline experiments, not as runtime inference behavior.",
    ]
    if identity.get("api_tested") is False:
        safe.append("API accepted-decision validation for the current proxy identity benchmark is not yet verified.")
    unsafe = [
        "The system is deployed or validated in a real oil facility.",
        "The system fully solves real device identity without real repeated physical device images.",
        "The uncertainty scores are calibrated probabilistic uncertainty.",
        "Runtime preprocessing improves live inference; current preprocessing evidence is offline experimental evidence.",
        "A VLM performs perception or identity reasoning; the current ask route is rule/evidence based.",
    ]
    return safe, unsafe


def write_summary_md(evidence: Dict[str, Any]) -> None:
    safe, unsafe = evidence["safe_claims"], evidence["unsafe_claims"]
    lines = ["# ValveLens Thesis Evidence Summary", ""]
    lines.append(f"Generated: `{evidence['generated_at']}`")
    lines.append("\n## 1. Executive Summary")
    lines.append("ValveLens is a computer-vision guided assistant for industrial device localization and detection. The repository contains a FastAPI backend, React frontend, YOLOv8 detector integration, FAISS-based zone/device retrieval, OCR/ReID identity paths, feedback storage, tracking, and offline robustness experiments. The current evidence supports a thesis methodology and results chapter, with explicit limits around real industrial identity validation.")
    lines.append("\n## 1.1 Development Storyline")
    lines.append("The project did not begin from one perfect industrial dataset. It evolved through a sequence of smaller, defensible experiments: public indoor datasets were used to test zone retrieval, Roboflow valve/gauge datasets were consolidated for detector training, synthetic degradations and ExDARK supported robustness analysis, and controlled proxy device inventories were created because real repeated device-reference photos were not available yet.")
    lines.append("The detailed timeline is saved in `artifacts/thesis/development_timeline.md`; dataset-specific history is saved in `artifacts/thesis/dataset_exploration_timeline.md`.")
    lines.append("\n## 2. Repository Structure")
    lines.append("See `artifacts/thesis/repo_structure_summary.md` for the detailed file map.")
    lines.append("\n## 3. Implemented System Modules")
    lines.append("See `artifacts/thesis/implementation_status_table.md`.")
    lines.append("\n## 4. Current Pipeline")
    lines.append("The verified pipeline is documented in `artifacts/thesis/methodology_pipeline.md`.")
    lines.append("\n## 5. Dataset Inventory")
    lines.append("See `artifacts/thesis/dataset_inventory.md`, `artifacts/thesis/dataset_counts.csv`, and `artifacts/thesis/dataset_exploration_timeline.md`. The important distinction is that detection, zone localization, robustness, and identity use different data sources and answer different research questions.")
    lines.append("\n## 6. Detection Dataset Preparation")
    combined = evidence["datasets"].get("combined_valve_gauge", {})
    lines.append(f"The main combined YOLO dataset is `{combined.get('path')}` with `{combined.get('total_images')}` images, `{combined.get('total_label_files')}` label files, and `{combined.get('total_annotations')}` annotations. Class distribution: `{combined.get('class_distribution')}`.")
    lines.append("\n## 7. Detector Training and Results")
    lines.append("Detector metrics are stored in `artifacts/detection_training/valvelens_v1_cuda_summary.json` and `artifacts/detection_training/valvelens_v1_cuda_testval_test_summary.json`; extracted tables are in `artifacts/thesis/results_tables.md`.")
    lines.append("\n## 8. Zone Localization / VPR Setup")
    zone_index = evidence["faiss_status"]["indexes"].get("zones", {})
    lines.append(f"Zone index metadata exists: `{zone_index.get('meta_exists')}`; metadata items: `{zone_index.get('meta_items', 'not verified')}`. Zone smoke results are in `artifacts/thesis/zone_smoke_results.md`.")
    lines.append("\n## 9. OCR and ReID Status")
    identity = evidence["identity_status"]
    lines.append(f"Identity benchmark: devices `{identity.get('devices_count')}`, device refs `{identity.get('device_refs_count')}`, FAISS size `{identity.get('device_faiss_size')}`, ReID top-1 `{identity.get('reid_top1_accuracy')}`, OCR exact-match `{identity.get('ocr_visible_tag_exact_match_rate')}`. This is proxy benchmark evidence, not real industrial identity validation.")
    lines.append("For thesis writing, treat identity rates as controlled benchmark diagnostics. The main claim is that the enrollment, indexing, OCR, and ReID mechanics are measurable and working under a controlled proxy setup; do not describe these values as deployment accuracy.")
    lines.append("\n## 10. Feedback / HITL Status")
    db = evidence["database_status"]
    lines.append(f"Feedback table count: `{db.get('counts', {}).get('feedback', 'not found')}`. Feedback type distribution: `{db.get('feedback_types', {})}`.")
    lines.append("\n## 11. Robustness / Preprocessing Status")
    lines.append("Robustness outputs are summarized in `artifacts/robustness/robustness_summary.json` and extracted in `artifacts/thesis/results_tables.md`. These are offline experiments.")
    lines.append("\n## 12. Tests and Validation")
    tests = evidence["tests"]
    lines.append(f"Backend tests: `{tests.get('status')}`, passed `{tests.get('passed')}`, failed `{tests.get('failed')}`. Full output: `{tests.get('output')}`.")
    lines.append("\n## 13. Results That Can Be Claimed Safely")
    lines.extend(f"- {claim}" for claim in safe)
    lines.append("\n## 14. Partial / Ongoing Work")
    lines.extend([
        "- Real device identity remains stronger only after collecting repeated physical device images.",
        "- API accepted-decision validation is separate from proxy ReID/OCR validation and remains unverified if `api_tested=false`.",
        "- Expanded oil/gas datasets are staged and inspected, not merged into the active runtime detector.",
    ])
    lines.append("\n## 15. Things That Must NOT Be Overclaimed")
    lines.extend(f"- {claim}" for claim in unsafe)
    lines.append("\n## 16. Recommended Methodology Text")
    lines.append("Describe ValveLens as a modular evidence pipeline: quality assessment, place/zone retrieval, object detection, OCR/ReID identity evidence, fusion, abstention policy, observation logging, and human feedback. Emphasize that the VLM/interactive assistant direction is evidence-aware and not a replacement for perception.")
    lines.append("\n## 17. Recommended Results Text")
    lines.append("Report detector metrics from the saved YOLO evaluation artifacts, identity proxy benchmark metrics from `artifacts/identity_benchmark`, database/index counts from `backend/data`, and robustness metrics from `artifacts/robustness`. State that identity metrics are proxy-controlled benchmark results. If a percentage is very high, explain that the benchmark is controlled and generated from selected object crops, so the number demonstrates pipeline mechanics rather than real-world generalization.")
    lines.append("\n## 18. Recommended Thesis Tables and Figures")
    lines.append("Use `results_tables.md`, `implementation_status_table.md`, `dataset_inventory.md`, `figures_to_capture.md`, and `thesis_figures_mermaid.md`.")
    lines.append("\n## Safe Claims vs Unsafe Claims")
    lines.append("\n### Safe")
    lines.extend(f"- {claim}" for claim in safe)
    lines.append("\n### Unsafe")
    lines.extend(f"- {claim}" for claim in unsafe)
    write_text(OUT_DIR / "thesis_evidence_summary.md", "\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_structure = collect_repo_structure()
    config = collect_config()
    routes = collect_routes()
    frontend = collect_frontend()
    database = collect_database_status()
    faiss = collect_faiss_status()
    datasets = collect_dataset_inventory()
    detector = collect_detector_results()
    robustness = collect_robustness()
    identity = collect_identity()
    oilgas = collect_oilgas_reports()
    timeline = collect_development_timeline()
    tests = run_pytest()
    zone_smoke = run_zone_smoke(database)
    status_rows = implementation_status(identity, database, datasets)
    safe, unsafe = safe_claims(identity)

    evidence = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo": repo_structure,
        "config": config,
        "routes": routes,
        "frontend": frontend,
        "datasets": datasets,
        "models": detector.get("integration_weights", {}),
        "detector_results": detector,
        "zone_results": zone_smoke,
        "database_status": database,
        "faiss_status": faiss,
        "identity_status": identity,
        "feedback_status": {
            "count": database.get("counts", {}).get("feedback", "not found"),
            "types": database.get("feedback_types", {}),
        },
        "robustness_status": robustness,
        "oilgas_expanded_reports": oilgas,
        "development_timeline": timeline,
        "dataset_history": dataset_history_rows(datasets),
        "tests": tests,
        "safe_claims": safe,
        "unsafe_claims": unsafe,
        "recommended_figures": [
            "system architecture diagram",
            "backend pipeline diagram",
            "dataset decomposition diagram",
            "frontend UI screenshot",
            "detector prediction examples",
            "robustness preprocessing comparison",
        ],
    }

    write_repo_structure(repo_structure)
    write_dataset_inventory(datasets, oilgas)
    write_development_timeline(timeline, datasets)
    write_methodology(config, routes, frontend)
    write_results_tables(detector, identity, robustness, datasets)
    write_experiment_inventory(detector, robustness, identity, oilgas)
    write_status_table(status_rows)
    write_figures()
    write_summary_md(evidence)
    write_json(OUT_DIR / "thesis_evidence_summary.json", evidence)
    write_json(OUT_DIR / "backend_status.json", {"database": database, "faiss": faiss, "routes": routes})

    print("Thesis evidence package generated.")
    print(f"  output: {rel(OUT_DIR)}")
    print(f"  tests: {tests.get('status')} passed={tests.get('passed')} failed={tests.get('failed')}")
    print(f"  combined images: {datasets.get('combined_valve_gauge', {}).get('total_images')}")
    print(f"  detector model exists: {detector.get('integration_weights', {}).get('exists')}")
    print(f"  identity devices: {identity.get('devices_count', 'not found')}")
    print(f"  identity refs: {identity.get('device_refs_count', 'not found')}")
    print(f"  identity FAISS size: {identity.get('device_faiss_size', 'not found')}")
    print(f"  API tested: {identity.get('api_tested', 'not found')}")
    print("  timeline: artifacts\\thesis\\development_timeline.md")
    print("  dataset history: artifacts\\thesis\\dataset_exploration_timeline.md")


if __name__ == "__main__":
    main()
