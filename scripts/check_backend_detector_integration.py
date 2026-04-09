from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
CONFIG_PATH = REPO_ROOT / "backend" / "app" / "config.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    configured_path = config.get("detector_model", "models/detector.pt")

    from app.detector import Detector

    detector = Detector(configured_path)
    sample_image = next(
        iter(sorted((REPO_ROOT / "data" / "detection" / "combined" / "test" / "images").glob("*"))),
        None,
    )

    sample_result = None
    if sample_image is not None and detector.model is not None:
        import cv2

        frame = cv2.imread(str(sample_image))
        if frame is not None:
            detections = detector.detect(frame, conf_thres=0.25)
            sample_result = {
                "sample_image": str(sample_image),
                "detections": len(detections),
                "classes": sorted({det["cls"] for det in detections}),
            }

    summary = {
        "config_path": str(CONFIG_PATH),
        "configured_detector_model": configured_path,
        "resolved_detector_model": str(detector.resolved_model_path),
        "resolved_exists": detector.resolved_model_path.exists(),
        "ultralytics_loaded": detector.model is not None,
        "fallback_reason": detector.fallback_reason,
        "class_names": detector.class_names,
        "sample_result": sample_result,
    }

    print(json.dumps(summary, indent=2))

    if not summary["resolved_exists"] or not summary["ultralytics_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
