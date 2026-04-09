import logging
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np


LOGGER = logging.getLogger("valvelens.detector")
REPO_ROOT = Path(__file__).resolve().parents[2]


class Detector:
    def __init__(self, model_name: str = "models/detector.pt") -> None:
        self.model_name = model_name
        self.resolved_model_path = self._resolve_model_path(model_name)
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.fallback_reason = None
        self._fallback_warned = False

        LOGGER.info(
            "Initializing detector with configured path '%s' resolved to '%s'",
            self.model_name,
            self.resolved_model_path,
        )

        if not self.resolved_model_path.exists():
            self.fallback_reason = f"detector weights not found: {self.resolved_model_path}"
            LOGGER.warning("Detector fallback enabled because %s", self.fallback_reason)
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.resolved_model_path))
            self.class_names = self._normalize_class_names(self.model.names)
            LOGGER.info(
                "Loaded detector weights from '%s' with classes: %s",
                self.resolved_model_path,
                self.class_names,
            )
        except ImportError as exc:
            self.fallback_reason = f"ultralytics import failed: {exc}"
            LOGGER.warning("Detector fallback enabled because %s", self.fallback_reason)
            self.model = None
            self.class_names = {}
        except Exception as exc:
            self.fallback_reason = f"failed to load detector model from {self.resolved_model_path}: {exc}"
            LOGGER.warning("Detector fallback enabled because %s", self.fallback_reason)
            self.model = None
            self.class_names = {}

    def _resolve_model_path(self, model_name: str) -> Path:
        candidate = Path(model_name).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (REPO_ROOT / candidate).resolve()

    def _normalize_class_names(self, names: object) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {idx: str(value) for idx, value in enumerate(names)}
        return {}

    def _map_class(self, name: str) -> str:
        name_lower = name.lower()
        if "valve" in name_lower:
            return "valve"
        if "gauge" in name_lower or "meter" in name_lower:
            return "gauge"
        if "panel" in name_lower or "screen" in name_lower:
            return "panel"
        if "tag" in name_lower or "label" in name_lower:
            return "tag"
        return "unknown"

    def _fallback_detection(self, width: int, height: int) -> List[Dict]:
        if not self._fallback_warned:
            LOGGER.warning("Using fallback detector output because %s", self.fallback_reason or "model is unavailable")
            self._fallback_warned = True
        return [
            {
                "det_id": str(uuid4()),
                "cls": "unknown",
                "conf": 0.5,
                "bbox": {"x1": 0, "y1": 0, "x2": width - 1, "y2": height - 1},
            }
        ]

    def detect(self, frame_bgr: np.ndarray, conf_thres: float = 0.25) -> List[Dict]:
        height, width = frame_bgr.shape[:2]
        detections: List[Dict] = []
        if self.model is None:
            return self._fallback_detection(width, height)

        try:
            results = self.model.predict(source=frame_bgr, verbose=False)
        except Exception as exc:
            LOGGER.warning(
                "Detector inference failed for '%s' (%s); using fallback output",
                self.resolved_model_path,
                exc,
            )
            return self._fallback_detection(width, height)

        if not results:
            return detections

        for res in results:
            boxes = res.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_thres:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                cls_name = self.class_names.get(cls_id, "unknown")
                detections.append(
                    {
                        "det_id": str(uuid4()),
                        "cls": self._map_class(cls_name),
                        "conf": conf,
                        "bbox": {
                            "x1": int(max(0, x1)),
                            "y1": int(max(0, y1)),
                            "x2": int(min(width - 1, x2)),
                            "y2": int(min(height - 1, y2)),
                        },
                    }
                )
        return detections
