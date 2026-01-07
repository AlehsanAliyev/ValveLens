from typing import Dict, List
from uuid import uuid4

import numpy as np


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model = None
        self.class_names = {}
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            self.class_names = self.model.names
        except Exception:
            self.model = None
            self.class_names = {}

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

    def detect(self, frame_bgr: np.ndarray, conf_thres: float = 0.25) -> List[Dict]:
        height, width = frame_bgr.shape[:2]
        detections: List[Dict] = []
        if self.model is None:
            detections.append(
                {
                    "det_id": str(uuid4()),
                    "cls": "unknown",
                    "conf": 0.5,
                    "bbox": {"x1": 0, "y1": 0, "x2": width - 1, "y2": height - 1},
                }
            )
            return detections

        results = self.model.predict(source=frame_bgr, verbose=False)
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
