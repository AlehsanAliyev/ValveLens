from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from uuid import uuid4


def iou(box_a: dict, box_b: dict) -> float:
    x1 = max(int(box_a["x1"]), int(box_b["x1"]))
    y1 = max(int(box_a["y1"]), int(box_b["y1"]))
    x2 = min(int(box_a["x2"]), int(box_b["x2"]))
    y2 = min(int(box_a["y2"]), int(box_b["y2"]))
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = max(1, (int(box_a["x2"]) - int(box_a["x1"]) + 1) * (int(box_a["y2"]) - int(box_a["y1"]) + 1))
    area_b = max(1, (int(box_b["x2"]) - int(box_b["x1"]) + 1) * (int(box_b["y2"]) - int(box_b["y1"]) + 1))
    union = max(1, area_a + area_b - inter)
    return float(inter / union)


@dataclass
class Track:
    track_id: str
    bbox: dict
    window_size: int
    streak: int = 1
    hits: int = 1
    missed: int = 0
    det_conf_history: Deque[float] = field(init=False)
    ocr_conf_history: Deque[float] = field(init=False)
    reid_top1_history: Deque[float] = field(init=False)
    selected_device_history: Deque[str] = field(init=False)

    def __post_init__(self) -> None:
        self.det_conf_history = deque(maxlen=self.window_size)
        self.ocr_conf_history = deque(maxlen=self.window_size)
        self.reid_top1_history = deque(maxlen=self.window_size)
        self.selected_device_history = deque(maxlen=self.window_size)

    def mark_matched(self, bbox: dict) -> None:
        self.bbox = bbox
        self.streak += 1
        self.hits += 1
        self.missed = 0

    def mark_missed(self) -> None:
        self.streak = 0
        self.missed += 1

    def add_signals(
        self,
        det_conf: float,
        ocr_conf: float,
        reid_top1: float,
        selected_device_id: Optional[str],
    ) -> Dict[str, object]:
        self.det_conf_history.append(float(det_conf))
        self.ocr_conf_history.append(float(ocr_conf))
        self.reid_top1_history.append(float(reid_top1))
        if selected_device_id:
            self.selected_device_history.append(str(selected_device_id))

        smoothed_det = sum(self.det_conf_history) / len(self.det_conf_history)
        smoothed_ocr = sum(self.ocr_conf_history) / len(self.ocr_conf_history)
        smoothed_reid = sum(self.reid_top1_history) / len(self.reid_top1_history)

        stable_device = None
        if self.selected_device_history:
            stable_device = Counter(self.selected_device_history).most_common(1)[0][0]

        return {
            "smoothed_det_conf": smoothed_det,
            "smoothed_ocr_conf": smoothed_ocr,
            "smoothed_reid_top1": smoothed_reid,
            "smoothed_selected_device_id": stable_device,
            "track_stability": self.streak,
            "track_history_len": len(self.det_conf_history),
        }


class IOUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        smoothing_window: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.smoothing_window = max(1, int(smoothing_window))
        self.tracks: Dict[str, Track] = {}

    def _new_track(self, bbox: dict) -> Track:
        return Track(track_id=str(uuid4()), bbox=bbox, window_size=self.smoothing_window)

    def update(self, detections: List[dict]) -> Dict[str, Dict[str, object]]:
        assignments: Dict[str, Dict[str, object]] = {}
        unmatched_track_ids = set(self.tracks.keys())

        for det in detections:
            det_box = det["bbox"]
            best_track_id = None
            best_iou = 0.0
            for track_id in list(unmatched_track_ids):
                track = self.tracks[track_id]
                score = iou(det_box, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_id]
                track.mark_matched(det_box)
                unmatched_track_ids.discard(best_track_id)
            else:
                track = self._new_track(det_box)
                self.tracks[track.track_id] = track

            assignments[det["det_id"]] = {
                "track_id": track.track_id,
                "track_stability": track.streak,
            }

        for track_id in unmatched_track_ids:
            self.tracks[track_id].mark_missed()

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track.missed <= self.max_missed
        }
        return assignments

    def update_signals(
        self,
        track_id: str,
        det_conf: float,
        ocr_conf: float,
        reid_top1: float,
        selected_device_id: Optional[str],
    ) -> Dict[str, object]:
        track = self.tracks.get(track_id)
        if track is None:
            return {
                "smoothed_det_conf": float(det_conf),
                "smoothed_ocr_conf": float(ocr_conf),
                "smoothed_reid_top1": float(reid_top1),
                "smoothed_selected_device_id": selected_device_id,
                "track_stability": 0,
                "track_history_len": 1,
            }
        return track.add_signals(det_conf, ocr_conf, reid_top1, selected_device_id)


class TrackerManager:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        smoothing_window: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.smoothing_window = smoothing_window
        self.trackers: Dict[str, IOUTracker] = {}

    def _get_tracker(self, session_id: str) -> IOUTracker:
        if session_id not in self.trackers:
            self.trackers[session_id] = IOUTracker(
                iou_threshold=self.iou_threshold,
                max_missed=self.max_missed,
                smoothing_window=self.smoothing_window,
            )
        return self.trackers[session_id]

    def update(self, session_id: str, detections: List[dict]) -> Dict[str, Dict[str, object]]:
        return self._get_tracker(session_id).update(detections)

    def update_signals(
        self,
        session_id: str,
        track_id: str,
        det_conf: float,
        ocr_conf: float,
        reid_top1: float,
        selected_device_id: Optional[str],
    ) -> Dict[str, object]:
        return self._get_tracker(session_id).update_signals(
            track_id=track_id,
            det_conf=det_conf,
            ocr_conf=ocr_conf,
            reid_top1=reid_top1,
            selected_device_id=selected_device_id,
        )
