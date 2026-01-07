from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import uuid4


def iou(box_a: dict, box_b: dict) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = (box_a["x2"] - box_a["x1"] + 1) * (box_a["y2"] - box_a["y1"] + 1)
    area_b = (box_b["x2"] - box_b["x1"] + 1) * (box_b["y2"] - box_b["y1"] + 1)
    union = max(1, area_a + area_b - inter)
    return inter / union


@dataclass
class Track:
    track_id: str
    bbox: dict
    streak: int = 0
    age: int = 0


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []

    def update(self, detections: List[dict]) -> Dict[str, Tuple[str, int]]:
        matches: Dict[str, Tuple[str, int]] = {}
        assigned_tracks = set()

        for det in detections:
            best_iou = 0.0
            best_track = None
            for idx, track in enumerate(self.tracks):
                if idx in assigned_tracks:
                    continue
                score = iou(det["bbox"], track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = idx
            if best_track is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track]
                track.bbox = det["bbox"]
                track.streak += 1
                track.age = 0
                assigned_tracks.add(best_track)
                matches[det["det_id"]] = (track.track_id, track.streak)
            else:
                new_track = Track(track_id=str(uuid4()), bbox=det["bbox"], streak=1, age=0)
                self.tracks.append(new_track)
                matches[det["det_id"]] = (new_track.track_id, new_track.streak)

        # Age and prune stale tracks
        for idx, track in enumerate(self.tracks):
            if idx not in assigned_tracks:
                track.streak = 0
            track.age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return matches


class TrackerManager:
    def __init__(self) -> None:
        self.trackers: Dict[str, IOUTracker] = {}

    def update(self, session_id: str, detections: List[dict]) -> Dict[str, Tuple[str, int]]:
        if session_id not in self.trackers:
            self.trackers[session_id] = IOUTracker()
        return self.trackers[session_id].update(detections)
