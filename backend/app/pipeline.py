import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import yaml
from PIL import Image

from app.db import fetch_device_ids
from app.detector import Detector
from app.embeddings import Embedder
from app.faiss_store import FaissIndex
from app.fusion import fuse_scores
from app.ocr import OCRReader, extract_device_id, normalize_text
from app.policy import decide_policy
from app.quality import compute_quality
from app.reid import ReIDEmbedder
from app.schemas import (
    BBox,
    DecisionInfo,
    DetectionInfo,
    InferenceResponse,
    InputInfo,
    OcrInfo,
    QualityInfo,
    ReIdInfo,
    ReIdMatch,
    RoiInfo,
    ZoneCandidate,
    ZoneInfo,
    ZoneTop1,
)
from app.segmenter import Segmenter
from app.tracker import TrackerManager

LOGGER = logging.getLogger("valvelens")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ROIS_DIR = DATA_DIR / "rois"
ROIS_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_CACHE: Optional[Dict] = None


def load_config() -> Dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg_path = Path(__file__).resolve().parent / "config.yaml"
        _CONFIG_CACHE = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return _CONFIG_CACHE


def _save_image(image: Image.Image, filename: str) -> str:
    path = ROIS_DIR / filename
    image.save(path)
    return str(path)


class InferencePipeline:
    def __init__(self) -> None:
        self.config = load_config()
        self.embedder = Embedder()
        self.zone_index = FaissIndex("zones", self.embedder.dim)
        self.device_index = FaissIndex("devices", self.embedder.dim)
        self.detector = Detector(self.config.get("detector_model", "yolov8n.pt"))
        self.segmenter = Segmenter()
        self.ocr = OCRReader()
        self.reid = ReIDEmbedder(self.embedder)
        self.tracker = TrackerManager()

    def _zone_candidates(self, image: Image.Image) -> Tuple[List[ZoneCandidate], Optional[ZoneTop1]]:
        vec = self.embedder.embed_image(image)
        matches = self.zone_index.search(vec, topk=self.config.get("max_zone_candidates", 5))
        candidates = [ZoneCandidate(zone_id=m[0]["zone_id"], score=float(m[1])) for m in matches]
        top1 = ZoneTop1(**candidates[0].dict()) if candidates else None
        return candidates, top1

    def _filter_device_matches(self, matches: List[Tuple[Dict, float]], zone_id: Optional[str]) -> List[Tuple[Dict, float]]:
        if not zone_id or not matches:
            return matches
        filtered = [m for m in matches if m[0].get("zone_id") == zone_id]
        return filtered if filtered else matches

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        input_type: str,
        source: str,
        frame_index: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> InferenceResponse:
        request_id = str(uuid4())
        timestamp = datetime.utcnow().isoformat()

        quality = compute_quality(
            frame_bgr,
            tau_blur=self.config["tau_blur"],
            tau_low_light=self.config["tau_low_light"],
        )
        LOGGER.info(
            "stage=quality request_id=%s blur=%.3f brightness=%.3f glare=%.3f",
            request_id,
            quality["blur_score"],
            quality["brightness"],
            quality["glare_score"],
        )

        pil_image = Image.fromarray(frame_bgr[:, :, ::-1])
        zone_candidates, zone_top1 = self._zone_candidates(pil_image)
        zone_score = zone_top1.score if zone_top1 else 0.0
        zone_id = zone_top1.zone_id if zone_top1 else None
        LOGGER.info(
            "stage=zone request_id=%s top1=%s score=%.3f candidates=%d",
            request_id,
            zone_id,
            zone_score,
            len(zone_candidates),
        )

        detections_raw = self.detector.detect(frame_bgr, conf_thres=self.config.get("tau_det", 0.4))
        LOGGER.info(
            "stage=detect request_id=%s detections=%d",
            request_id,
            len(detections_raw),
        )
        track_map = {}
        if session_id:
            track_map = self.tracker.update(session_id, detections_raw)

        device_ids = fetch_device_ids()
        norm_device_ids = {}
        for dev_id in device_ids:
            key = normalize_text(dev_id)
            if key:
                norm_device_ids[key] = dev_id
                norm_device_ids[key.replace("-", "")] = dev_id

        detections: List[DetectionInfo] = []
        best_det_conf = 0.0
        highlight_det_id = None
        best_ocr_match = None
        best_reid_match = None
        best_fused_score = -1.0

        for det in detections_raw:
            bbox = det["bbox"]
            det_conf = float(det["conf"])
            if det_conf > best_det_conf:
                best_det_conf = det_conf
                highlight_det_id = det["det_id"]

            crop_img, mask = self.segmenter.refine_roi(frame_bgr, bbox)
            LOGGER.info(
                "stage=roi request_id=%s det_id=%s mask=%s",
                request_id,
                det["det_id"],
                "yes" if mask is not None else "no",
            )
            crop_filename = f"{request_id}_{det['det_id']}.png"
            crop_path = _save_image(crop_img, crop_filename)
            mask_path = None
            if mask is not None:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_path = _save_image(mask_img, f"{request_id}_{det['det_id']}_mask.png")

            ocr_result = self.ocr.read(crop_img)
            ocr_text = ocr_result.get("text")
            ocr_conf = float(ocr_result.get("conf") or 0.0)
            LOGGER.info(
                "stage=ocr request_id=%s det_id=%s conf=%.3f text=%s",
                request_id,
                det["det_id"],
                ocr_conf,
                ocr_text,
            )
            ocr_match = False
            ocr_device_id = None
            if ocr_text:
                candidate = extract_device_id(ocr_text) or normalize_text(ocr_text)
                if candidate:
                    key = normalize_text(candidate)
                    key_nodash = key.replace("-", "")
                    if key in norm_device_ids:
                        ocr_match = True
                        ocr_device_id = norm_device_ids[key]
                    elif key_nodash in norm_device_ids:
                        ocr_match = True
                        ocr_device_id = norm_device_ids[key_nodash]

            if ocr_match and (
                best_ocr_match is None or ocr_conf > best_ocr_match["conf"]
            ):
                best_ocr_match = {"device_id": ocr_device_id, "conf": ocr_conf}

            reid_data = self.reid.embed(crop_img)
            matches = self.device_index.search(
                reid_data["embedding"], topk=self.config.get("max_device_matches", 5)
            )
            matches = self._filter_device_matches(matches, zone_id)
            LOGGER.info(
                "stage=reid request_id=%s det_id=%s matches=%d",
                request_id,
                det["det_id"],
                len(matches),
            )
            top_matches = [
                ReIdMatch(device_id=m[0]["device_id"], score=float(m[1])) for m in matches
            ]
            reid_top1 = float(top_matches[0].score) if top_matches else 0.0
            reid_top2 = float(top_matches[1].score) if len(top_matches) > 1 else 0.0
            reid_gap = reid_top1 - reid_top2
            if top_matches and (
                best_reid_match is None or reid_top1 > best_reid_match["score"]
            ):
                best_reid_match = {
                    "device_id": top_matches[0].device_id,
                    "score": reid_top1,
                    "gap": reid_gap,
                }

            gap_small = reid_gap < self.config["tau_gap"] if top_matches else False
            fused_score, breakdown = fuse_scores(
                zone_score=zone_score,
                det_conf=det_conf,
                reid_top1=reid_top1,
                ocr_conf=ocr_conf,
                ocr_match=ocr_match,
                gap_small=gap_small,
            )

            track_id = None
            track_stability = 0
            if det["det_id"] in track_map:
                track_id, track_stability = track_map[det["det_id"]]
                breakdown["track_stability"] = track_stability

            fused_device_id = None
            if ocr_match and ocr_device_id:
                fused_device_id = ocr_device_id
            elif top_matches:
                fused_device_id = top_matches[0].device_id

            if fused_score > best_fused_score:
                best_fused_score = fused_score
                if fused_device_id:
                    highlight_det_id = det["det_id"]

            detections.append(
                DetectionInfo(
                    det_id=det["det_id"],
                    cls=det["cls"],
                    conf=det_conf,
                    bbox=BBox(**bbox),
                    track_id=track_id,
                    roi=RoiInfo(crop_path=crop_path, mask_path=mask_path),
                    ocr=OcrInfo(
                        text=ocr_text,
                        conf=ocr_conf if ocr_text else None,
                        boxes=ocr_result.get("boxes"),
                    ),
                    reid=ReIdInfo(
                        embedding_type=reid_data["embedding_type"], top_matches=top_matches
                    ),
                    fused={
                        "device_id": fused_device_id,
                        "final_score": fused_score,
                        "score_breakdown": breakdown,
                    },
                )
            )

        decision_payload = decide_policy(
            quality=quality,
            zone_top1_score=zone_top1.score if zone_top1 else None,
            best_det_conf=best_det_conf,
            ocr_match=best_ocr_match,
            reid_match=best_reid_match,
            thresholds=self.config,
            highlight_det_id=highlight_det_id,
        )
        LOGGER.info(
            "stage=policy request_id=%s status=%s action=%s",
            request_id,
            decision_payload["status"],
            decision_payload["action"],
        )

        response = InferenceResponse(
            request_id=request_id,
            timestamp=timestamp,
            input=InputInfo(type=input_type, source=source, frame_index=frame_index),
            quality=QualityInfo(**quality),
            zone=ZoneInfo(
                candidates=zone_candidates,
                top1=zone_top1,
            ),
            detections=detections,
            decision=DecisionInfo(**decision_payload),
        )

        LOGGER.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "input_type": input_type,
                    "source": source,
                    "frame_index": frame_index,
                }
            )
        )
        return response
