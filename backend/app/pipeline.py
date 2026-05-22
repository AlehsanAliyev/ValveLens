import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import yaml
from PIL import Image

from app.db import fetch_device_ids, fetch_latest_session_feedback_device
from app.detector import Detector
from app.embeddings import Embedder
from app.faiss_store import FaissIndex
from app.fusion import fuse_scores
from app.ocr import OCRReader, match_enrolled_device_id
from app.policy import decide_policy
from app.quality import compute_quality
from app.reid import ReIDEmbedder, group_matches_by_device
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
from app.zone_localizer import ZoneLocalizer

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


def _expanded_crop(frame_bgr: np.ndarray, bbox: Dict, expand_ratio: float) -> Image.Image:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    pad_x = int(box_w * max(0.0, expand_ratio))
    pad_y = int(box_h * max(0.0, expand_ratio))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(width - 1, x2 + pad_x)
    ey2 = min(height - 1, y2 + pad_y)
    crop = frame_bgr[ey1 : ey2 + 1, ex1 : ex2 + 1]
    return Image.fromarray(crop[:, :, ::-1])


class InferencePipeline:
    def __init__(self) -> None:
        self.config = load_config()
        self.embedder = Embedder()
        self.zone_index = FaissIndex("zones", self.embedder.dim)
        self.device_index = FaissIndex("devices", self.embedder.dim)
        self.detector = Detector(self.config.get("detector_model", "yolov8n.pt"))
        self.segmenter = Segmenter()
        self.ocr = OCRReader(
            enable_preprocessing=bool(self.config.get("ocr_preprocess", True)),
            resize_factor=float(self.config.get("ocr_resize_factor", 2.0)),
        )
        self.reid = ReIDEmbedder(self.embedder)
        self.tracker = TrackerManager(
            iou_threshold=float(self.config.get("tracker_iou_threshold", 0.3)),
            max_missed=int(self.config.get("tracker_max_missed", 5)),
            smoothing_window=int(self.config.get("tracker_smoothing_window", 5)),
        )
        self.zone_localizer = ZoneLocalizer(self.embedder, self.zone_index)

    def _zone_candidates(
        self, image: Image.Image
    ) -> Tuple[List[ZoneCandidate], Optional[ZoneTop1], List[Dict]]:
        search_topk = int(self.config.get("zone_search_topk", 20))
        localization = self.zone_localizer.localize(
            image=image,
            topk_keyframes=search_topk,
            topk_zones=5,
            aggregate_mode=self.config.get("zone_aggregate_mode", "sum"),
        )
        candidates = [ZoneCandidate(**item) for item in localization["zone_candidates"]]
        top1 = ZoneTop1(**localization["zone_top1"]) if localization["zone_top1"] else None
        return candidates, top1, localization["top_k_keyframes"]

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
        zone_candidates, zone_top1, top_k_keyframes = self._zone_candidates(pil_image)
        zone_score = zone_top1.score if zone_top1 else 0.0
        zone_id = zone_top1.zone_id if zone_top1 else None
        LOGGER.info(
            "stage=zone request_id=%s top1=%s score=%.3f candidates=%d",
            request_id,
            zone_id,
            zone_score,
            len(zone_candidates),
        )
        if top_k_keyframes:
            LOGGER.info(
                "stage=zone_keyframes request_id=%s top_match_zone=%s top_match_score=%.3f",
                request_id,
                top_k_keyframes[0].get("zone_id"),
                float(top_k_keyframes[0].get("score", 0.0)),
            )

        detections_raw = self.detector.detect(frame_bgr, conf_thres=self.config.get("tau_det", 0.4))
        LOGGER.info(
            "stage=detect request_id=%s detections=%d",
            request_id,
            len(detections_raw),
        )
        track_map: Dict[str, Dict] = {}
        if session_id:
            track_map = self.tracker.update(session_id, detections_raw)
            LOGGER.info(
                "stage=track request_id=%s session_id=%s assigned=%d",
                request_id,
                session_id,
                len(track_map),
            )

        device_ids = fetch_device_ids()

        detections: List[DetectionInfo] = []
        best_det_conf = 0.0
        highlight_det_id = None
        best_ocr_match = None
        best_reid_match = None
        best_fused_score = -1.0
        session_device_id = None
        session_feedback_type = None
        if session_id:
            latest_feedback = fetch_latest_session_feedback_device(session_id)
            if latest_feedback:
                feedback_data = latest_feedback.get("data_json") or {}
                if isinstance(feedback_data, str):
                    try:
                        feedback_data = json.loads(feedback_data)
                    except Exception:
                        feedback_data = {}
                session_device_id = feedback_data.get("device_id")
                session_feedback_type = latest_feedback.get("feedback_type")
                if session_device_id:
                    LOGGER.info(
                        "stage=session_hint request_id=%s session_id=%s device_id=%s feedback_type=%s",
                        request_id,
                        session_id,
                        session_device_id,
                        session_feedback_type,
                    )
        best_session_match: Optional[Dict[str, object]] = None

        for det in detections_raw:
            bbox = det["bbox"]
            det_conf = float(det["conf"])
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

            ocr_crop_img = crop_img
            ocr_expand_ratio = float(self.config.get("ocr_expand_ratio", 0.0))
            if ocr_expand_ratio > 0:
                ocr_crop_img = _expanded_crop(frame_bgr, bbox, ocr_expand_ratio)
            ocr_result = self.ocr.read(ocr_crop_img)
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
                ocr_device_id = match_enrolled_device_id(ocr_text, device_ids)
                ocr_match = bool(ocr_device_id)
                if ocr_match:
                    LOGGER.info(
                        "stage=ocr_match request_id=%s det_id=%s device_id=%s",
                        request_id,
                        det["det_id"],
                        ocr_device_id,
                    )

            reid_data = self.reid.embed(crop_img)
            max_device_matches = int(self.config.get("max_device_matches", 5))
            matches = self.device_index.search(
                reid_data["embedding"], topk=max(max_device_matches * 4, 20)
            )
            matches = self._filter_device_matches(matches, zone_id)
            grouped_matches = group_matches_by_device(matches)
            LOGGER.info(
                "stage=reid request_id=%s det_id=%s matches=%d",
                request_id,
                det["det_id"],
                len(matches),
            )
            raw_top_matches = [
                ReIdMatch(device_id=m[0]["device_id"], score=float(m[1]))
                for m in matches[:max_device_matches]
            ]
            top_matches = [
                ReIdMatch(
                    device_id=m["device_id"],
                    score=float(m["best_score"]),
                    mean_score=float(m["mean_score"]),
                    ref_count=int(m["ref_count"]),
                    best_reference_image=m.get("best_reference_image"),
                )
                for m in grouped_matches[:max_device_matches]
            ]
            reid_top1 = float(top_matches[0].score) if top_matches else 0.0
            reid_top2 = float(top_matches[1].score) if len(top_matches) > 1 else 0.0
            reid_gap = reid_top1 - reid_top2
            candidate_device_id = None
            if ocr_match and ocr_device_id:
                candidate_device_id = ocr_device_id
            elif top_matches:
                candidate_device_id = top_matches[0].device_id

            track_id = None
            track_stability = 0
            smoothed_det_conf = det_conf
            smoothed_ocr_conf = ocr_conf
            smoothed_reid_top1 = reid_top1
            smoothed_selected_device_id = candidate_device_id
            if det["det_id"] in track_map:
                track_info = track_map[det["det_id"]]
                track_id = str(track_info.get("track_id"))
                smoothed = self.tracker.update_signals(
                    session_id=session_id or "",
                    track_id=track_id,
                    det_conf=det_conf,
                    ocr_conf=ocr_conf,
                    reid_top1=reid_top1,
                    selected_device_id=candidate_device_id,
                )
                smoothed_det_conf = float(smoothed.get("smoothed_det_conf", det_conf))
                smoothed_ocr_conf = float(smoothed.get("smoothed_ocr_conf", ocr_conf))
                smoothed_reid_top1 = float(smoothed.get("smoothed_reid_top1", reid_top1))
                smoothed_selected_device_id = smoothed.get("smoothed_selected_device_id")
                track_stability = int(smoothed.get("track_stability", 0))
                LOGGER.info(
                    "stage=smoothing request_id=%s det_id=%s track_id=%s stability=%d",
                    request_id,
                    det["det_id"],
                    track_id,
                    track_stability,
                )

            if smoothed_det_conf > best_det_conf:
                best_det_conf = smoothed_det_conf
                highlight_det_id = det["det_id"]

            if ocr_match and (
                best_ocr_match is None or smoothed_ocr_conf > best_ocr_match["conf"]
            ):
                best_ocr_match = {"device_id": ocr_device_id, "conf": smoothed_ocr_conf}

            if top_matches and (
                best_reid_match is None or smoothed_reid_top1 > best_reid_match["score"]
            ):
                best_reid_match = {
                    "device_id": top_matches[0].device_id,
                    "score": smoothed_reid_top1,
                    "gap": reid_gap,
                }

            gap_small = reid_gap < self.config["tau_gap"] if top_matches else False
            session_prior_match = False
            if session_device_id:
                top_match_ids = {match.device_id for match in top_matches}
                if ocr_device_id == session_device_id or session_device_id in top_match_ids:
                    session_prior_match = True
                    if not candidate_device_id:
                        candidate_device_id = session_device_id
            fused_score, breakdown = fuse_scores(
                zone_score=zone_score,
                det_conf=smoothed_det_conf,
                reid_top1=smoothed_reid_top1,
                ocr_conf=smoothed_ocr_conf,
                ocr_match=ocr_match,
                gap_small=gap_small,
                session_prior_match=session_prior_match,
            )
            breakdown["track_stability"] = track_stability
            breakdown["det_conf_raw"] = det_conf
            breakdown["ocr_conf_raw"] = ocr_conf
            breakdown["reid_top1_raw"] = reid_top1
            if session_device_id:
                breakdown["session_device_id"] = session_device_id
                breakdown["session_prior_match"] = session_prior_match

            fused_device_id = str(smoothed_selected_device_id) if smoothed_selected_device_id else None
            if session_prior_match and session_device_id:
                fused_device_id = session_device_id

            if fused_score > best_fused_score:
                best_fused_score = fused_score
                if fused_device_id:
                    highlight_det_id = det["det_id"]
            if session_prior_match and session_device_id:
                current_session_match = {
                    "device_id": session_device_id,
                    "det_id": det["det_id"],
                    "score": fused_score,
                }
                if best_session_match is None or float(current_session_match["score"]) > float(best_session_match["score"]):
                    best_session_match = current_session_match

            detections.append(
                DetectionInfo(
                    det_id=det["det_id"],
                    cls=det["cls"],
                    class_id=det.get("class_id"),
                    class_name=det.get("class_name") or det.get("cls"),
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
                        embedding_type=reid_data["embedding_type"],
                        top_matches=top_matches,
                        raw_top_matches=raw_top_matches,
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
        if (
            decision_payload["status"] == "UNCERTAIN"
            and session_feedback_type == "confirm"
            and session_device_id
            and best_session_match
            and zone_top1 is not None
            and zone_top1.score >= self.config["tau_zone"]
            and not quality["is_blurry"]
            and not quality["is_low_light"]
            and best_det_conf >= self.config["tau_det"]
            and float(best_session_match["score"]) >= 0.60
        ):
            decision_payload = {
                "status": "ACCEPTED",
                "selected_device": {
                    "device_id": session_device_id,
                    "score": float(best_session_match["score"]),
                },
                "action": "NONE",
                "message": f"Maintaining confirmed session device {session_device_id}.",
                "ui_hints": {"highlight_det_id": best_session_match["det_id"]},
            }
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
                candidates=zone_candidates[:5],
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
