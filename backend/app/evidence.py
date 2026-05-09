import re
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS = {
    "tau_zone": 0.65,
    "tau_det": 0.40,
    "tau_ocr": 0.70,
    "tau_reid": 0.50,
    "tau_gap": 0.08,
}


def _model_to_dict(value: Any) -> Dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _round(value: Any, digits: int = 4) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _device_ids_from_text(text: str) -> List[str]:
    if not text:
        return []
    pattern = re.compile(r"\b[A-Z]{1,3}-?\d{2,6}\b", re.IGNORECASE)
    ids: List[str] = []
    seen = set()
    for match in pattern.finditer(text):
        raw = match.group(0).upper()
        if "-" not in raw:
            split = re.split(r"(\d+)", raw, maxsplit=1)
            if len(split) >= 2 and split[0] and split[1]:
                raw = f"{split[0]}-{split[1]}"
        if raw not in seen:
            seen.add(raw)
            ids.append(raw)
    return ids


def _quality_status(quality: Dict) -> str:
    flags = []
    if quality.get("is_blurry"):
        flags.append("blurry")
    if quality.get("is_low_light"):
        flags.append("low_light")
    glare = float(quality.get("glare_score") or 0.0)
    if glare >= 0.05:
        flags.append("glare")
    return "good" if not flags else ",".join(flags)


def _compact_detection(det: Dict) -> Dict:
    ocr = det.get("ocr") or {}
    reid = det.get("reid") or {}
    fused = det.get("fused") or {}
    top_matches = (reid.get("top_matches") or [])[:3]
    compact_matches = [
        {
            "device_id": item.get("device_id"),
            "score": _round(item.get("score")),
        }
        for item in top_matches
    ]
    return {
        "det_id": det.get("det_id"),
        "class": det.get("cls"),
        "confidence": _round(det.get("conf")),
        "track_id": det.get("track_id"),
        "bbox": det.get("bbox"),
        "ocr": {
            "text": ocr.get("text"),
            "confidence": _round(ocr.get("conf")),
            "parsed_device_ids": _device_ids_from_text(ocr.get("text") or ""),
        },
        "reid": {
            "top_matches": compact_matches,
            "top1_device_id": compact_matches[0]["device_id"] if compact_matches else None,
            "top1_score": compact_matches[0]["score"] if compact_matches else None,
            "gap": _round(
                (compact_matches[0]["score"] or 0.0) - (compact_matches[1]["score"] or 0.0)
            )
            if len(compact_matches) > 1
            else None,
        },
        "fusion": {
            "device_id": fused.get("device_id"),
            "score": _round(fused.get("final_score")),
            "breakdown": fused.get("score_breakdown") or {},
        },
    }


def _uncertainty_reasons(payload: Dict, thresholds: Dict) -> List[str]:
    reasons = []
    zone_top1 = ((payload.get("zone") or {}).get("top1")) or {}
    quality = payload.get("quality") or {}
    detections = payload.get("detections") or []

    if not zone_top1 or float(zone_top1.get("score") or 0.0) < thresholds["tau_zone"]:
        reasons.append("zone confidence is below threshold")
    if quality.get("is_blurry"):
        reasons.append("image appears blurry")
    if quality.get("is_low_light"):
        reasons.append("image appears low-light")

    best_det = max([float(det.get("conf") or 0.0) for det in detections] or [0.0])
    if not detections:
        reasons.append("no detector boxes are visible")
    elif best_det < thresholds["tau_det"]:
        reasons.append("best detector confidence is below threshold")

    ocr_dets = [det for det in detections if ((det.get("ocr") or {}).get("text"))]
    strong_ocr = [
        det
        for det in ocr_dets
        if float(((det.get("ocr") or {}).get("conf")) or 0.0) >= thresholds["tau_ocr"]
    ]
    selected = (payload.get("decision") or {}).get("selected_device")
    if not selected:
        if not ocr_dets:
            reasons.append("no readable enrolled OCR tag was found")
        elif not strong_ocr:
            reasons.append("OCR confidence is below threshold")
        else:
            reasons.append("OCR text did not produce an accepted enrolled device")

    top_match_sets = [
        ((det.get("reid") or {}).get("top_matches") or [])
        for det in detections
        if ((det.get("reid") or {}).get("top_matches") or [])
    ]
    if not selected:
        if not top_match_sets:
            reasons.append("device ReID index returned no matches")
        else:
            best_reid = max(float(matches[0].get("score") or 0.0) for matches in top_match_sets)
            if best_reid < thresholds["tau_reid"]:
                reasons.append("best ReID score is below threshold")
            if any(
                len(matches) > 1
                and float(matches[0].get("score") or 0.0) - float(matches[1].get("score") or 0.0)
                < thresholds["tau_gap"]
                for matches in top_match_sets
            ):
                reasons.append("ReID top matches are too close")

    return list(dict.fromkeys(reasons))


def build_evidence(
    response_payload: Any,
    selected_det_id: Optional[str] = None,
    thresholds: Optional[Dict] = None,
) -> Dict:
    payload = _model_to_dict(response_payload)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    decision = payload.get("decision") or {}
    quality = payload.get("quality") or {}
    zone = payload.get("zone") or {}
    detections = [_compact_detection(det) for det in payload.get("detections", [])]

    if selected_det_id is None:
        selected_det_id = ((decision.get("ui_hints") or {}).get("highlight_det_id"))
    selected_detection = next(
        (det for det in detections if det.get("det_id") == selected_det_id),
        None,
    )

    visible_device_ids = []
    for det in detections:
        candidates = [
            det["fusion"].get("device_id"),
            det["reid"].get("top1_device_id"),
            *det["ocr"].get("parsed_device_ids", []),
        ]
        for device_id in candidates:
            if device_id and device_id not in visible_device_ids:
                visible_device_ids.append(device_id)

    selected_device = decision.get("selected_device") or {}
    evidence = {
        "request_id": payload.get("request_id"),
        "input": payload.get("input") or {},
        "quality": {
            "status": _quality_status(quality),
            "blur_score": _round(quality.get("blur_score")),
            "brightness": _round(quality.get("brightness")),
            "glare_score": _round(quality.get("glare_score")),
            "is_blurry": bool(quality.get("is_blurry")),
            "is_low_light": bool(quality.get("is_low_light")),
        },
        "zone": {
            "top1": zone.get("top1"),
            "candidates": (zone.get("candidates") or [])[:5],
        },
        "detections": detections,
        "selected_detection": selected_detection,
        "decision": {
            "status": decision.get("status"),
            "action": decision.get("action"),
            "message": decision.get("message"),
            "selected_device": selected_device or None,
        },
        "uncertainty": {
            "reasons": _uncertainty_reasons(payload, thresholds),
        },
        "visible_device_ids": visible_device_ids,
    }
    return evidence


def answer_from_evidence(question: str, evidence: Dict) -> Dict:
    q = (question or "").strip().lower()
    decision = evidence.get("decision") or {}
    zone_top1 = (evidence.get("zone") or {}).get("top1") or {}
    selected_device = decision.get("selected_device") or {}
    selected_detection = evidence.get("selected_detection") or {}
    detections = evidence.get("detections") or []
    reasons = (evidence.get("uncertainty") or {}).get("reasons") or []

    confidence = 0.0
    if selected_device:
        confidence = float(selected_device.get("score") or 0.0)
    elif selected_detection:
        confidence = float(selected_detection.get("fusion", {}).get("score") or 0.0)
    elif detections:
        confidence = max(float(det.get("confidence") or 0.0) for det in detections)

    action = decision.get("action") or "NONE"
    next_action = decision.get("message") or "Run inference before asking a question."

    if "why" in q and ("uncertain" in q or "not sure" in q):
        if reasons:
            answer = "I am uncertain because " + "; ".join(reasons) + "."
        else:
            answer = "The current evidence is strong enough for the active decision."
    elif "what should" in q or "next" in q:
        answer = next_action
    elif "where" in q:
        target_ids = _device_ids_from_text(question)
        zone_name = zone_top1.get("zone_name") or zone_top1.get("zone_id") or "unknown zone"
        if target_ids:
            target = target_ids[0]
            if target in evidence.get("visible_device_ids", []):
                answer = f"{target} appears visible in {zone_name}."
            else:
                answer = f"I do not see strong evidence for {target} in the current frame. The current zone estimate is {zone_name}."
        else:
            answer = f"The current zone estimate is {zone_name}."
    elif "which" in q and ("device" in q or "visible" in q):
        ids = evidence.get("visible_device_ids") or []
        if ids:
            answer = "Visible device candidates: " + ", ".join(ids) + "."
        elif detections:
            classes = ", ".join(sorted({str(det.get("class")) for det in detections if det.get("class")}))
            answer = f"I see {len(detections)} detected object(s), but no enrolled device identity yet. Classes: {classes}."
        else:
            answer = "No devices are confidently visible in the current frame."
    elif "what" in q or "this" in q or "identify" in q:
        if selected_device:
            answer = f"This is most likely {selected_device.get('device_id')} based on ValveLens evidence."
        elif selected_detection:
            cls = selected_detection.get("class") or "device"
            fused_id = selected_detection.get("fusion", {}).get("device_id")
            if fused_id:
                answer = f"The selected {cls} candidate looks like {fused_id}, but it has not been accepted yet."
            else:
                answer = f"The selected object appears to be a {cls}, but identity is not resolved yet."
        elif detections:
            best = max(detections, key=lambda det: float(det.get("confidence") or 0.0))
            answer = f"I see a {best.get('class')} candidate, but identity is not resolved yet."
        else:
            answer = "I do not have a detected device to identify yet."
    else:
        answer = "I can answer from the current ValveLens evidence: zone, detections, OCR, ReID, fusion, quality, and decision state."

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "recommended_next_action": action,
        "evidence": evidence,
    }
