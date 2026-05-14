import re
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS = {
    "tau_zone": 0.65,
    "tau_det": 0.40,
    "tau_ocr": 0.70,
    "tau_reid": 0.50,
    "tau_gap": 0.08,
}


def _model_to_dict(value: Any) -> Dict[str, Any]:
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


def extract_device_ids_from_text(text: str) -> List[str]:
    if not text:
        return []
    pattern = re.compile(r"\b[A-Z]{1,4}-?\d{2,6}\b", re.IGNORECASE)
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


def _quality_status(quality: Dict[str, Any]) -> str:
    flags = []
    if quality.get("is_blurry"):
        flags.append("blurry")
    if quality.get("is_low_light"):
        flags.append("low_light")
    glare = float(quality.get("glare_score") or 0.0)
    if glare >= 0.05:
        flags.append("glare")
    return "good" if not flags else ",".join(flags)


def _bbox_center(bbox: Dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox.get("x1") or 0.0) + float(bbox.get("x2") or 0.0)) / 2.0,
        (float(bbox.get("y1") or 0.0) + float(bbox.get("y2") or 0.0)) / 2.0,
    )


def _relative_location(det: Dict[str, Any], detections: List[Dict[str, Any]]) -> str:
    bbox = det.get("bbox") or {}
    cx, cy = _bbox_center(bbox)
    max_x = max(
        [float((item.get("bbox") or {}).get("x2") or 0.0) for item in detections]
        + [float(bbox.get("x2") or 1.0)]
    )
    max_y = max(
        [float((item.get("bbox") or {}).get("y2") or 0.0) for item in detections]
        + [float(bbox.get("y2") or 1.0)]
    )
    x_ratio = cx / max(max_x, 1.0)
    y_ratio = cy / max(max_y, 1.0)
    horizontal = "left" if x_ratio < 0.33 else "right" if x_ratio > 0.66 else "center"
    vertical = "top" if y_ratio < 0.33 else "bottom" if y_ratio > 0.66 else "middle"
    return f"{vertical}-{horizontal}"


def _compact_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    ocr = det.get("ocr") or {}
    reid = det.get("reid") or {}
    fused = det.get("fused") or {}
    top_matches = (reid.get("top_matches") or [])[:5]
    compact_matches = [
        {
            "device_id": item.get("device_id"),
            "score": _round(item.get("score")),
        }
        for item in top_matches
    ]
    return {
        "det_id": det.get("det_id"),
        "bbox": det.get("bbox"),
        "class_name": det.get("cls"),
        "detector_confidence": _round(det.get("conf")),
        "track_id": det.get("track_id"),
        "roi": det.get("roi") or {},
        "ocr": {
            "raw_text": ocr.get("text"),
            "confidence": _round(ocr.get("conf")),
            "parsed_device_ids": extract_device_ids_from_text(ocr.get("text") or ""),
        },
        "reid": {
            "top_k": compact_matches,
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


def _uncertainty_reasons(payload: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    reasons = []
    zone_top1 = ((payload.get("zone") or {}).get("top1")) or {}
    quality = payload.get("quality") or {}
    detections = payload.get("detections") or []

    if not zone_top1 or float(zone_top1.get("score") or 0.0) < float(thresholds["tau_zone"]):
        reasons.append("zone confidence is below threshold")
    if quality.get("is_blurry"):
        reasons.append("image appears blurry")
    if quality.get("is_low_light"):
        reasons.append("image appears low-light")

    best_det = max([float(det.get("conf") or 0.0) for det in detections] or [0.0])
    if not detections:
        reasons.append("no detector boxes are visible")
    elif best_det < float(thresholds["tau_det"]):
        reasons.append("best detector confidence is below threshold")

    selected = (payload.get("decision") or {}).get("selected_device")
    ocr_dets = [det for det in detections if ((det.get("ocr") or {}).get("text"))]
    strong_ocr = [
        det
        for det in ocr_dets
        if float(((det.get("ocr") or {}).get("conf")) or 0.0) >= float(thresholds["tau_ocr"])
    ]
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
            if best_reid < float(thresholds["tau_reid"]):
                reasons.append("best ReID score is below threshold")
            if any(
                len(matches) > 1
                and float(matches[0].get("score") or 0.0)
                - float(matches[1].get("score") or 0.0)
                < float(thresholds["tau_gap"])
                for matches in top_match_sets
            ):
                reasons.append("ReID top matches are too close")

    return list(dict.fromkeys(reasons))


def _feedback_summary(feedback_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    rows = feedback_rows or []
    return {
        "count": len(rows),
        "latest_type": rows[-1].get("feedback_type") if rows else None,
        "latest_data": rows[-1].get("data_json") if rows else None,
        "types": [row.get("feedback_type") for row in rows if row.get("feedback_type")],
    }


def build_evidence(
    response_payload: Any,
    selected_detection_id: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    feedback_rows: Optional[List[Dict[str, Any]]] = None,
    selected_det_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _model_to_dict(response_payload)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    decision = payload.get("decision") or {}
    quality = payload.get("quality") or {}
    zone = payload.get("zone") or {}
    detections = [_compact_detection(det) for det in payload.get("detections", [])]

    for det in detections:
        det["location"] = _relative_location(det, detections)

    selected_detection_id = selected_detection_id or selected_det_id
    if selected_detection_id is None:
        selected_detection_id = ((decision.get("ui_hints") or {}).get("highlight_det_id"))
    selected_detection = next(
        (det for det in detections if det.get("det_id") == selected_detection_id),
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
    return {
        "session_id": session_id or (payload.get("input") or {}).get("source"),
        "observation_id": observation_id or payload.get("request_id"),
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
        "zone_candidates": (zone.get("candidates") or [])[:5],
        "zone": {
            "top1": zone.get("top1"),
            "candidates": (zone.get("candidates") or [])[:5],
        },
        "detections": detections,
        "selected_detection": selected_detection,
        "fused_identity": selected_device or None,
        "decision": {
            "status": decision.get("status"),
            "action": decision.get("action"),
            "message": decision.get("message"),
            "selected_device": selected_device or None,
        },
        "accepted_or_deferred_reason": decision.get("message"),
        "uncertainty": {
            "reasons": _uncertainty_reasons(payload, thresholds),
        },
        "feedback": _feedback_summary(feedback_rows),
        "visible_device_ids": visible_device_ids,
    }


def _detection_mentions_device(det: Dict[str, Any], device_id: str) -> bool:
    device_id = device_id.upper()
    return any(
        candidate == device_id
        for candidate in [
            det.get("fusion", {}).get("device_id"),
            det.get("reid", {}).get("top1_device_id"),
            *det.get("ocr", {}).get("parsed_device_ids", []),
            *[
                item.get("device_id")
                for item in det.get("reid", {}).get("top_k", [])
                if item.get("device_id")
            ],
        ]
    )


def _evidence_used(*items: str) -> List[str]:
    return [item for item in items if item]


def answer_from_evidence(question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    q = (question or "").strip().lower()
    decision = evidence.get("decision") or {}
    zone_top1 = (evidence.get("zone") or {}).get("top1") or {}
    selected_device = decision.get("selected_device") or {}
    selected_detection = evidence.get("selected_detection")
    detections = evidence.get("detections") or []
    reasons = (evidence.get("uncertainty") or {}).get("reasons") or []
    selected_detection = selected_detection or None

    confidence = 0.0
    if selected_device:
        confidence = float(selected_device.get("score") or 0.0)
    elif selected_detection:
        confidence = float(selected_detection.get("fusion", {}).get("score") or 0.0)
    elif detections:
        confidence = max(float(det.get("detector_confidence") or 0.0) for det in detections)

    recommended_next_action = decision.get("message") or "Run inference before asking a question."
    evidence_used = _evidence_used("decision", "detections")
    uncertainty_reason = "; ".join(reasons) if reasons else ""

    if "tag" in q or "ocr" in q or "read" in q:
        texts = [
            str(det.get("ocr", {}).get("raw_text"))
            for det in detections
            if det.get("ocr", {}).get("raw_text")
        ]
        if texts:
            answer = "OCR read: " + " | ".join(texts[:5])
        else:
            answer = "I did not read a usable tag in the current evidence."
        evidence_used = _evidence_used("ocr", "detections")
    elif "candidate" in q or "top" in q:
        parts = []
        for det in detections:
            fused_id = det.get("fusion", {}).get("device_id")
            ocr_ids = det.get("ocr", {}).get("parsed_device_ids") or []
            matches = det.get("reid", {}).get("top_k", [])[:3]
            evidence_parts = []
            if fused_id:
                evidence_parts.append(f"fused={fused_id}")
            if ocr_ids:
                evidence_parts.append("ocr=" + ",".join(ocr_ids))
            if matches:
                evidence_parts.append(
                    "reid="
                    + ", ".join(f"{m.get('device_id')} ({m.get('score')})" for m in matches)
                )
            if evidence_parts:
                parts.append(f"{det.get('det_id')}: " + "; ".join(evidence_parts))
        answer = "Top candidates: " + " | ".join(parts) if parts else "No ReID candidates are available."
        evidence_used = _evidence_used("reid", "detections")
    elif "why" in q and ("uncertain" in q or "not sure" in q):
        answer = (
            "I am uncertain because " + "; ".join(reasons) + "."
            if reasons
            else "The current evidence is strong enough for the active decision."
        )
        evidence_used = _evidence_used("quality", "zone", "ocr", "reid", "decision")
    elif "what should" in q or "next" in q:
        answer = recommended_next_action
        evidence_used = _evidence_used("decision", "uncertainty")
    elif "where" in q:
        target_ids = extract_device_ids_from_text(question)
        zone_name = zone_top1.get("zone_name") or zone_top1.get("zone_id") or "unknown zone"
        if target_ids:
            target = target_ids[0]
            matching_det = next((det for det in detections if _detection_mentions_device(det, target)), None)
            if matching_det:
                answer = f"{target} appears in the {matching_det.get('location')} of the current frame, in zone {zone_name}."
                confidence = float(
                    matching_det.get("fusion", {}).get("score")
                    or matching_det.get("reid", {}).get("top1_score")
                    or confidence
                    or 0.0
                )
            else:
                answer = f"{target} is not confidently visible in the current frame. The current zone estimate is {zone_name}."
        else:
            answer = f"The current zone estimate is {zone_name}."
        evidence_used = _evidence_used("zone", "detections", "ocr", "reid")
    elif "which" in q and ("device" in q or "visible" in q):
        ids = evidence.get("visible_device_ids") or []
        accepted_id = (selected_device or {}).get("device_id")
        if accepted_id:
            other_ids = [item for item in ids if item != accepted_id]
            answer = f"Accepted visible device: {accepted_id}."
            if other_ids:
                answer += " Other evidence candidates: " + ", ".join(other_ids) + "."
        elif ids:
            answer = "Visible device candidates, not accepted identities: " + ", ".join(ids) + "."
        elif detections:
            classes = ", ".join(
                sorted({str(det.get("class_name")) for det in detections if det.get("class_name")})
            )
            answer = f"I see {len(detections)} detected object(s), but no enrolled device identity yet. Classes: {classes}."
        else:
            answer = "No devices are confidently visible in the current frame."
        evidence_used = _evidence_used("detections", "ocr", "reid", "fusion")
    elif "what" in q or "this" in q or "identify" in q:
        if selected_device:
            answer = f"This is likely {selected_device.get('device_id')}. ValveLens accepted this identity from the available OCR/ReID evidence."
            evidence_used = _evidence_used("decision", "ocr", "reid", "fusion")
        elif selected_detection:
            cls = selected_detection.get("class_name") or "device"
            fused_id = selected_detection.get("fusion", {}).get("device_id")
            if fused_id:
                answer = f"The selected {cls} candidate may be {fused_id}, but identity has not been accepted yet."
            else:
                answer = f"The selected object appears to be a {cls}, but identity is uncertain."
            evidence_used = _evidence_used("selected_detection", "detector", "ocr", "reid", "fusion")
        elif detections:
            answer = "Please select a detected object first so I can answer about that device."
            evidence_used = _evidence_used("detections")
        else:
            answer = "I do not have a detected device to identify yet."
    else:
        answer = "I can answer from the current ValveLens evidence: zone, detections, OCR, ReID, fusion, quality, and decision state."

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "mode": "rule_based",
        "evidence_used": evidence_used,
        "recommended_next_action": recommended_next_action,
        "uncertainty_reason": uncertainty_reason,
        "evidence": evidence,
    }
