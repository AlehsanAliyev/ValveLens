from typing import Dict, Optional


def decide_policy(
    quality: Dict,
    zone_top1_score: Optional[float],
    best_det_conf: float,
    ocr_match: Optional[Dict],
    reid_match: Optional[Dict],
    thresholds: Dict,
    highlight_det_id: Optional[str],
) -> Dict:
    tau_zone = thresholds["tau_zone"]
    tau_det = thresholds["tau_det"]
    tau_ocr = thresholds["tau_ocr"]
    tau_reid = thresholds["tau_reid"]
    tau_gap = thresholds["tau_gap"]

    if zone_top1_score is None or zone_top1_score < tau_zone:
        return {
            "status": "UNCERTAIN",
            "selected_device": None,
            "action": "ASK_WIDER_VIEW",
            "message": "Wider view needed. Step back and capture more of the area.",
            "ui_hints": {
                "highlight_det_id": highlight_det_id,
                "suggested_moves": ["step back", "pan slowly left-right"],
            },
        }

    if quality["is_blurry"] or quality["is_low_light"]:
        return {
            "status": "UNCERTAIN",
            "selected_device": None,
            "action": "ASK_VIEWPOINT",
            "message": "Image is unclear. Move closer, hold steady, or tilt to reduce glare.",
            "ui_hints": {
                "highlight_det_id": highlight_det_id,
                "suggested_moves": ["move closer", "hold steady", "tilt down 15 deg"],
            },
        }

    if best_det_conf < tau_det:
        return {
            "status": "UNCERTAIN",
            "selected_device": None,
            "action": "ASK_VIEWPOINT",
            "message": "Point camera towards the device; try moving closer.",
            "ui_hints": {
                "highlight_det_id": highlight_det_id,
                "suggested_moves": ["move closer", "pan slightly"],
            },
        }

    if ocr_match and ocr_match.get("conf", 0.0) >= tau_ocr:
        return {
            "status": "ACCEPTED",
            "selected_device": {
                "device_id": ocr_match["device_id"],
                "score": ocr_match["conf"],
            },
            "action": "NONE",
            "message": f"Identified device {ocr_match['device_id']} via OCR.",
            "ui_hints": {"highlight_det_id": highlight_det_id},
        }

    if reid_match and reid_match.get("score", 0.0) >= tau_reid:
        gap_ok = reid_match.get("gap", 0.0) >= tau_gap
        if gap_ok:
            return {
                "status": "ACCEPTED",
                "selected_device": {
                    "device_id": reid_match["device_id"],
                    "score": reid_match["score"],
                },
                "action": "NONE",
                "message": f"Identified device {reid_match['device_id']} via visual match.",
                "ui_hints": {"highlight_det_id": highlight_det_id},
            }

    return {
        "status": "UNCERTAIN",
        "selected_device": None,
        "action": "ASK_TAP",
        "message": "Tap the correct device in the image.",
        "ui_hints": {"highlight_det_id": highlight_det_id},
    }
