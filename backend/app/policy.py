from typing import Dict, Optional


def _quality_reasons(quality: Dict) -> list[str]:
    reasons: list[str] = []
    if quality.get("is_blurry"):
        blur = quality.get("blur_score")
        reasons.append(
            f"image quality low: blur score {float(blur):.2f} below threshold"
            if blur is not None
            else "image quality low: blur score below threshold"
        )
    if quality.get("is_low_light"):
        brightness = quality.get("brightness")
        reasons.append(
            f"image quality low: brightness {float(brightness):.2f} below threshold"
            if brightness is not None
            else "image quality low: brightness below threshold"
        )
    return reasons


def _payload(
    status: str,
    selected_device: Optional[Dict],
    action: str,
    message: str,
    highlight_det_id: Optional[str],
    reasons: list[str],
    suggested_moves: Optional[list[str]] = None,
    next_action: Optional[str] = None,
) -> Dict:
    return {
        "status": status,
        "selected_device": selected_device,
        "action": action,
        "message": message,
        "reasons": reasons,
        "next_action": next_action or message,
        "ui_hints": {
            "highlight_det_id": highlight_det_id,
            "suggested_moves": suggested_moves or [],
        },
    }


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
        return _payload(
            status="UNCERTAIN",
            selected_device=None,
            action="ASK_WIDER_VIEW",
            message="Wider view needed. Step back and capture more of the area.",
            highlight_det_id=highlight_det_id,
            reasons=["zone confidence below threshold"],
            suggested_moves=["step back", "pan slowly left-right"],
            next_action="move wider and retry",
        )

    if quality["is_blurry"] or quality["is_low_light"]:
        reasons = _quality_reasons(quality)
        if quality.get("is_blurry") and quality.get("is_low_light"):
            message = "Image quality is low from blur and low light."
        elif quality.get("is_blurry"):
            message = "Blur score is below threshold; hold steady or move closer."
        else:
            message = "Brightness is below threshold; improve lighting or move closer."
        return _payload(
            status="UNCERTAIN",
            selected_device=None,
            action="ASK_VIEWPOINT",
            message=message,
            highlight_det_id=highlight_det_id,
            reasons=reasons,
            suggested_moves=["move closer", "hold steady", "reduce glare"],
            next_action="improve viewpoint or lighting",
        )

    if best_det_conf < tau_det:
        return _payload(
            status="UNCERTAIN",
            selected_device=None,
            action="ASK_VIEWPOINT",
            message="Detector confidence is low; point camera toward the object and try moving closer.",
            highlight_det_id=highlight_det_id,
            reasons=["detector confidence low"],
            suggested_moves=["move closer", "pan slightly", "try VLM description"],
            next_action="move closer or try VLM description",
        )

    if ocr_match and ocr_match.get("conf", 0.0) >= tau_ocr:
        return _payload(
            status="ACCEPTED",
            selected_device={
                "device_id": ocr_match["device_id"],
                "score": ocr_match["conf"],
            },
            action="NONE",
            message=f"Identified device {ocr_match['device_id']} via OCR.",
            highlight_det_id=highlight_det_id,
            reasons=["OCR tag matched enrolled device"],
            next_action="none",
        )

    if reid_match and reid_match.get("score", 0.0) >= tau_reid:
        gap_ok = reid_match.get("gap", 0.0) >= tau_gap
        if gap_ok:
            return _payload(
                status="ACCEPTED",
                selected_device={
                    "device_id": reid_match["device_id"],
                    "score": reid_match["score"],
                },
                action="NONE",
                message=f"Identified device {reid_match['device_id']} via visual match.",
                highlight_det_id=highlight_det_id,
                reasons=["ReID unique device score above threshold"],
                next_action="none",
            )

    reasons = ["OCR tag missing or below threshold"]
    if reid_match and reid_match.get("score", 0.0) >= tau_reid:
        reasons.append("ReID identity margin low")
    else:
        reasons.append("ReID score below threshold")
    return _payload(
        status="UNCERTAIN",
        selected_device=None,
        action="ASK_TAP",
        message="Identity is uncertain. Tap-select the correct object or try VLM description.",
        highlight_det_id=highlight_det_id,
        reasons=reasons,
        suggested_moves=["show the tag", "tap-select the correct object", "try VLM description"],
        next_action="tap-select the correct object or ask for a VLM description",
    )
