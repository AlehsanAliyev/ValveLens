import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.evidence import answer_from_evidence
from app.quality import compute_quality
from app.schemas import (
    BBox,
    DecisionInfo,
    DetectionInfo,
    FusedInfo,
    InferenceResponse,
    InputInfo,
    OcrInfo,
    QualityInfo,
    ReIdInfo,
    ReIdMatch,
    RoiInfo,
    UiHints,
    ZoneCandidate,
    ZoneInfo,
    ZoneTop1,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEPINFRA_ENDPOINT = "https://api.deepinfra.com/v1/openai"
DEFAULT_DEEPINFRA_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
VLM_DETECTION_CLASSES = {"valve", "gauge", "pipe", "flange", "tag", "panel", "tank", "pump", "unknown"}


SYSTEM_PROMPT = (
    "You are an assistant for ValveLens. Use the image for general visual "
    "description and the ValveLens structured evidence for exact system "
    "claims. If a tag is clearly readable in the image, answer with that "
    "visible tag directly. Do not invent hidden or unreadable identities. "
    "Keep answers practical and concise."
)


def _load_env_file() -> None:
    """Load repo .env values for local demo commands without printing secrets."""
    for env_path in (REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _deepinfra_endpoint() -> str:
    endpoint = (os.environ.get("DEEPINFRA_ENDPOINT") or DEFAULT_DEEPINFRA_ENDPOINT).rstrip("/")
    if endpoint.endswith("/v1"):
        return f"{endpoint}/openai"
    return endpoint


def _assistant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    _load_env_file()
    assistant = config.get("assistant") or {}
    provider = (
        os.environ.get("VALVELENS_VLM_PROVIDER")
        or assistant.get("provider")
        or "openai"
    ).lower()
    model = assistant.get("model") or os.environ.get("VALVELENS_VLM_MODEL")
    if not model and provider == "deepinfra":
        model = DEFAULT_DEEPINFRA_MODEL
    enable_env = _env_bool("VALVELENS_ENABLE_VLM")
    return {
        "enable_vlm": bool(assistant.get("enable_vlm", False)) if enable_env is None else enable_env,
        "provider": provider,
        "model": model,
        "include_image": bool(assistant.get("include_image", True)),
        "max_tokens": int(assistant.get("max_tokens", 300)),
        "use_rule_fallback": bool(assistant.get("use_rule_fallback", True)),
        "endpoint": _deepinfra_endpoint(),
    }


def vlm_available(config: Dict[str, Any], force: bool = False) -> tuple[bool, str]:
    assistant = _assistant_config(config)
    if not assistant["enable_vlm"] and not force:
        return False, "VLM disabled in config"
    if not assistant["model"]:
        return False, "assistant.model is not configured"
    provider = assistant["provider"]
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not configured"
    if provider == "deepinfra" and not (
        os.environ.get("DEEPINFRA_TOKEN") or os.environ.get("DEEPINFRA_API_KEY")
    ):
        return False, "DEEPINFRA_API_KEY is not configured"
    try:
        from openai import OpenAI  # noqa: F401
    except Exception:
        return False, "openai package is not installed"
    return True, "available"


def _image_data_url(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _client_for_provider(provider: str):
    from openai import OpenAI

    if provider == "deepinfra":
        token = os.environ.get("DEEPINFRA_TOKEN") or os.environ.get("DEEPINFRA_API_KEY")
        return OpenAI(api_key=token, base_url=_deepinfra_endpoint())
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _vlm_prompt(question: str, evidence: Dict[str, Any]) -> str:
    visible_ids = evidence.get("visible_device_ids") or []
    decision = evidence.get("decision") or {}
    quality = evidence.get("quality") or {}
    classes = sorted(
        {
            str(det.get("class_name") or det.get("display_class"))
            for det in evidence.get("detections", [])
            if det.get("class_name") or det.get("display_class")
        }
    )
    question_lower = question.lower()
    wants_evidence_details = any(
        word in question_lower
        for word in [
            "why",
            "uncertain",
            "fail",
            "failed",
            "evidence",
            "detector",
            "ocr",
            "reid",
            "quality",
            "confidence",
        ]
    )
    answer_instruction = (
        "The user is asking for system/evidence details. Answer with two concise sections: "
        "Visual observation, then ValveLens evidence. Explain only the relevant detector/OCR/ReID/quality evidence."
        if wants_evidence_details
        else "The user is asking a normal visual question. Answer directly in one short paragraph. "
        "Do not create numbered sections. Do not include a ValveLens evidence/debug section. "
        "Do not mention detector, OCR, ReID, pipeline failure, or raw quality metrics. "
        "Do not comment on blur, lighting, or image quality unless the user asks about quality or uncertainty. "
        "If a visible tag is readable, say the object appears tagged with that text."
    )
    return (
        f"Question: {question}\n"
        "ValveLens evidence summary:\n"
        f"- decision: {decision.get('status')} / {decision.get('message')}\n"
        f"- registered IDs supported by OCR/ReID/fusion: {', '.join(visible_ids) or 'none'}\n"
        f"- detector classes confirmed by YOLO: {', '.join(classes) or 'none'}\n"
        f"- automated quality metrics: blur_score={quality.get('blur_score')} brightness={quality.get('brightness')} "
        f"glare_score={quality.get('glare_score')}\n\n"
        f"Answer instruction: {answer_instruction}\n"
        "When system/evidence details are requested: if YOLO has no detector classes, say "
        "'the detector did not confirm a class'; do not say 'no class/object is visible'. "
        "If the automated quality metric is low, mention it as a pipeline metric only, and do not "
        "call the image blurry or dark unless that is visually obvious. "
        "You may transcribe a clearly visible tag as 'the tag appears to read ...', but do not state "
        "that this is a confirmed registered device identity unless it appears in the ValveLens evidence summary."
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("VLM response did not contain a JSON object")


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except Exception:
        return default
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _clamp_bbox(raw: Dict[str, Any], width: int, height: int) -> BBox:
    x1 = int(round(float(raw.get("x1", 0))))
    y1 = int(round(float(raw.get("y1", 0))))
    x2 = int(round(float(raw.get("x2", width - 1))))
    y2 = int(round(float(raw.get("y2", height - 1))))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _vlm_inference_prompt(width: int, height: int) -> str:
    allowed = ", ".join(sorted(VLM_DETECTION_CLASSES))
    return (
        "You are running ValveLens in VLM-only demo mode. Inspect the image and return only JSON, "
        "with no markdown. This is a demo estimate, not validated YOLO/OCR/ReID output.\n\n"
        f"Image size: width={width}, height={height}. Bounding boxes must be pixel coordinates in this image. "
        "Use tight boxes around the visible object or tag, not the entire image.\n"
        f"Allowed detection classes: {allowed}.\n\n"
        "Return this schema:\n"
        "{\n"
        '  "scene_description": "short visual description",\n'
        '  "zone_candidates": [{"zone_id": "vlm_zone_1", "zone_name": "Valve manifold area", "score": 0.82}],\n'
        '  "image_quality": {"brightness": 0.6, "contrast": 0.5, "blur_score": 0.7, "glare_score": 0.0, "is_low_light": false, "is_blurry": false},\n'
        '  "detections": [\n'
        '    {"class_name": "valve", "confidence": 0.86, "bbox": {"x1": 100, "y1": 80, "x2": 400, "y2": 360}, "ocr_text": "V-1023", "ocr_confidence": 0.9, "identity": "V-1023", "identity_confidence": 0.82}\n'
        "  ],\n"
        '  "decision": {"status": "ACCEPTED", "message": "VLM visually identified a valve tag.", "reasons": ["visible valve assembly", "tag appears readable"], "next_action": "Verify with ValveLens OCR/ReID for a thesis-grade identity claim"}\n'
        "}\n\n"
        "Rules:\n"
        "- If a visible tag is clear, put it in ocr_text and identity, but this is only VLM-estimated.\n"
        "- If no tag is readable, use null for ocr_text and identity.\n"
        "- Use ACCEPTED only when the visible object/tag is clear; otherwise use UNCERTAIN.\n"
        "- Prefer one to five useful boxes. Do not invent registered database evidence."
    )


def _fallback_vlm_response(
    image_path: str,
    width: int,
    height: int,
    quality: Dict[str, Any],
    reason: str,
) -> InferenceResponse:
    zone = ZoneTop1(zone_id="vlm_unknown_zone", zone_name="VLM estimated industrial area", score=0.35)
    return InferenceResponse(
        request_id=str(uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        input=InputInfo(type="image", source=image_path),
        quality=QualityInfo(
            blur_score=float(quality.get("blur_score", 0.0)),
            brightness=float(quality.get("brightness", 0.0)),
            glare_score=float(quality.get("glare_score", 0.0)),
            is_low_light=bool(quality.get("is_low_light", False)),
            is_blurry=bool(quality.get("is_blurry", False)),
        ),
        zone=ZoneInfo(candidates=[zone], top1=zone),
        detections=[],
        decision=DecisionInfo(
            status="UNCERTAIN",
            selected_device=None,
            action="ASK_WIDER_VIEW",
            message="VLM-only demo could not produce structured detections.",
            reasons=[reason],
            next_action="Use the normal ValveLens model path or try a clearer image.",
            ui_hints=UiHints(),
        ),
    )


def infer_image_with_vlm_only(image_path: str, config: Dict[str, Any]) -> InferenceResponse:
    from PIL import Image

    import cv2
    import numpy as np

    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    quality = compute_quality(
        frame_bgr,
        tau_blur=float(config.get("tau_blur", 0.25)),
        tau_low_light=float(config.get("tau_low_light", 0.35)),
    )

    available, reason = vlm_available(config, force=True)
    if not available:
        return _fallback_vlm_response(str(path), width, height, quality, reason)

    assistant = _assistant_config(config)
    client = _client_for_provider(assistant["provider"])
    image_url = _image_data_url(str(path)) if assistant["include_image"] else None
    if not image_url:
        return _fallback_vlm_response(str(path), width, height, quality, "image file is not available for VLM")

    response = client.chat.completions.create(
        model=assistant["model"],
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON for a VLM-only industrial visual inspection demo.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _vlm_inference_prompt(width, height)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=max(assistant["max_tokens"], 700),
    )
    raw_text = response.choices[0].message.content or "{}"
    try:
        parsed = _extract_json_object(raw_text)
    except Exception as exc:
        return _fallback_vlm_response(str(path), width, height, quality, f"VLM JSON parse failed: {exc}")

    raw_zones = parsed.get("zone_candidates") or []
    zones: List[ZoneCandidate] = []
    for index, item in enumerate(raw_zones[:5]):
        zones.append(
            ZoneCandidate(
                zone_id=str(item.get("zone_id") or f"vlm_zone_{index + 1}"),
                zone_name=str(item.get("zone_name") or "VLM estimated industrial area"),
                score=_clamp_score(item.get("score"), 0.5),
            )
        )
    if not zones:
        zones = [
            ZoneCandidate(
                zone_id="vlm_zone_1",
                zone_name="VLM estimated industrial area",
                score=0.5,
            )
        ]
    zones.sort(key=lambda item: item.score, reverse=True)
    top1 = ZoneTop1(zone_id=zones[0].zone_id, zone_name=zones[0].zone_name, score=zones[0].score)

    raw_quality = parsed.get("image_quality") or {}
    brightness = _clamp_score(raw_quality.get("brightness"), float(quality.get("brightness", 0.0)))
    blur_score = _clamp_score(raw_quality.get("blur_score"), float(quality.get("blur_score", 0.0)))
    glare_score = _clamp_score(raw_quality.get("glare_score"), float(quality.get("glare_score", 0.0)))

    detections: List[DetectionInfo] = []
    for item in (parsed.get("detections") or [])[:8]:
        class_name = str(item.get("class_name") or item.get("cls") or "unknown").strip().lower()
        if class_name not in VLM_DETECTION_CLASSES:
            class_name = "unknown"
        confidence = _clamp_score(item.get("confidence"), 0.5)
        bbox = _clamp_bbox(item.get("bbox") or {}, width, height)
        ocr_text = item.get("ocr_text")
        ocr_text = str(ocr_text).strip() if ocr_text else None
        ocr_conf = _clamp_score(item.get("ocr_confidence"), confidence if ocr_text else 0.0)
        identity = item.get("identity") or ocr_text
        identity = str(identity).strip() if identity else None
        identity_score = _clamp_score(item.get("identity_confidence"), confidence if identity else 0.0)
        matches = [ReIdMatch(device_id=identity, score=identity_score)] if identity else []
        detections.append(
            DetectionInfo(
                det_id=str(uuid4()),
                cls=class_name,
                class_id=None,
                class_name=class_name,
                conf=confidence,
                bbox=bbox,
                track_id=None,
                roi=RoiInfo(),
                ocr=OcrInfo(text=ocr_text, conf=ocr_conf if ocr_text else None, boxes=None),
                reid=ReIdInfo(embedding_type="vlm_estimated", top_matches=matches),
                fused=FusedInfo(
                    device_id=identity,
                    final_score=identity_score if identity else confidence,
                    score_breakdown={
                        "mode": "vlm_only_demo",
                        "vlm_conf": confidence,
                        "vlm_ocr_conf": ocr_conf if ocr_text else 0.0,
                        "vlm_identity_conf": identity_score if identity else 0.0,
                    },
                ),
            )
        )

    raw_decision = parsed.get("decision") or {}
    best_identity = max(
        (det for det in detections if det.fused.device_id),
        key=lambda det: det.fused.final_score,
        default=None,
    )
    status = str(raw_decision.get("status") or ("ACCEPTED" if best_identity else "UNCERTAIN")).upper()
    if status not in {"ACCEPTED", "UNCERTAIN", "UNKNOWN"}:
        status = "UNCERTAIN"
    selected = None
    if best_identity and status == "ACCEPTED":
        selected = {"device_id": best_identity.fused.device_id, "score": best_identity.fused.final_score}
    reasons = raw_decision.get("reasons") or ["VLM-only demo estimate; not validated by YOLO/OCR/ReID."]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    return InferenceResponse(
        request_id=str(uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        input=InputInfo(type="image", source=str(path)),
        quality=QualityInfo(
            blur_score=blur_score,
            brightness=brightness,
            glare_score=glare_score,
            is_low_light=bool(raw_quality.get("is_low_light", brightness < float(config.get("tau_low_light", 0.35)))),
            is_blurry=bool(raw_quality.get("is_blurry", blur_score < float(config.get("tau_blur", 0.25)))),
        ),
        zone=ZoneInfo(candidates=zones, top1=top1),
        detections=detections,
        decision=DecisionInfo(
            status=status,
            selected_device=selected,
            action="NONE" if status == "ACCEPTED" else "ASK_TAP",
            message=str(raw_decision.get("message") or "VLM-only demo visual estimate."),
            reasons=[str(reason) for reason in reasons],
            next_action=str(raw_decision.get("next_action") or "Verify with the normal ValveLens model path for evidence-backed claims."),
            ui_hints=UiHints(highlight_det_id=best_identity.det_id if best_identity else None),
        ),
    )


def answer_with_vlm(
    question: str,
    evidence: Dict[str, Any],
    config: Dict[str, Any],
    image_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    assistant = _assistant_config(config)
    provider = assistant["provider"]
    client = _client_for_provider(provider)
    content: list[Dict[str, Any]] = [
        {"type": "text", "text": _vlm_prompt(question, evidence)}
    ]
    image_url = _image_data_url(image_path) if assistant["include_image"] else None
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    response = client.chat.completions.create(
        model=assistant["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=assistant["max_tokens"],
    )
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer.strip(),
        "confidence": 0.0,
        "mode": "vlm",
        "evidence_used": ["image", "structured_evidence"],
        "recommended_next_action": "Use ValveLens OCR/ReID evidence for exact identity, or tap-select the object.",
        "uncertainty_reason": "",
        "evidence": evidence,
        "vlm_status": "provider_response",
        "provider": provider,
        "model": assistant["model"],
    }


def answer_with_vlm_or_fallback(
    question: str,
    evidence: Dict[str, Any],
    config: Dict[str, Any],
    image_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    assistant = _assistant_config(config)
    available, reason = vlm_available(config, force=force)
    if not available:
        fallback = answer_from_evidence(question, evidence)
        fallback["mode"] = "rule_based"
        fallback["fallback_reason"] = reason
        fallback["vlm_status"] = reason
        return fallback
    if assistant["include_image"] and image_path and not Path(image_path).exists():
        fallback = answer_from_evidence(question, evidence)
        fallback["mode"] = "rule_based"
        fallback["fallback_reason"] = "image file is not available for VLM"
        fallback["vlm_status"] = "image_missing"
        return fallback

    try:
        return answer_with_vlm(question, evidence, config, image_path=image_path, force=force)
    except Exception as exc:
        if assistant["use_rule_fallback"]:
            fallback = answer_from_evidence(question, evidence)
            fallback["mode"] = "rule_based"
            fallback["fallback_reason"] = f"VLM provider error: {exc}"
            fallback["vlm_status"] = "provider_error"
            return fallback
        raise
