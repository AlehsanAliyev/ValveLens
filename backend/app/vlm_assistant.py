import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.evidence import answer_from_evidence


SYSTEM_PROMPT = (
    "You are an assistant for ValveLens. Answer using only the provided "
    "ValveLens evidence. Do not invent device IDs, locations, or confidence. "
    "If evidence is weak, say uncertain and recommend the next action."
)


def _assistant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    assistant = config.get("assistant") or {}
    model = assistant.get("model") or os.environ.get("VALVELENS_VLM_MODEL")
    return {
        "enable_vlm": bool(assistant.get("enable_vlm", False)),
        "provider": (assistant.get("provider") or "openai").lower(),
        "model": model,
        "include_image": bool(assistant.get("include_image", True)),
        "max_tokens": int(assistant.get("max_tokens", 300)),
        "use_rule_fallback": bool(assistant.get("use_rule_fallback", True)),
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
        return False, "DEEPINFRA_TOKEN is not configured"
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
        return OpenAI(api_key=token, base_url="https://api.deepinfra.com/v1/openai")
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
    return (
        f"Question: {question}\n"
        "ValveLens evidence summary:\n"
        f"- decision: {decision.get('status')} / {decision.get('message')}\n"
        f"- visible device IDs from OCR/ReID/fusion: {', '.join(visible_ids) or 'none'}\n"
        f"- detector classes: {', '.join(classes) or 'none'}\n"
        f"- quality: blur_score={quality.get('blur_score')} brightness={quality.get('brightness')} "
        f"glare_score={quality.get('glare_score')}\n\n"
        "Answer visually and conservatively. You may describe visible objects generally "
        "(industrial pipe, valve assembly, flange, handwheel, gauge, tag visibility). "
        "Do not assign an exact device ID unless it appears in the ValveLens evidence summary."
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
