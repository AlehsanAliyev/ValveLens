import os
from typing import Any, Dict, Optional

from app.evidence import answer_from_evidence


SYSTEM_PROMPT = (
    "You are an assistant for ValveLens. Answer using only the provided "
    "ValveLens evidence. Do not invent device IDs, locations, or confidence. "
    "If evidence is weak, say uncertain and recommend the next action."
)


def _assistant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    assistant = config.get("assistant") or {}
    return {
        "enable_vlm": bool(assistant.get("enable_vlm", False)),
        "provider": assistant.get("provider") or "env_configured",
        "model": assistant.get("model"),
        "max_tokens": int(assistant.get("max_tokens", 300)),
        "use_rule_fallback": bool(assistant.get("use_rule_fallback", True)),
    }


def vlm_available(config: Dict[str, Any]) -> tuple[bool, str]:
    assistant = _assistant_config(config)
    if not assistant["enable_vlm"]:
        return False, "VLM disabled in config"
    if not assistant["model"]:
        return False, "assistant.model is not configured"
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not configured"
    try:
        import openai  # noqa: F401
    except Exception:
        return False, "openai package is not installed"
    return True, "available"


def answer_with_vlm_or_fallback(
    question: str,
    evidence: Dict[str, Any],
    config: Dict[str, Any],
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    assistant = _assistant_config(config)
    available, reason = vlm_available(config)
    if not available:
        fallback = answer_from_evidence(question, evidence)
        fallback["mode"] = "rule_based"
        fallback["vlm_status"] = reason
        return fallback

    # VLM integration is intentionally gated until credentials/model are configured.
    # The rule fallback remains the reliability path for tests and offline demos.
    if assistant["use_rule_fallback"]:
        fallback = answer_from_evidence(question, evidence)
        fallback["mode"] = "rule_based"
        fallback["vlm_status"] = "VLM scaffold available; provider call not enabled in this build"
        if image_path:
            fallback["image_path"] = image_path
        return fallback

    return {
        "answer": "VLM mode is configured but provider execution is not enabled in this build.",
        "confidence": 0.0,
        "mode": "vlm",
        "evidence_used": ["structured_evidence"],
        "recommended_next_action": "Use rule-based fallback or enable the provider adapter.",
        "uncertainty_reason": "VLM provider call is not enabled",
        "evidence": evidence,
        "vlm_status": "provider execution not enabled",
    }
