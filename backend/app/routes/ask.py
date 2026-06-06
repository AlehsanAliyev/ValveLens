import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app import db
from app.evidence import answer_from_evidence, build_evidence
from app.pipeline import load_config
from app.vlm_assistant import answer_with_vlm_or_fallback

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    observation_id: Optional[str] = None
    selected_detection_id: Optional[str] = None
    use_vlm: bool = False
    # Backward-compatible names used by the first frontend scaffold.
    obs_id: Optional[str] = None
    selected_det_id: Optional[str] = None


def _safe_json_load(raw: Optional[str]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _obs_session_id(obs: Dict[str, Any]) -> str:
    source_name = obs.get("source_name")
    if source_name:
        return str(source_name)
    payload = _safe_json_load(obs.get("payload_json"))
    source = ((payload.get("input") or {}).get("source")) if payload else None
    return str(source or "")


def _latest_observation(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    rows = db.fetch_observations()
    if session_id:
        rows = [row for row in rows if _obs_session_id(row) == session_id]
    if not rows:
        return None
    return rows[-1]


def _observation_for_request(payload: AskRequest) -> Optional[Dict[str, Any]]:
    observation_id = payload.observation_id or payload.obs_id
    if observation_id:
        return db.fetch_observation(observation_id)
    return _latest_observation(payload.session_id)


def _feedback_for_context(
    observation_id: Optional[str], session_id: Optional[str]
) -> List[Dict[str, Any]]:
    rows = db.fetch_feedback_rows()
    filtered = []
    for row in rows:
        if observation_id and row.get("obs_id") == observation_id:
            filtered.append(row)
        elif session_id and row.get("session_id") == session_id:
            filtered.append(row)
    for row in filtered:
        row["data_json"] = _safe_json_load(row.get("data_json"))
    return filtered


@router.post("/ask")
def ask(payload: AskRequest) -> Dict[str, Any]:
    db.init_db()
    observation = _observation_for_request(payload)
    if not observation:
        return {
            "answer": "I do not have an inference result to answer from yet.",
            "confidence": 0.0,
            "mode": "rule_based",
            "evidence_used": [],
            "recommended_next_action": "RUN_INFERENCE",
            "uncertainty_reason": "no observation found",
            "suggested_questions": ["What can you do?", "What image should I upload?"],
            "evidence": {},
        }

    response_payload = _safe_json_load(observation.get("payload_json"))
    thresholds = load_config()
    observation_id = payload.observation_id or payload.obs_id or observation.get("obs_id")
    selected_detection_id = payload.selected_detection_id or payload.selected_det_id
    session_id = payload.session_id or _obs_session_id(observation)
    evidence = build_evidence(
        response_payload,
        selected_detection_id=selected_detection_id,
        thresholds=thresholds,
        session_id=session_id,
        observation_id=observation_id,
        feedback_rows=_feedback_for_context(observation_id, session_id),
    )

    if payload.use_vlm:
        image_path = (response_payload.get("input") or {}).get("source")
        return answer_with_vlm_or_fallback(
            question=payload.question,
            evidence=evidence,
            config=thresholds,
            image_path=image_path,
        )
    return answer_from_evidence(payload.question, evidence)
