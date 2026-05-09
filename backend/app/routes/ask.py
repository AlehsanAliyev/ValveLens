import json
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app import db
from app.evidence import answer_from_evidence, build_evidence
from app.pipeline import load_config

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    obs_id: Optional[str] = None
    selected_det_id: Optional[str] = None


def _safe_json_load(raw: Optional[str]) -> Dict[str, Any]:
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
    if payload.obs_id:
        return db.fetch_observation(payload.obs_id)
    return _latest_observation(payload.session_id)


@router.post("/ask")
def ask(payload: AskRequest) -> Dict[str, Any]:
    db.init_db()
    observation = _observation_for_request(payload)
    if not observation:
        return {
            "answer": "I do not have an inference result to answer from yet.",
            "confidence": 0.0,
            "recommended_next_action": "RUN_INFERENCE",
            "evidence": {},
        }

    response_payload = _safe_json_load(observation.get("payload_json"))
    thresholds = load_config()
    evidence = build_evidence(
        response_payload,
        selected_det_id=payload.selected_det_id,
        thresholds=thresholds,
    )
    return answer_from_evidence(payload.question, evidence)
