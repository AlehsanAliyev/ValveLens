import json
from typing import Dict, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel

from app import db

router = APIRouter()


class FeedbackRequest(BaseModel):
    obs_id: str
    feedback_type: str
    data_json: Dict
    session_id: Optional[str] = None


def _resolve_selected_device(
    observation: Dict, det_id: str, device_id_override: Optional[str]
) -> Tuple[Optional[str], Optional[float]]:
    payload_json = json.loads(observation["payload_json"])
    for det in payload_json.get("detections", []):
        if det.get("det_id") != det_id:
            continue

        if device_id_override:
            score = det.get("fused", {}).get("final_score")
            return device_id_override, float(score or 1.0)

        fused = det.get("fused") or {}
        if fused.get("device_id"):
            return fused["device_id"], float(fused.get("final_score") or 1.0)

        ocr = det.get("ocr") or {}
        if ocr.get("text") and ocr.get("conf"):
            return ocr["text"], float(ocr.get("conf") or 1.0)

        top_matches = ((det.get("reid") or {}).get("top_matches") or [])
        if top_matches:
            top1 = top_matches[0]
            return top1.get("device_id"), float(top1.get("score") or 1.0)
        break
    return None, None


def _apply_tap_select(
    observation: Dict, det_id: str, device_id_override: Optional[str]
) -> Dict:
    payload_json = json.loads(observation["payload_json"])
    selected, selected_score = _resolve_selected_device(
        observation, det_id, device_id_override
    )
    if selected:
        payload_json["decision"]["status"] = "ACCEPTED"
        payload_json["decision"]["selected_device"] = {
            "device_id": selected,
            "score": float(selected_score or 1.0),
        }
        payload_json["decision"]["action"] = "NONE"
        payload_json["decision"]["message"] = f"User selected device {selected}."
        if "ui_hints" not in payload_json["decision"]:
            payload_json["decision"]["ui_hints"] = {}
        payload_json["decision"]["ui_hints"]["highlight_det_id"] = det_id
    return payload_json


def _resolve_session_id(
    observation: Optional[Dict], explicit_session_id: Optional[str]
) -> Optional[str]:
    if explicit_session_id:
        return explicit_session_id
    if not observation:
        return None
    if observation.get("source_name"):
        return str(observation["source_name"])
    payload = observation.get("payload_json")
    if payload:
        try:
            parsed = json.loads(payload)
            input_obj = parsed.get("input") or {}
            source = input_obj.get("source")
            if source:
                return str(source)
        except Exception:
            return None
    return None


@router.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    observation = db.fetch_observation(payload.obs_id)
    session_id = _resolve_session_id(observation, payload.session_id)
    feedback_data = dict(payload.data_json)

    if payload.feedback_type == "tap_select":
        if not observation:
            fb_id = db.insert_feedback(
                payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
            )
            return {"feedback_id": fb_id, "error": "observation_not_found"}
        det_id = payload.data_json.get("det_id")
        device_id_override = payload.data_json.get("device_id")
        if not det_id:
            fb_id = db.insert_feedback(
                payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
            )
            return {"feedback_id": fb_id, "error": "missing_det_id"}
        selected_device_id, selected_score = _resolve_selected_device(
            observation, det_id, device_id_override
        )
        if selected_device_id:
            feedback_data["device_id"] = selected_device_id
            feedback_data["score"] = float(selected_score or 1.0)
        fb_id = db.insert_feedback(
            payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
        )
        updated_payload = _apply_tap_select(observation, det_id, device_id_override)
        selected = updated_payload["decision"].get("selected_device")
        db.update_observation(
            payload.obs_id,
            updated_payload,
            selected.get("device_id") if selected else None,
            selected.get("score") if selected else None,
            updated_payload["decision"].get("action"),
        )
        return {"feedback_id": fb_id, "decision": updated_payload["decision"]}

    if payload.feedback_type == "confirm":
        if feedback_data.get("device_id"):
            fb_id = db.insert_feedback(
                payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
            )
        else:
            fb_id = None
        if observation:
            payload_json = json.loads(observation["payload_json"])
            selected = payload_json["decision"].get("selected_device")
            if not selected and feedback_data.get("device_id"):
                selected = {
                    "device_id": feedback_data["device_id"],
                    "score": float(feedback_data.get("score") or 1.0),
                }
                payload_json["decision"]["selected_device"] = selected
            if selected:
                if fb_id is None:
                    feedback_data["device_id"] = selected.get("device_id")
                    feedback_data["score"] = float(selected.get("score") or 1.0)
                    fb_id = db.insert_feedback(
                        payload.obs_id,
                        payload.feedback_type,
                        feedback_data,
                        session_id=session_id,
                    )
                payload_json["decision"]["status"] = "ACCEPTED"
                payload_json["decision"]["action"] = "NONE"
                payload_json["decision"]["message"] = (
                    f"Confirmed device {selected.get('device_id')}."
                )
                db.update_observation(
                    payload.obs_id,
                    payload_json,
                    selected.get("device_id"),
                    selected.get("score"),
                    payload_json["decision"].get("action"),
                )
                return {"feedback_id": fb_id, "decision": payload_json["decision"]}
        if fb_id is None:
            fb_id = db.insert_feedback(
                payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
            )
        return {"feedback_id": fb_id}

    if payload.feedback_type == "reject":
        fb_id = db.insert_feedback(
            payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
        )
        if observation:
            payload_json = json.loads(observation["payload_json"])
            payload_json["decision"]["status"] = "UNCERTAIN"
            payload_json["decision"]["action"] = "ASK_TAP"
            payload_json["decision"]["message"] = "Selection rejected. Please tap a device."
            db.update_observation(
                payload.obs_id,
                payload_json,
                None,
                None,
                payload_json["decision"].get("action"),
            )
            return {"feedback_id": fb_id, "decision": payload_json["decision"]}

    fb_id = db.insert_feedback(
        payload.obs_id, payload.feedback_type, feedback_data, session_id=session_id
    )
    return {"feedback_id": fb_id}
