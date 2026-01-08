from typing import Dict

import json
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app import db

router = APIRouter()


class FeedbackRequest(BaseModel):
    obs_id: str
    feedback_type: str
    data_json: Dict


def _apply_tap_select(
    observation: Dict, det_id: str, device_id_override: Optional[str]
) -> Dict:
    payload_json = json.loads(observation["payload_json"])
    selected = None
    selected_score = None
    for det in payload_json.get("detections", []):
        if det.get("det_id") == det_id:
            selected = det.get("fused", {}).get("device_id")
            selected_score = det.get("fused", {}).get("final_score")
            break
    if device_id_override:
        selected = device_id_override
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


@router.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    fb_id = db.insert_feedback(payload.obs_id, payload.feedback_type, payload.data_json)

    if payload.feedback_type == "tap_select":
        observation = db.fetch_observation(payload.obs_id)
        if not observation:
            return {"feedback_id": fb_id, "error": "observation_not_found"}
        det_id = payload.data_json.get("det_id")
        device_id_override = payload.data_json.get("device_id")
        if not det_id:
            return {"feedback_id": fb_id, "error": "missing_det_id"}
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
        observation = db.fetch_observation(payload.obs_id)
        if observation:
            payload_json = json.loads(observation["payload_json"])
            selected = payload_json["decision"].get("selected_device")
            if selected:
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

    if payload.feedback_type == "reject":
        observation = db.fetch_observation(payload.obs_id)
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

    return {"feedback_id": fb_id}
