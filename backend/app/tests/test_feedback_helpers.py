import json
from uuid import uuid4

from app.db import fetch_latest_session_feedback_device, init_db, insert_feedback
from app.routes.feedback import _apply_tap_select


def test_apply_tap_select_uses_top_reid_match_when_fused_missing() -> None:
    observation = {
        "payload_json": json.dumps(
            {
                "detections": [
                    {
                        "det_id": "det-1",
                        "ocr": {"text": None, "conf": None},
                        "reid": {
                            "top_matches": [
                                {"device_id": "V-1023", "score": 0.81},
                            ]
                        },
                        "fused": {"device_id": None, "final_score": 0.42},
                    }
                ],
                "decision": {
                    "status": "UNCERTAIN",
                    "selected_device": None,
                    "action": "ASK_TAP",
                    "message": "Tap",
                    "ui_hints": {},
                },
            }
        )
    }

    updated = _apply_tap_select(observation, "det-1", None)
    assert updated["decision"]["status"] == "ACCEPTED"
    assert updated["decision"]["selected_device"]["device_id"] == "V-1023"


def test_fetch_latest_session_feedback_device_returns_latest_device() -> None:
    init_db()
    session_id = f"session-feedback-test-{uuid4()}"
    insert_feedback("obs-1", "tap_select", {"device_id": "V-1001"}, session_id=session_id)
    insert_feedback("obs-2", "confirm", {"device_id": "V-2002"}, session_id=session_id)

    latest = fetch_latest_session_feedback_device(session_id)
    assert latest is not None
    assert latest["data_json"]["device_id"] == "V-2002"
