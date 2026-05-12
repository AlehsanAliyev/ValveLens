import json

from app.routes import ask as ask_route


def _observation() -> dict:
    payload = {
        "request_id": "obs-ask-1",
        "input": {"type": "image", "source": "sample.jpg"},
        "quality": {
            "blur_score": 0.8,
            "brightness": 0.6,
            "glare_score": 0.0,
            "is_low_light": False,
            "is_blurry": False,
        },
        "zone": {
            "top1": {"zone_id": "zone-a", "zone_name": "Pump room", "score": 0.9},
            "candidates": [{"zone_id": "zone-a", "zone_name": "Pump room", "score": 0.9}],
        },
        "detections": [
            {
                "det_id": "det-1",
                "cls": "valve",
                "conf": 0.8,
                "bbox": {"x1": 10, "y1": 10, "x2": 120, "y2": 120},
                "track_id": None,
                "roi": {"crop_path": "crop.png", "mask_path": None},
                "ocr": {"text": "V-1023", "conf": 0.82, "boxes": None},
                "reid": {
                    "embedding_type": "clip",
                    "top_matches": [{"device_id": "V-1023", "score": 0.9}],
                },
                "fused": {
                    "device_id": "V-1023",
                    "final_score": 0.82,
                    "score_breakdown": {},
                },
            }
        ],
        "decision": {
            "status": "ACCEPTED",
            "selected_device": {"device_id": "V-1023", "score": 0.82},
            "action": "NONE",
            "message": "Identified device V-1023 via OCR.",
            "ui_hints": {"highlight_det_id": "det-1"},
        },
    }
    return {
        "obs_id": "obs-ask-1",
        "source_name": "session-ask",
        "payload_json": json.dumps(payload),
    }


def test_ask_route_answers_from_observation(monkeypatch) -> None:
    monkeypatch.setattr(ask_route.db, "init_db", lambda: None)
    monkeypatch.setattr(ask_route.db, "fetch_observation", lambda obs_id: _observation())
    monkeypatch.setattr(ask_route.db, "fetch_feedback_rows", lambda: [])

    response = ask_route.ask(
        ask_route.AskRequest(
            question="What is this?",
            observation_id="obs-ask-1",
            selected_detection_id="det-1",
        )
    )

    assert response["mode"] == "rule_based"
    assert "V-1023" in response["answer"]
    assert response["evidence"]["observation_id"] == "obs-ask-1"


def test_ask_route_vlm_unavailable_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(ask_route.db, "init_db", lambda: None)
    monkeypatch.setattr(ask_route.db, "fetch_observation", lambda obs_id: _observation())
    monkeypatch.setattr(ask_route.db, "fetch_feedback_rows", lambda: [])

    response = ask_route.ask(
        ask_route.AskRequest(
            question="What is this?",
            observation_id="obs-ask-1",
            selected_detection_id="det-1",
            use_vlm=True,
        )
    )

    assert response["mode"] == "rule_based"
    assert "vlm_status" in response
    assert "V-1023" in response["answer"]


def test_ask_route_no_observation(monkeypatch) -> None:
    monkeypatch.setattr(ask_route.db, "init_db", lambda: None)
    monkeypatch.setattr(ask_route.db, "fetch_observations", lambda: [])
    response = ask_route.ask(ask_route.AskRequest(question="What is this?"))

    assert response["recommended_next_action"] == "RUN_INFERENCE"
    assert response["confidence"] == 0.0
