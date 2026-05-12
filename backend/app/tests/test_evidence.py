from app.evidence import answer_from_evidence, build_evidence


def _payload(status: str = "ACCEPTED") -> dict:
    selected_device = {"device_id": "V-1023", "score": 0.82} if status == "ACCEPTED" else None
    return {
        "request_id": "obs-1",
        "input": {"type": "image", "source": "sample.jpg"},
        "quality": {
            "blur_score": 0.9,
            "brightness": 0.7,
            "glare_score": 0.0,
            "is_low_light": False,
            "is_blurry": False,
        },
        "zone": {
            "top1": {"zone_id": "zone-a", "zone_name": "Pump room", "score": 0.91},
            "candidates": [{"zone_id": "zone-a", "zone_name": "Pump room", "score": 0.91}],
        },
        "detections": [
            {
                "det_id": "det-1",
                "cls": "valve",
                "conf": 0.88,
                "bbox": {"x1": 10, "y1": 20, "x2": 110, "y2": 120},
                "track_id": None,
                "roi": {"crop_path": "crop.png", "mask_path": None},
                "ocr": {"text": "V-1023", "conf": 0.86, "boxes": None},
                "reid": {
                    "embedding_type": "clip",
                    "top_matches": [
                        {"device_id": "V-1023", "score": 0.93},
                        {"device_id": "V-1212", "score": 0.71},
                    ],
                },
                "fused": {
                    "device_id": "V-1023",
                    "final_score": 0.82,
                    "score_breakdown": {"ocr": 0.86, "reid": 0.93},
                },
            }
        ],
        "decision": {
            "status": status,
            "selected_device": selected_device,
            "action": "NONE" if status == "ACCEPTED" else "ASK_TAP",
            "message": "Identified device V-1023 via OCR." if status == "ACCEPTED" else "Tap the correct device.",
            "ui_hints": {"highlight_det_id": "det-1"},
        },
    }


def test_build_evidence_compacts_identity_signals() -> None:
    evidence = build_evidence(
        _payload(),
        selected_detection_id="det-1",
        session_id="session-1",
        observation_id="obs-1",
        feedback_rows=[{"feedback_type": "confirm", "data_json": {"device_id": "V-1023"}}],
    )

    assert evidence["session_id"] == "session-1"
    assert evidence["observation_id"] == "obs-1"
    assert evidence["selected_detection"]["det_id"] == "det-1"
    assert evidence["selected_detection"]["ocr"]["parsed_device_ids"] == ["V-1023"]
    assert evidence["feedback"]["latest_type"] == "confirm"


def test_what_is_this_uses_accepted_identity() -> None:
    evidence = build_evidence(_payload(), selected_detection_id="det-1")
    answer = answer_from_evidence("What is this?", evidence)

    assert answer["mode"] == "rule_based"
    assert "V-1023" in answer["answer"]
    assert answer["confidence"] == 0.82


def test_where_device_uses_detection_location() -> None:
    evidence = build_evidence(_payload(), selected_detection_id="det-1")
    answer = answer_from_evidence("Where is V-1023?", evidence)

    assert "V-1023" in answer["answer"]
    assert "current frame" in answer["answer"]


def test_why_uncertain_explains_blocking_reasons() -> None:
    payload = _payload(status="UNCERTAIN")
    payload["quality"]["is_blurry"] = True
    payload["detections"][0]["ocr"] = {"text": None, "conf": None, "boxes": None}
    evidence = build_evidence(payload, selected_detection_id="det-1")
    answer = answer_from_evidence("Why are you uncertain?", evidence)

    assert "uncertain" in answer["answer"].lower()
    assert "blurry" in answer["answer"].lower()


def test_no_selected_detection_requests_selection() -> None:
    payload = _payload(status="UNCERTAIN")
    payload["decision"]["ui_hints"] = {}
    evidence = build_evidence(payload)
    answer = answer_from_evidence("What is this?", evidence)

    assert "select" in answer["answer"].lower()
