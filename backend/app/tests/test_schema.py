from app.schemas import InferenceResponse


def test_schema_valid() -> None:
    payload = {
        "request_id": "req-1",
        "timestamp": "2024-01-01T00:00:00",
        "input": {"type": "image", "source": "sample.jpg", "frame_index": None},
        "quality": {
            "blur_score": 0.7,
            "brightness": 0.5,
            "glare_score": 0.1,
            "is_low_light": False,
            "is_blurry": False,
        },
        "zone": {
            "candidates": [{"zone_id": "z1", "score": 0.8}],
            "top1": {"zone_id": "z1", "score": 0.8},
        },
        "detections": [],
        "decision": {
            "status": "UNCERTAIN",
            "selected_device": None,
            "action": "ASK_TAP",
            "message": "Tap the correct device in the image.",
            "ui_hints": {"highlight_det_id": None, "suggested_moves": None},
        },
    }

    obj = InferenceResponse(**payload)
    assert obj.request_id == "req-1"
