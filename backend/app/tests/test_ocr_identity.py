from app.ocr import extract_device_ids, match_enrolled_device_id
from app.policy import decide_policy


def test_device_id_regex() -> None:
    text = "Valve V1023 near gauge PG-45 panel AB1234 tag X-12"
    ids = extract_device_ids(text)
    assert "V-1023" in ids
    assert "PG-45" in ids
    assert "AB-1234" in ids
    assert "X-12" in ids


def test_ocr_match_accepts_when_enrolled() -> None:
    matched = match_enrolled_device_id(
        "target id is v1023",
        enrolled_device_ids=["PG-45", "V-1023"],
    )
    assert matched == "V-1023"

    decision = decide_policy(
        quality={"is_blurry": False, "is_low_light": False},
        zone_top1_score=0.9,
        best_det_conf=0.8,
        ocr_match={"device_id": matched, "conf": 0.92},
        reid_match={"device_id": "PG-45", "score": 0.95, "gap": 0.2},
        thresholds={
            "tau_zone": 0.65,
            "tau_det": 0.4,
            "tau_ocr": 0.7,
            "tau_reid": 0.5,
            "tau_gap": 0.08,
        },
        highlight_det_id="det-1",
    )
    assert decision["status"] == "ACCEPTED"
    assert decision["selected_device"]["device_id"] == "V-1023"
