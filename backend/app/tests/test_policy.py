from app.policy import decide_policy


def test_policy_zone_low() -> None:
    decision = decide_policy(
        quality={"is_blurry": False, "is_low_light": False},
        zone_top1_score=0.2,
        best_det_conf=0.9,
        ocr_match=None,
        reid_match=None,
        thresholds={
            "tau_zone": 0.65,
            "tau_det": 0.4,
            "tau_ocr": 0.7,
            "tau_reid": 0.5,
            "tau_gap": 0.08,
        },
        highlight_det_id=None,
    )
    assert decision["action"] == "ASK_WIDER_VIEW"


def test_policy_accept_ocr() -> None:
    decision = decide_policy(
        quality={"is_blurry": False, "is_low_light": False},
        zone_top1_score=0.8,
        best_det_conf=0.9,
        ocr_match={"device_id": "V-1023", "conf": 0.9},
        reid_match=None,
        thresholds={
            "tau_zone": 0.65,
            "tau_det": 0.4,
            "tau_ocr": 0.7,
            "tau_reid": 0.5,
            "tau_gap": 0.08,
        },
        highlight_det_id=None,
    )
    assert decision["status"] == "ACCEPTED"
