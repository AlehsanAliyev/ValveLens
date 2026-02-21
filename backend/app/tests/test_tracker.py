from app.tracker import TrackerManager


def test_iou_tracker_persists_track_ids() -> None:
    manager = TrackerManager(iou_threshold=0.1, max_missed=2, smoothing_window=3)
    session_id = "session-a"

    frame1 = [
        {
            "det_id": "d1",
            "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
        }
    ]
    frame2 = [
        {
            "det_id": "d2",
            "bbox": {"x1": 11, "y1": 11, "x2": 31, "y2": 31},
        }
    ]

    match1 = manager.update(session_id, frame1)
    track_id = match1["d1"]["track_id"]
    assert track_id
    assert match1["d1"]["track_stability"] >= 1

    match2 = manager.update(session_id, frame2)
    assert match2["d2"]["track_id"] == track_id
    assert int(match2["d2"]["track_stability"]) >= 2


def test_tracker_temporal_smoothing_means() -> None:
    manager = TrackerManager(iou_threshold=0.1, max_missed=2, smoothing_window=3)
    session_id = "session-b"
    det = [{"det_id": "d1", "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}}]
    track = manager.update(session_id, det)["d1"]["track_id"]

    s1 = manager.update_signals(session_id, track, 0.2, 0.1, 0.3, "V-1")
    s2 = manager.update_signals(session_id, track, 0.6, 0.5, 0.7, "V-1")
    s3 = manager.update_signals(session_id, track, 1.0, 0.9, 0.8, "V-2")

    assert abs(float(s1["smoothed_det_conf"]) - 0.2) < 1e-6
    assert abs(float(s2["smoothed_det_conf"]) - 0.4) < 1e-6
    assert abs(float(s3["smoothed_det_conf"]) - 0.6) < 1e-6
    assert s3["smoothed_selected_device_id"] in {"V-1", "V-2"}
