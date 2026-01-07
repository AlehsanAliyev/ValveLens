from typing import Dict, Tuple


def fuse_scores(
    zone_score: float,
    det_conf: float,
    reid_top1: float,
    ocr_conf: float,
    ocr_match: bool,
    gap_small: bool,
) -> Tuple[float, Dict]:
    base = 0.25 * zone_score + 0.30 * det_conf + 0.25 * reid_top1 + 0.20 * ocr_conf
    bonus = 0.20 if ocr_match else 0.0
    penalty = 0.10 if gap_small else 0.0
    final = max(0.0, min(1.0, base + bonus - penalty))
    breakdown = {
        "zone_score": zone_score,
        "det_conf": det_conf,
        "reid_top1": reid_top1,
        "ocr_conf": ocr_conf,
        "ocr_bonus": bonus,
        "gap_penalty": penalty,
    }
    return final, breakdown
