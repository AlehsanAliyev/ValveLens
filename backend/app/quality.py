from typing import Dict

import cv2
import numpy as np


def compute_quality(frame_bgr: np.ndarray, tau_blur: float, tau_low_light: float) -> Dict:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, blur_var / 1000.0)
    brightness = float(np.mean(gray) / 255.0)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    glare_pixels = np.sum(v > 240)
    glare_score = float(glare_pixels / v.size)

    is_low_light = brightness < tau_low_light
    is_blurry = blur_score < tau_blur

    return {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "glare_score": float(glare_score),
        "is_low_light": bool(is_low_light),
        "is_blurry": bool(is_blurry),
    }
