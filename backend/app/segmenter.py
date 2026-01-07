from typing import Optional, Tuple

import numpy as np
from PIL import Image


class Segmenter:
    def __init__(self) -> None:
        self.available = False
        self.predictor = None
        try:
            import os
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            checkpoint = os.environ.get("SAM_CHECKPOINT")
            if checkpoint:
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
                self.predictor = SamPredictor(sam)
                self.available = True
        except Exception:
            self.available = False

    def refine_roi(
        self, frame_bgr: np.ndarray, bbox: dict
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
        crop_img = Image.fromarray(crop[:, :, ::-1])
        if not self.available or self.predictor is None:
            return crop_img, None

        # Optional SAM support: mask within the bbox prompt
        try:
            self.predictor.set_image(frame_bgr[:, :, ::-1])
            box = np.array([x1, y1, x2, y2])
            masks, _, _ = self.predictor.predict(box=box)
            if masks is None or len(masks) == 0:
                return crop_img, None
            return crop_img, masks[0].astype(np.uint8)
        except Exception:
            return crop_img, None
