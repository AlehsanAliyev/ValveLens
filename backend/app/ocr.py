import re
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


_DEVICE_ID_PATTERNS = [
    re.compile(r"\bV-?\d{2,6}\b", re.IGNORECASE),
    re.compile(r"\bPG-?\d{2,6}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{1,3}-?\d{2,6}\b", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    text = text.upper().strip()
    return re.sub(r"[^A-Z0-9-]", "", text)


def _canonicalize_device_id(candidate: str) -> str:
    candidate = normalize_text(candidate)
    if "-" in candidate:
        return candidate
    parts = re.split(r"(\d+)", candidate, maxsplit=1)
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}-{parts[1]}"
    return candidate


def extract_device_ids(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    seen = set()
    for pattern in _DEVICE_ID_PATTERNS:
        for match in pattern.finditer(text):
            device_id = _canonicalize_device_id(match.group(0))
            if device_id not in seen:
                seen.add(device_id)
                found.append(device_id)
    return found


def extract_device_id(text: str) -> Optional[str]:
    ids = extract_device_ids(text)
    return ids[0] if ids else None


def match_enrolled_device_id(text: str, enrolled_device_ids: Iterable[str]) -> Optional[str]:
    candidates = extract_device_ids(text)
    if not candidates:
        return None
    enrolled_map: Dict[str, str] = {}
    for raw in enrolled_device_ids:
        normalized = normalize_text(raw)
        if not normalized:
            continue
        enrolled_map[normalized] = raw
        enrolled_map[normalized.replace("-", "")] = raw
    for candidate in candidates:
        norm = normalize_text(candidate)
        if norm in enrolled_map:
            return enrolled_map[norm]
        nodash = norm.replace("-", "")
        if nodash in enrolled_map:
            return enrolled_map[nodash]
    return None


def _to_builtin(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_builtin(value) for key, value in obj.items()}
    return obj


class OCRReader:
    def __init__(
        self,
        enable_preprocessing: bool = True,
        resize_factor: float = 2.0,
    ) -> None:
        self.reader = None
        self.backend = None
        self.enable_preprocessing = enable_preprocessing
        self.resize_factor = max(1.0, float(resize_factor))
        try:
            import easyocr  # type: ignore

            self.reader = easyocr.Reader(["en"], gpu=False)
            self.backend = "easyocr"
        except Exception:
            try:
                import pytesseract  # type: ignore

                self.reader = pytesseract
                self.backend = "tesseract"
            except Exception:
                self.reader = None
                self.backend = None

    def _preprocess_variants(self, image: Image.Image) -> List[Image.Image]:
        rgb = image.convert("RGB")
        if not self.enable_preprocessing:
            return [rgb]

        gray = ImageOps.grayscale(rgb)
        contrast = ImageOps.autocontrast(gray)
        sharpened = contrast.filter(ImageFilter.SHARPEN)
        scale = self.resize_factor
        resized = sharpened.resize(
            (max(1, int(sharpened.width * scale)), max(1, int(sharpened.height * scale))),
            Image.Resampling.LANCZOS,
        )
        threshold = resized.point(lambda px: 255 if px > 150 else 0)
        boosted = ImageEnhance.Contrast(resized).enhance(1.6)

        return [
            rgb,
            gray.convert("RGB"),
            contrast.convert("RGB"),
            sharpened.convert("RGB"),
            resized.convert("RGB"),
            boosted.convert("RGB"),
            threshold.convert("RGB"),
        ]

    def _read_once(self, image: Image.Image) -> Dict:
        if self.reader is None:
            return {"text": None, "conf": None, "boxes": []}

        if self.backend == "easyocr":
            results = self.reader.readtext(np.array(image))
            if not results:
                return {"text": None, "conf": None, "boxes": []}
            best = max(results, key=lambda r: r[2])
            boxes = [_to_builtin(r[0]) for r in results]
            return {"text": best[1], "conf": float(best[2]), "boxes": boxes}

        if self.backend == "tesseract":
            try:
                data = self.reader.image_to_data(image, output_type="dict")
            except Exception:
                return {"text": None, "conf": None, "boxes": []}
            texts: List[str] = []
            confs: List[float] = []
            boxes: List[List[int]] = []
            for i, text in enumerate(data.get("text", [])):
                if not text:
                    continue
                conf = float(data.get("conf", [0])[i])
                if conf <= 0:
                    continue
                x, y, w, h = (
                    data.get("left", [0])[i],
                    data.get("top", [0])[i],
                    data.get("width", [0])[i],
                    data.get("height", [0])[i],
                )
                texts.append(text)
                confs.append(conf / 100.0)
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
            if not texts:
                return {"text": None, "conf": None, "boxes": []}
            best_idx = int(np.argmax(confs))
            return {
                "text": texts[best_idx],
                "conf": float(confs[best_idx]),
                "boxes": boxes,
            }

        return {"text": None, "conf": None, "boxes": []}

    def read(self, image: Image.Image) -> Dict:
        best = {"text": None, "conf": None, "boxes": []}
        for variant in self._preprocess_variants(image):
            result = self._read_once(variant)
            result_conf = float(result.get("conf") or 0.0)
            best_conf = float(best.get("conf") or 0.0)
            if result.get("text") and result_conf >= best_conf:
                best = result
        return best
