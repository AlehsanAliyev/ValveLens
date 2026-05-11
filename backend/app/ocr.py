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
    for dash in ("—", "–", "−", "‐", "‑", "‒", "―"):
        text = text.replace(dash, "-")
    return re.sub(r"[^A-Z0-9-]", "", text)


def _normalize_ocr_scan_text(text: str) -> str:
    text = text.upper()
    for dash in ("—", "–", "−", "‐", "‑", "‒", "―"):
        text = text.replace(dash, "-")
    return text


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
    text = _normalize_ocr_scan_text(text)
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

        variants: List[Image.Image] = []
        variants.extend(self._white_region_crops(rgb))
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

        variants.extend(
            [
            rgb,
            gray.convert("RGB"),
            contrast.convert("RGB"),
            sharpened.convert("RGB"),
            resized.convert("RGB"),
            boosted.convert("RGB"),
            threshold.convert("RGB"),
            ]
        )
        return variants

    def _white_region_crops(self, image: Image.Image) -> List[Image.Image]:
        try:
            import cv2  # type: ignore
        except Exception:
            return []

        arr = np.array(image.convert("RGB"))
        gray = np.mean(arr, axis=2).astype(np.uint8)
        mask = (gray > 242).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        crops: List[Image.Image] = []
        image_area = max(1, image.width * image.height)
        for idx in range(1, num_labels):
            x, y, w, h, area = [int(value) for value in stats[idx]]
            if area < 1200 or area > image_area * 0.45:
                continue
            if w < 90 or h < 28:
                continue
            aspect = w / max(1, h)
            if aspect < 1.6 or aspect > 8.5:
                continue
            pad = 10
            left = max(0, x - pad)
            top = max(0, y - pad)
            right = min(image.width, x + w + pad)
            bottom = min(image.height, y + h + pad)
            crop = image.crop((left, top, right, bottom))
            crop = crop.resize((max(1, crop.width * 3), max(1, crop.height * 3)), Image.Resampling.LANCZOS)
            crops.append(crop)
        return crops[:4]

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
            config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-—–−"
            try:
                string_text = str(self.reader.image_to_string(image, config=config) or "").strip()
                if extract_device_ids(string_text):
                    return {"text": string_text, "conf": 0.70, "boxes": []}
            except Exception:
                pass
            try:
                data = self.reader.image_to_data(image, output_type="dict", config=config)
            except Exception:
                return {"text": None, "conf": None, "boxes": []}
            texts: List[str] = []
            confs: List[float] = []
            boxes: List[List[int]] = []
            for i, text in enumerate(data.get("text", [])):
                text = str(text or "").strip()
                if not normalize_text(text):
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
            combined = " ".join(texts)
            if extract_device_ids(combined):
                return {
                    "text": combined,
                    "conf": float(max(confs)),
                    "boxes": boxes,
                }
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
            result_text = str(result.get("text") or "").strip()
            best_text = str(best.get("text") or "").strip()
            result_has_id = bool(extract_device_ids(result_text))
            best_has_id = bool(extract_device_ids(best_text))
            if result_text and (
                (result_has_id and not best_has_id)
                or (result_has_id == best_has_id and result_conf >= best_conf)
            ):
                best = result
        return best
