from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from app.ocr import OCRReader, extract_device_ids, normalize_text


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _ocr_backend_available(reader: OCRReader) -> bool:
    if reader.backend == "easyocr":
        return True
    if reader.backend == "tesseract":
        return shutil.which("tesseract") is not None
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on one image and compare to an expected device ID.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected", default="")
    args = parser.parse_args()

    image_path = _resolve(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    reader = OCRReader()
    backend_available = _ocr_backend_available(reader)
    print("OCR smoke test:")
    print(f"  image: {image_path}")
    print(f"  backend: {reader.backend or 'none'}")
    print(f"  backend_available: {str(backend_available).lower()}")
    if not backend_available:
        print(
            "  warning: OCR backend is unavailable. If using pytesseract, install "
            "Tesseract-OCR and add C:\\Program Files\\Tesseract-OCR to PATH."
        )

    image = Image.open(image_path).convert("RGB")
    result = reader.read(image)
    text = result.get("text") or ""
    candidates = extract_device_ids(text)
    expected = normalize_text(args.expected) if args.expected else ""
    expected_nodash = expected.replace("-", "")
    exact_match = bool(
        expected
        and any(
            normalize_text(candidate) == expected
            or normalize_text(candidate).replace("-", "") == expected_nodash
            for candidate in candidates
        )
    )

    print(f"  raw_text: {text or '<none>'}")
    print(f"  confidence: {result.get('conf') if result.get('conf') is not None else '<none>'}")
    print(f"  parsed_device_ids: {candidates}")
    print(f"  expected: {args.expected or '<none>'}")
    print(f"  exact_match: {str(exact_match).lower()}")


if __name__ == "__main__":
    main()
