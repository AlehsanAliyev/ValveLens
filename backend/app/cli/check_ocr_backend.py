from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _import_status(module_name: str) -> str:
    try:
        __import__(module_name)
        return "OK"
    except Exception:
        return "MISSING"


def _tesseract_status() -> tuple[str, str]:
    exe = shutil.which("tesseract")
    if not exe:
        return "MISSING", ""
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0]
        return f"OK ({exe})", first_line
    except Exception as exc:
        return f"FOUND ({exe})", f"version check failed: {exc}"


def main() -> None:
    config = _load_config()
    pytesseract_status = _import_status("pytesseract")
    easyocr_status = _import_status("easyocr")
    tesseract_status, tesseract_version = _tesseract_status()

    ocr_ready = easyocr_status == "OK" or (
        pytesseract_status == "OK" and tesseract_status.startswith("OK")
    )
    print("OCR backend check:")
    print(f"  pytesseract import: {pytesseract_status}")
    print(f"  tesseract executable: {tesseract_status}")
    if tesseract_version:
        print(f"  tesseract version: {tesseract_version}")
    print(f"  easyocr import: {easyocr_status}")
    print("  config:")
    for key in [
        "ocr_preprocess",
        "ocr_resize_factor",
        "ocr_expand_ratio",
        "tau_ocr",
    ]:
        if key in config:
            print(f"    {key}: {config[key]}")
    print(f"  status: {'OCR ready' if ocr_ready else 'OCR not ready'}")
    if not ocr_ready:
        print(
            "  fix: install Tesseract-OCR and add "
            "C:\\Program Files\\Tesseract-OCR to PATH, or install/configure EasyOCR."
        )


if __name__ == "__main__":
    main()
