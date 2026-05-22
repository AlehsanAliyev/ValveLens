from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from app.detector import Detector
from app.evidence import build_evidence
from app.pipeline import InferencePipeline, load_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "final_audit"


def _model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _detections_for_model(frame_bgr: np.ndarray, model_path: str) -> Dict[str, Any]:
    detector = Detector(model_path)
    detections = detector.detect(frame_bgr, conf_thres=0.01)
    return {
        "model_path": str(detector.resolved_model_path),
        "class_names": detector.class_names,
        "fallback_reason": detector.fallback_reason,
        "detections": [
            {
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name") or det.get("cls"),
                "display_class": det.get("cls"),
                "confidence": det.get("conf"),
                "bbox": det.get("bbox"),
            }
            for det in detections
        ],
    }


def _write_report(audit: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "image_inference_audit.json"
    md_path = out_dir / "image_inference_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Image Inference Audit",
        "",
        f"- Image: `{audit['image']['path']}`",
        f"- Size: {audit['image']['width']} x {audit['image']['height']}",
        f"- Detector model: `{audit['primary_detector']['model_path']}`",
        "",
        "## Quality",
    ]
    quality = audit["quality"]
    for key in ["brightness", "contrast", "blur_score", "glare_score", "is_blurry", "is_low_light"]:
        lines.append(f"- {key}: {quality.get(key)}")

    lines.extend(["", "## Primary Detector"])
    for det in audit["primary_detector"]["detections"]:
        lines.append(
            f"- class_id={det.get('class_id')} class={det.get('class_name')} "
            f"conf={float(det.get('confidence') or 0.0):.3f} bbox={det.get('bbox')}"
        )
    if not audit["primary_detector"]["detections"]:
        lines.append("- no detections")

    if audit.get("secondary_detector"):
        lines.extend(["", "## Secondary Detector"])
        for det in audit["secondary_detector"]["detections"]:
            lines.append(
                f"- class_id={det.get('class_id')} class={det.get('class_name')} "
                f"conf={float(det.get('confidence') or 0.0):.3f} bbox={det.get('bbox')}"
            )

    lines.extend(["", "## OCR"])
    for item in audit["ocr"]:
        lines.append(f"- det={item.get('det_id')} text={item.get('raw_text') or '<none>'} ids={item.get('parsed_ids')}")

    lines.extend(["", "## ReID"])
    for item in audit["reid"]:
        lines.append(f"- det={item.get('det_id')} top={item.get('top_matches')}")

    lines.extend(["", "## Fusion And Policy"])
    lines.append(f"- Decision: {audit['decision'].get('status')}")
    lines.append(f"- Message: {audit['decision'].get('message')}")
    lines.append(f"- Reasons: {audit['decision'].get('reasons') or audit['uncertainty_reasons']}")
    lines.append(f"- Next action: {audit['decision'].get('next_action')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ValveLens inference on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models\\detector.pt")
    parser.add_argument("--also-model", default=None)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    pipeline = InferencePipeline()
    pipeline.detector = Detector(args.model)
    response = pipeline.process_frame(frame_bgr, input_type="image", source=str(image_path))
    payload = _model_dump(response)
    evidence = build_evidence(payload, thresholds=load_config())

    primary = _detections_for_model(frame_bgr, args.model)
    secondary = _detections_for_model(frame_bgr, args.also_model) if args.also_model else None

    detections = payload.get("detections") or []
    audit = {
        "image": {
            "path": str(image_path),
            "width": image.width,
            "height": image.height,
        },
        "primary_detector": primary,
        "secondary_detector": secondary,
        "quality": {
            **(payload.get("quality") or {}),
            "contrast": float(np.std(gray) / 255.0),
        },
        "ocr": [
            {
                "det_id": det.get("det_id"),
                "raw_text": ((det.get("ocr") or {}).get("text")),
                "confidence": ((det.get("ocr") or {}).get("conf")),
                "parsed_ids": (
                    next((item for item in evidence.get("detections", []) if item.get("det_id") == det.get("det_id")), {})
                    .get("ocr", {})
                    .get("parsed_device_ids", [])
                ),
            }
            for det in detections
        ],
        "reid": [
            {
                "det_id": det.get("det_id"),
                "top_matches": (det.get("reid") or {}).get("top_matches") or [],
                "raw_top_matches": (det.get("reid") or {}).get("raw_top_matches") or [],
            }
            for det in detections
        ],
        "fusion": [
            {
                "det_id": det.get("det_id"),
                "fused": det.get("fused"),
            }
            for det in detections
        ],
        "decision": payload.get("decision") or {},
        "uncertainty_reasons": (evidence.get("uncertainty") or {}).get("reasons") or [],
    }

    _write_report(audit, DEFAULT_OUT)

    print("Image inference audit:")
    print(f"  image: {image_path}")
    print(f"  size: {image.width}x{image.height}")
    print(f"  detector model: {primary['model_path']}")
    print(f"  detector class names: {primary['class_names']}")
    print(f"  detections: {len(primary['detections'])}")
    for det in primary["detections"]:
        print(
            f"    class_id={det.get('class_id')} class_name={det.get('class_name')} "
            f"confidence={float(det.get('confidence') or 0.0):.3f} bbox={det.get('bbox')}"
        )
    q = audit["quality"]
    print(
        "  quality: "
        f"brightness={q.get('brightness'):.3f} contrast={q.get('contrast'):.3f} "
        f"blur_score={q.get('blur_score'):.3f} glare={q.get('glare_score'):.3f}"
    )
    print(f"  decision: {audit['decision'].get('status')} - {audit['decision'].get('message')}")
    print(f"  reasons: {audit['decision'].get('reasons') or audit['uncertainty_reasons']}")
    print(f"  saved: {DEFAULT_OUT / 'image_inference_audit.md'}")
    print(f"  saved: {DEFAULT_OUT / 'image_inference_audit.json'}")


if __name__ == "__main__":
    main()
