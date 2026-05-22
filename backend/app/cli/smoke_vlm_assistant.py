from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from app.evidence import build_evidence
from app.pipeline import InferencePipeline, load_config
from app.vlm_assistant import answer_with_vlm_or_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test ValveLens VLM/rule assistant on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="What do you see in this image?")
    parser.add_argument("--use-vlm", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    import cv2
    import numpy as np

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    pipeline = InferencePipeline()
    response = pipeline.process_frame(frame_bgr, input_type="image", source=str(image_path))
    evidence = build_evidence(response.dict(), thresholds=load_config())
    result = answer_with_vlm_or_fallback(
        question=args.question,
        evidence=evidence,
        config=load_config(),
        image_path=str(image_path),
        force=args.use_vlm,
    )

    print("VLM assistant smoke test:")
    print(f"  image: {image_path}")
    print(f"  question: {args.question}")
    print(f"  mode: {result.get('mode')}")
    if result.get("fallback_reason"):
        print(f"  fallback_reason: {result.get('fallback_reason')}")
    if result.get("vlm_status"):
        print(f"  vlm_status: {result.get('vlm_status')}")
    print(f"  answer: {result.get('answer')}")


if __name__ == "__main__":
    main()
