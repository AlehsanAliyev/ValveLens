import argparse
import json
from pathlib import Path

import cv2

from app import db
from app.pipeline import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    db.init_db()
    pipeline = InferencePipeline()
    cap = cv2.VideoCapture(args.video)
    frame_stride = int(pipeline.config.get("frame_stride", 5))
    responses = []
    frame_index = 0
    session_id = Path(args.video).stem
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride == 0:
            response = pipeline.process_frame(
                frame,
                input_type="video",
                source=args.video,
                frame_index=frame_index,
                session_id=session_id,
            )
            responses.append(response.dict())
        frame_index += 1
    cap.release()

    Path(args.out).write_text(json.dumps(responses, indent=2), encoding="utf-8")
    print(f"Saved {len(responses)} frames to {args.out}")


if __name__ == "__main__":
    main()
