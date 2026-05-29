import base64
from io import BytesIO
from pathlib import Path
from typing import List
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel
from PIL import Image

from app import db
from app.schemas import InferenceResponse
from app.vlm_assistant import infer_image_with_vlm_only

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class WebcamFrameRequest(BaseModel):
    session_id: str
    frame_index: int
    image_b64: str


def _store_observation(response: InferenceResponse) -> None:
    db.insert_observation(
        {
            "obs_id": response.request_id,
            "created_at": response.timestamp,
            "input_type": response.input.type,
            "source_name": response.input.source,
            "zone_top1": response.zone.top1.zone_id if response.zone.top1 else None,
            "zone_conf": response.zone.top1.score if response.zone.top1 else None,
            "final_device_id": response.decision.selected_device.device_id
            if response.decision.selected_device
            else None,
            "final_conf": response.decision.selected_device.score
            if response.decision.selected_device
            else None,
            "policy_action": response.decision.action,
            "payload_json": response.dict(),
        }
    )


@router.post("/infer/image", response_model=InferenceResponse)
async def infer_image(request: Request, file: UploadFile = File(...)) -> InferenceResponse:
    data = await file.read()
    safe_name = Path(file.filename or "upload.jpg").name
    stored_path = UPLOAD_DIR / f"{uuid4()}_{safe_name}"
    stored_path.write_bytes(data)
    img = Image.open(BytesIO(data)).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    response = request.app.state.pipeline.process_frame(
        frame_bgr,
        input_type="image",
        source=str(stored_path),
    )
    _store_observation(response)
    return response


@router.post("/infer/image_vlm", response_model=InferenceResponse)
async def infer_image_vlm(request: Request, file: UploadFile = File(...)) -> InferenceResponse:
    data = await file.read()
    safe_name = Path(file.filename or "upload.jpg").name
    stored_path = UPLOAD_DIR / f"{uuid4()}_{safe_name}"
    stored_path.write_bytes(data)
    response = infer_image_with_vlm_only(str(stored_path), request.app.state.pipeline.config)
    _store_observation(response)
    return response


@router.post("/infer/video", response_model=List[InferenceResponse])
async def infer_video(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> List[InferenceResponse]:
    session_id = session_id or str(uuid4())
    filename = file.filename or f"{session_id}.mp4"
    temp_path = UPLOAD_DIR / f"{session_id}_{filename}"
    temp_path.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(temp_path))
    frame_stride = int(request.app.state.pipeline.config.get("frame_stride", 5))
    responses: List[InferenceResponse] = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride == 0:
            response = request.app.state.pipeline.process_frame(
                frame,
                input_type="video",
                source=session_id,
                frame_index=frame_index,
                session_id=session_id,
            )
            _store_observation(response)
            responses.append(response)
        frame_index += 1
    cap.release()
    return responses


@router.post("/infer/webcam/frame", response_model=InferenceResponse)
async def infer_webcam_frame(
    request: Request, payload: WebcamFrameRequest
) -> InferenceResponse:
    image_b64 = payload.image_b64
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(data)).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    response = request.app.state.pipeline.process_frame(
        frame_bgr,
        input_type="webcam",
        source=payload.session_id,
        frame_index=payload.frame_index,
        session_id=payload.session_id,
    )
    _store_observation(response)
    return response
