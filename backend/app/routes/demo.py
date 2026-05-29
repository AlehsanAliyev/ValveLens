from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db
from app.schemas import InferenceResponse
from app.vlm_assistant import infer_image_with_vlm_only


router = APIRouter(prefix="/demo", tags=["demo"])

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SAMPLE_FOLDERS = [
    "data/device_benchmark/queries",
    "data/detection/combined/test/images",
    "data/detection/industrial_multiclass/test/images",
    "data/device_benchmark/fullframe_demo",
    "data/device_benchmark/manual_v1023/queries",
]


class SampleInferRequest(BaseModel):
    path: str


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


@router.get("/samples")
def list_demo_samples(limit: int = 200) -> Dict[str, List[Dict[str, str]]]:
    samples: List[Dict[str, str]] = []
    for folder in SAMPLE_FOLDERS:
        root = REPO_ROOT / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(samples) >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            samples.append(
                {
                    "label": str(path.relative_to(root)).replace("\\", "/"),
                    "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "folder": folder,
                }
            )
    return {"samples": samples}


@router.post("/infer_sample", response_model=InferenceResponse)
def infer_demo_sample(request: Request, payload: SampleInferRequest) -> InferenceResponse:
    path = (REPO_ROOT / payload.path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Sample path must stay inside repository.")
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=404, detail="Sample image not found.")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read sample image.")
    response = request.app.state.pipeline.process_frame(
        frame,
        input_type="image",
        source=str(path),
    )
    _store_observation(response)
    return response


@router.post("/infer_sample_vlm", response_model=InferenceResponse)
def infer_demo_sample_vlm(request: Request, payload: SampleInferRequest) -> InferenceResponse:
    path = (REPO_ROOT / payload.path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Sample path must stay inside repository.")
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=404, detail="Sample image not found.")
    response = infer_image_with_vlm_only(str(path), request.app.state.pipeline.config)
    _store_observation(response)
    return response


@router.get("/sample_file")
def get_demo_sample_file(path: str) -> FileResponse:
    sample_path = (REPO_ROOT / path).resolve()
    try:
        sample_path.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Sample path must stay inside repository.")
    if not sample_path.exists() or sample_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(sample_path)
