from io import BytesIO
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, Request
from pydantic import BaseModel
from PIL import Image

from app import db
from app.faiss_store import rebuild_device_index

router = APIRouter()

DEVICE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "devices"
DEVICE_DIR.mkdir(parents=True, exist_ok=True)


class CreateDeviceRequest(BaseModel):
    device_id: str
    zone_id: str
    device_type: str
    description: str = ""


@router.post("/devices/create")
def create_device(payload: CreateDeviceRequest) -> dict:
    device_id = db.create_device(
        payload.device_id, payload.zone_id, payload.device_type, payload.description
    )
    return {"device_id": device_id}


@router.post("/devices/{device_id}/refs")
async def add_device_refs(
    request: Request, device_id: str, files: List[UploadFile] = File(...)
) -> dict:
    pipeline = request.app.state.pipeline
    stored = 0
    device_path = DEVICE_DIR / device_id
    device_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"
        filename = f"{uuid4()}{suffix}"
        save_path = device_path / filename
        with save_path.open("wb") as f:
            f.write(data)
        emb = pipeline.embedder.embed_image(img).astype("float32").tobytes()
        db.add_device_ref(device_id, str(save_path), pipeline.embedder.embedding_type, emb)
        stored += 1

    return {"count": stored}


@router.post("/devices/rebuild_index")
def rebuild_index(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    count = rebuild_device_index(pipeline.embedder.dim)
    return {"count": count}
