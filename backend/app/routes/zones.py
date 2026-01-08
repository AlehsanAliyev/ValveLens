from io import BytesIO
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, Request
from pydantic import BaseModel
from PIL import Image

from app import db
from app.faiss_store import rebuild_zone_index

router = APIRouter()

ZONE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "zones"
ZONE_DIR.mkdir(parents=True, exist_ok=True)


class CreateZoneRequest(BaseModel):
    name: str
    description: str = ""


@router.post("/zones/create")
def create_zone(payload: CreateZoneRequest) -> dict:
    zone_id = db.create_zone(payload.name, payload.description)
    return {"zone_id": zone_id}


@router.post("/zones/{zone_id}/keyframes")
async def add_keyframes(
    request: Request, zone_id: str, files: List[UploadFile] = File(...)
) -> dict:
    pipeline = request.app.state.pipeline
    stored = 0
    zone_path = ZONE_DIR / zone_id
    zone_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")

        suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"
        filename = f"{uuid4()}{suffix}"
        save_path = zone_path / filename
        with save_path.open("wb") as f:
            f.write(data)
        emb = pipeline.embedder.embed_image(img).astype("float32").tobytes()
        db.add_zone_keyframe(zone_id, str(save_path), pipeline.embedder.embedding_type, emb)
        stored += 1

    return {"count": stored}


@router.post("/zones/rebuild_index")
def rebuild_index(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    count = rebuild_zone_index(pipeline.embedder.dim)
    pipeline.zone_index.load()
    return {"count": count}
