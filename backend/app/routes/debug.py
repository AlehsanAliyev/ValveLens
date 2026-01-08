from fastapi import APIRouter, Request

from app import db
from app.faiss_store import FaissIndex

router = APIRouter()


@router.get("/debug/status")
def debug_status(request: Request) -> dict:
    counts = db.fetch_counts()
    pipeline = request.app.state.pipeline

    zone_index = FaissIndex("zones", pipeline.embedder.dim)
    zone_index.load()
    device_index = FaissIndex("devices", pipeline.embedder.dim)
    device_index.load()

    return {
        "counts": counts,
        "faiss": {
            "zones": len(zone_index.meta),
            "devices": len(device_index.meta),
        },
    }
