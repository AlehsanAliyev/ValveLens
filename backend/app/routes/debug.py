from pathlib import Path

from fastapi import APIRouter, Request

from app import db
from app.faiss_store import FaissIndex, get_index_built_at

router = APIRouter()


@router.get("/debug/status")
def debug_status(request: Request) -> dict:
    counts = db.fetch_counts()
    pipeline = request.app.state.pipeline

    zone_index = FaissIndex("zones", pipeline.embedder.dim)
    zone_index.load()
    device_index = FaissIndex("devices", pipeline.embedder.dim)
    device_index.load()

    extracted_root = Path("data_sources/extracted")
    dataset_folders = []
    if extracted_root.exists():
        dataset_folders = sorted([p.name for p in extracted_root.iterdir() if p.is_dir()])

    return {
        "counts": {
            "zones": counts.get("zones", 0),
            "zone_keyframes": counts.get("zone_keyframes", 0),
            "devices": counts.get("devices", 0),
            "device_refs": counts.get("device_refs", 0),
            "observations": counts.get("observations", 0),
            "feedback": counts.get("feedback", 0),
            "zones_count": counts.get("zones", 0),
            "zone_keyframes_count": counts.get("zone_keyframes", 0),
            "devices_count": counts.get("devices", 0),
            "device_refs_count": counts.get("device_refs", 0),
            "observations_count": counts.get("observations", 0),
            "feedback_count": counts.get("feedback", 0),
        },
        "faiss": {
            "zones": len(zone_index.meta),
            "devices": len(device_index.meta),
            "zone_faiss_size": len(zone_index.meta),
            "device_faiss_size": len(device_index.meta),
        },
        "dataset_folders_found": dataset_folders,
        "last_zone_index_build_time": get_index_built_at("zones"),
    }
