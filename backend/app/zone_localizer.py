from typing import Dict, List, Optional

from PIL import Image

from app.db import fetch_zone_name_map
from app.embeddings import Embedder
from app.faiss_store import FaissIndex


class ZoneLocalizer:
    def __init__(self, embedder: Embedder, zone_index: FaissIndex) -> None:
        self.embedder = embedder
        self.zone_index = zone_index

    def localize(
        self,
        image: Image.Image,
        topk_keyframes: int = 20,
        topk_zones: int = 5,
        aggregate_mode: str = "sum",
    ) -> Dict:
        zone_name_map = fetch_zone_name_map()
        vec = self.embedder.embed_image(image)
        matches = self.zone_index.search(vec, topk=topk_keyframes)

        top_k_keyframes: List[Dict] = []
        zone_scores: Dict[str, float] = {}
        for meta, score in matches:
            zone_id = meta.get("zone_id")
            zone_name = zone_name_map.get(zone_id, zone_id) if zone_id else None
            score_f = float(score)
            top_k_keyframes.append(
                {
                    "image_path": meta.get("image_path"),
                    "score": score_f,
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                }
            )
            if not zone_id:
                continue
            if aggregate_mode == "max":
                zone_scores[zone_id] = max(zone_scores.get(zone_id, float("-inf")), score_f)
            else:
                zone_scores[zone_id] = zone_scores.get(zone_id, 0.0) + score_f

        ranked = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)[:topk_zones]
        zone_candidates = [
            {
                "zone_id": zone_id,
                "zone_name": zone_name_map.get(zone_id, zone_id),
                "score": float(score),
            }
            for zone_id, score in ranked
        ]
        zone_top1: Optional[Dict] = zone_candidates[0] if zone_candidates else None

        return {
            "top_k_keyframes": top_k_keyframes,
            "zone_candidates": zone_candidates,
            "zone_top1": zone_top1,
        }
