from typing import Dict, List, Tuple

from PIL import Image

from app.embeddings import Embedder


class ReIDEmbedder:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed(self, image: Image.Image) -> Dict:
        vec = self.embedder.embed_image(image)
        return {
            "embedding_type": self.embedder.embedding_type,
            "embedding": vec,
        }


def group_matches_by_device(matches: List[Tuple[Dict, float]]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for meta, score in matches:
        device_id = str(meta.get("device_id") or "").strip()
        if not device_id:
            continue
        score_value = float(score)
        item = grouped.setdefault(
            device_id,
            {
                "device_id": device_id,
                "scores": [],
                "best_score": score_value,
                "best_reference_image": meta.get("image_path"),
                "best_meta": meta,
            },
        )
        item["scores"].append(score_value)
        if score_value > float(item["best_score"]):
            item["best_score"] = score_value
            item["best_reference_image"] = meta.get("image_path")
            item["best_meta"] = meta

    rows: List[Dict] = []
    for item in grouped.values():
        scores = item["scores"]
        rows.append(
            {
                "device_id": item["device_id"],
                "score": float(item["best_score"]),
                "best_score": float(item["best_score"]),
                "mean_score": float(sum(scores) / len(scores)),
                "ref_count": len(scores),
                "best_reference_image": item.get("best_reference_image"),
                "best_meta": item.get("best_meta") or {},
            }
        )
    return sorted(rows, key=lambda row: row["best_score"], reverse=True)
