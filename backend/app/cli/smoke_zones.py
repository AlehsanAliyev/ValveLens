import argparse
from pathlib import Path

from PIL import Image

from app.embeddings import Embedder
from app.faiss_store import FaissIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    embedder = Embedder()
    index = FaissIndex("zones", embedder.dim)
    index.load()
    if not index.meta:
        raise SystemExit("Zone index empty. Rebuild index first.")

    img = Image.open(image_path).convert("RGB")
    vec = embedder.embed_image(img)
    matches = index.search(vec, topk=max(20, args.topk))
    zone_scores = {}
    zone_sample_image = {}
    for meta, score in matches:
        zone_id = meta.get("zone_id")
        if not zone_id:
            continue
        zone_scores[zone_id] = zone_scores.get(zone_id, 0.0) + float(score)
        zone_sample_image.setdefault(zone_id, meta.get("image_path"))
    ranked = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)[: args.topk]
    for zone_id, score in ranked:
        print(f"{zone_id}  score={score:.3f}  sample={zone_sample_image.get(zone_id)}")


if __name__ == "__main__":
    main()
