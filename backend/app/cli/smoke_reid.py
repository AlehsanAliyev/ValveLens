import argparse
from pathlib import Path

from PIL import Image

from app.embeddings import Embedder
from app.faiss_store import FaissIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--zone_id", default=None)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    embedder = Embedder()
    index = FaissIndex("devices", embedder.dim)
    index.load()
    if not index.meta:
        raise SystemExit("Device index is empty. Run rebuild_device_index first.")

    image = Image.open(image_path).convert("RGB")
    vec = embedder.embed_image(image)

    search_topk = max(args.topk, 20)
    matches = index.search(vec, topk=search_topk)
    if args.zone_id:
        zone_filtered = [m for m in matches if m[0].get("zone_id") == args.zone_id]
        if zone_filtered:
            matches = zone_filtered

    print("Top device matches:")
    for rank, (meta, score) in enumerate(matches[: args.topk], start=1):
        print(
            f"{rank}. device_id={meta.get('device_id')} "
            f"score={float(score):.4f} "
            f"zone_id={meta.get('zone_id')} "
            f"image_path={meta.get('image_path')}"
        )


if __name__ == "__main__":
    main()
