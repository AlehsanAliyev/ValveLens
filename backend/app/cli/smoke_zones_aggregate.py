import argparse
from pathlib import Path

from PIL import Image

from app.embeddings import Embedder
from app.faiss_store import FaissIndex
from app.zone_localizer import ZoneLocalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    embedder = Embedder()
    zone_index = FaissIndex("zones", embedder.dim)
    zone_index.load()
    if not zone_index.meta:
        raise SystemExit("Zone index empty. Rebuild index first.")

    localizer = ZoneLocalizer(embedder, zone_index)
    image = Image.open(image_path).convert("RGB")
    result = localizer.localize(image=image, topk_keyframes=20, topk_zones=args.topk, aggregate_mode="sum")

    print("Zone candidates:")
    for idx, zone in enumerate(result["zone_candidates"], start=1):
        print(
            f"{idx}. {zone['zone_name']} ({zone['zone_id']}) score={zone['score']:.3f}"
        )
    if result["zone_top1"]:
        top = result["zone_top1"]
        print(f"Top1: {top['zone_name']} ({top['zone_id']}) score={top['score']:.3f}")


if __name__ == "__main__":
    main()
