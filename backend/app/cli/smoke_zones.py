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
    matches = index.search(vec, topk=args.topk)
    for meta, score in matches:
        print(f"{meta.get('zone_id')}  score={score:.3f}  image={meta.get('image_path')}")


if __name__ == "__main__":
    main()
