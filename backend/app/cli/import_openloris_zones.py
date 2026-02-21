import argparse
from pathlib import Path
from typing import List

from PIL import Image

from app import db
from app.embeddings import Embedder
from app.faiss_store import rebuild_zone_index


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SCENES = ["corridor", "office", "station"]


def _list_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    files.sort(key=lambda p: str(p).lower())
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--max_per_zone", type=int, default=300)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    db.init_db()
    embedder = Embedder()

    zones_created = 0
    zones_existing = 0
    images_added = 0
    images_skipped = 0

    for scene in SCENES:
        scene_dir = root / scene
        if not scene_dir.exists() or not scene_dir.is_dir():
            continue
        for child in sorted([p for p in scene_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            zone_name = f"OpenLORIS_{scene}_{child.name}"
            zone_desc = f"OpenLORIS scene {scene}, folder {child.name}"
            row = db.get_zone_by_name(zone_name)
            if row:
                zone_id = row["zone_id"]
                zones_existing += 1
            else:
                zone_id = db.create_zone(zone_name, zone_desc)
                zones_created += 1

            images = _list_images(child)
            if args.max_per_zone > 0:
                images = images[: args.max_per_zone]

            for image_path in images:
                stable_path = str(image_path.resolve())
                if db.zone_keyframe_exists(stable_path):
                    images_skipped += 1
                    continue
                try:
                    img = Image.open(image_path).convert("RGB")
                    emb = embedder.embed_image(img).astype("float32").tobytes()
                    db.add_zone_keyframe(zone_id, stable_path, embedder.embedding_type, emb)
                    images_added += 1
                except Exception:
                    images_skipped += 1

    print(f"zones_created={zones_created}")
    print(f"zones_existing={zones_existing}")
    print(f"images_added={images_added}")
    print(f"images_skipped={images_skipped}")

    if args.rebuild:
        count = rebuild_zone_index(embedder.dim)
        print(f"zone_index_size={count}")


if __name__ == "__main__":
    main()
