import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image

from app import db
from app.datasets import zone_importers
from app.embeddings import Embedder
from app.faiss_store import rebuild_zone_index


def _sample_images(paths: List[str], max_per_zone: int) -> List[str]:
    ordered = sorted(paths)
    if max_per_zone <= 0 or len(ordered) <= max_per_zone:
        return ordered
    return ordered[:max_per_zone]


def _save_manifest(dataset: str, zone_specs: List[dict]) -> None:
    manifest_dir = Path("data_sources/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{dataset}_zones.json"
    manifest_path.write_text(json.dumps(zone_specs, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nyc_indoor_vpr", "openloris_location", "cold"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--max_per_zone", type=int, default=300)
    parser.add_argument("--per_location", action="store_true", default=True)
    parser.add_argument("--per_scene", action="store_true", default=False)
    parser.add_argument("--copy_to_backend_data", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    db.init_db()
    embedder = Embedder()

    if args.dataset == "nyc_indoor_vpr":
        zone_specs = zone_importers.scan_nyc_indoor_vpr(args.root)
    elif args.dataset == "openloris_location":
        per_location = True
        if args.per_scene:
            per_location = False
        zone_specs = zone_importers.scan_openloris_location(args.root, per_location=per_location)
    else:
        zone_specs = zone_importers.scan_cold_subset(args.root)

    if not zone_specs:
        print("No zones found. Check the dataset root path.")
        return

    _save_manifest(args.dataset, zone_specs)

    added = 0
    skipped = 0
    for zone in zone_specs:
        existing_zone = db.get_zone_by_name(zone["zone_name"])
        if existing_zone:
            zone_id = existing_zone["zone_id"]
        else:
            zone_id = db.create_zone(zone["zone_name"], zone["description"])

        image_paths = _sample_images(zone["image_paths"], args.max_per_zone)
        zone_dir = (
            Path("backend/data/zones") / zone_id if args.copy_to_backend_data else None
        )
        if zone_dir:
            zone_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            source_path = Path(img_path)
            if not source_path.exists():
                continue
            if args.copy_to_backend_data:
                dest = zone_dir / source_path.name
                if not dest.exists():
                    dest.write_bytes(source_path.read_bytes())
                image_path = str(dest)
            else:
                image_path = str(source_path)

            if db.zone_keyframe_exists(image_path):
                skipped += 1
                continue

            img = Image.open(source_path).convert("RGB")
            emb = embedder.embed_image(img).astype("float32").tobytes()
            db.add_zone_keyframe(zone_id, image_path, embedder.embedding_type, emb)
            added += 1

    print(f"Added {added} keyframes. Skipped {skipped} existing.")

    if args.rebuild:
        count = rebuild_zone_index(embedder.dim)
        print(f"Rebuilt zone index with {count} vectors.")


if __name__ == "__main__":
    main()
