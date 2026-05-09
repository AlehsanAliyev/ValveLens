import argparse
import csv
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from PIL import Image

from app import db
from app.embeddings import Embedder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "devices"


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _device_ref_count(device_id: str) -> int:
    conn = db.get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM device_refs WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def _iter_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda item: item.name.lower())


def _add_refs(device_id: str, source_folder: Path, embedder: Embedder) -> int:
    images = _iter_images(source_folder)
    target_dir = DEVICE_DATA_DIR / device_id
    target_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for image_path in images:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"WARNING: could not read {image_path}: {exc}")
            continue

        target = target_dir / f"{uuid4()}{image_path.suffix.lower()}"
        target.write_bytes(image_path.read_bytes())
        embedding = embedder.embed_image(image).astype("float32").tobytes()
        db.add_device_ref(device_id, str(target), embedder.embedding_type, embedding)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll benchmark devices from a CSV manifest."
    )
    parser.add_argument("--manifest", required=True, help="devices_manifest.csv path")
    parser.add_argument(
        "--refs-root",
        default=None,
        help="Optional root containing refs/<device_id> folders or direct device folders.",
    )
    parser.add_argument(
        "--force-add-refs",
        action="store_true",
        help="Add references even if the device already has refs in the DB.",
    )
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    refs_root = _resolve(args.refs_root) if args.refs_root else None
    db.init_db()
    embedder = Embedder() if refs_root else None

    created = 0
    existing = 0
    refs_found = 0
    refs_added = 0
    warnings: List[str] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"device_id", "type", "zone_id", "description", "has_visible_tag"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise SystemExit(
                "Manifest is missing columns: " + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            device_id = (row.get("device_id") or "").strip()
            device_type = (row.get("type") or "").strip()
            zone_id = (row.get("zone_id") or "").strip()
            description = (row.get("description") or "").strip()
            if not device_id:
                warnings.append("Skipped row with empty device_id.")
                continue
            if not zone_id:
                warnings.append(f"{device_id}: missing zone_id.")
                continue

            if db.get_device(device_id):
                existing += 1
                print(f"EXISTS device_id={device_id}")
            else:
                db.create_device(device_id, zone_id, device_type, description)
                created += 1
                print(f"CREATED device_id={device_id}")

            if refs_root:
                candidates = [refs_root / device_id, refs_root / "refs" / device_id]
                source_folder = next((path for path in candidates if path.exists()), candidates[0])
                images = _iter_images(source_folder)
                refs_found += len(images)
                if not source_folder.exists():
                    warnings.append(f"{device_id}: refs folder missing: {source_folder}")
                    continue
                if not images:
                    warnings.append(f"{device_id}: no reference images found in {source_folder}")
                    continue
                current_refs = _device_ref_count(device_id)
                if current_refs > 0 and not args.force_add_refs:
                    print(
                        f"SKIP refs device_id={device_id} existing_refs={current_refs} "
                        "use --force-add-refs to add more"
                    )
                    continue
                added = _add_refs(device_id, source_folder, embedder)
                refs_added += added
                print(f"ADDED refs device_id={device_id} count={added}")

    summary: Dict[str, int] = {
        "devices_created": created,
        "devices_already_existing": existing,
        "reference_images_found": refs_found,
        "reference_images_added": refs_added,
        "warnings": len(warnings),
    }

    print("\nEnrollment summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print("\nNext command:")
    print("  python -m app.cli.rebuild_device_index")


if __name__ == "__main__":
    main()
