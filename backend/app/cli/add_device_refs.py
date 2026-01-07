import argparse
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app import db
from app.embeddings import Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()

    db.init_db()
    embedder = Embedder()
    folder = Path(args.folder)
    device_dir = (
        Path(__file__).resolve().parent.parent.parent / "data" / "devices" / args.device_id
    )
    device_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in folder.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(path).convert("RGB")
        filename = f"{uuid4()}{path.suffix.lower()}"
        save_path = device_dir / filename
        save_path.write_bytes(path.read_bytes())
        emb = embedder.embed_image(img).astype("float32").tobytes()
        db.add_device_ref(args.device_id, str(save_path), embedder.embedding_type, emb)
        count += 1

    print(f"Added {count} device refs.")


if __name__ == "__main__":
    main()
