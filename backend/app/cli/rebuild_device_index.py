from app.embeddings import Embedder
from app.faiss_store import rebuild_device_index


def main() -> None:
    from app import db

    db.init_db()
    embedder = Embedder()
    count = rebuild_device_index(embedder.dim)
    print(f"Indexed {count} device refs.")


if __name__ == "__main__":
    main()
