from app.embeddings import Embedder
from app.faiss_store import rebuild_zone_index


def main() -> None:
    from app import db

    db.init_db()
    embedder = Embedder()
    count = rebuild_zone_index(embedder.dim)
    print(f"Indexed {count} zone keyframes.")


if __name__ == "__main__":
    main()
