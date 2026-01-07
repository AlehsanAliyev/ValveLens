from typing import Dict

from PIL import Image

from app.embeddings import Embedder


class ReIDEmbedder:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed(self, image: Image.Image) -> Dict:
        vec = self.embedder.embed_image(image)
        return {
            "embedding_type": self.embedder.embedding_type,
            "embedding": vec,
        }
