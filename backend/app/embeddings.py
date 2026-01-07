from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _simple_embed(image: Image.Image, dim: int = 512) -> np.ndarray:
    img = image.convert("RGB").resize((32, 32))
    arr = np.asarray(img).astype(np.float32) / 255.0
    flat = arr.reshape(-1)
    chunk = int(np.ceil(len(flat) / dim))
    emb = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        start = i * chunk
        end = min(len(flat), (i + 1) * chunk)
        if start >= len(flat):
            break
        emb[i] = float(np.mean(flat[start:end]))
    return _l2_normalize(emb)


@dataclass
class Embedder:
    embedding_type: str = "clip_vitb32"
    dim: int = 512
    model: Optional[object] = None
    preprocess: Optional[object] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            import open_clip

            result = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model = result[0]
            preprocess = result[2] if len(result) > 2 else result[1]
            model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(self.device)
            self.model = model
            self.preprocess = preprocess
            self.dim = model.visual.output_dim
        except Exception:
            self.model = None
            self.preprocess = None
            self.dim = 512

    def embed_image(self, image: Image.Image) -> np.ndarray:
        if self.model is None or self.preprocess is None:
            return _simple_embed(image, dim=self.dim)
        import torch

        img = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(img)
        vec = feats.cpu().numpy().astype(np.float32)[0]
        return _l2_normalize(vec)
