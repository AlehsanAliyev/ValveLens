import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.db import fetch_device_refs, fetch_zone_keyframes


try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class FaissIndex:
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.index_path = DATA_DIR / f"{name}.index"
        self.emb_path = DATA_DIR / f"{name}.npy"
        self.meta_path = DATA_DIR / f"{name}_meta.json"
        self.index = None
        self.meta: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def load(self) -> None:
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.emb_path.exists():
            self.embeddings = np.load(self.emb_path)
        if HAS_FAISS and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))

    def save(self, embeddings: np.ndarray, meta: List[Dict]) -> None:
        self.meta = meta
        self.embeddings = embeddings.astype(np.float32)
        np.save(self.emb_path, self.embeddings)
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
        if HAS_FAISS and len(embeddings) > 0:
            index = faiss.IndexFlatIP(self.dim)
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, str(self.index_path))
            self.index = index

    def search(self, query: np.ndarray, topk: int = 5) -> List[Tuple[Dict, float]]:
        if self.index is None and self.embeddings is None:
            self.load()
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        query_vec = query.astype(np.float32).reshape(1, -1)
        if HAS_FAISS and self.index is not None:
            scores, idxs = self.index.search(query_vec, min(topk, len(self.meta)))
            pairs = []
            for idx, score in zip(idxs[0], scores[0]):
                if idx < 0:
                    continue
                pairs.append((self.meta[idx], float(score)))
            return pairs
        emb = self.embeddings
        scores = np.dot(emb, query_vec[0])
        top_idx = np.argsort(scores)[::-1][:topk]
        return [(self.meta[i], float(scores[i])) for i in top_idx]


def rebuild_zone_index(dim: int) -> int:
    rows = fetch_zone_keyframes()
    if not rows:
        FaissIndex("zones", dim).save(np.zeros((0, dim), dtype=np.float32), [])
        return 0
    embeddings = []
    meta = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(emb)
        meta.append({"keyframe_id": row["keyframe_id"], "zone_id": row["zone_id"]})
    vecs = _normalize(np.vstack(embeddings))
    FaissIndex("zones", dim).save(vecs, meta)
    return len(meta)


def rebuild_device_index(dim: int) -> int:
    rows = fetch_device_refs()
    if not rows:
        FaissIndex("devices", dim).save(np.zeros((0, dim), dtype=np.float32), [])
        return 0
    embeddings = []
    meta = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(emb)
        meta.append(
            {
                "ref_id": row["ref_id"],
                "device_id": row["device_id"],
                "zone_id": row.get("zone_id"),
            }
        )
    vecs = _normalize(np.vstack(embeddings))
    FaissIndex("devices", dim).save(vecs, meta)
    return len(meta)
