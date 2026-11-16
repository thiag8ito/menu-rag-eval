import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from rag.chunking import Chunk


class VectorIndex:
    """Índice em memória (denso ou esparso). Vetores normalizados => produto interno = cosseno.
    Para ~1e3-1e5 chunks, força bruta é suficiente; para escalar, trocar por FAISS/pgvector."""

    def __init__(self, vectors, chunks: list):
        assert vectors.shape[0] == len(chunks)
        self.vectors, self.chunks = vectors, chunks

    def scores(self, qvec) -> np.ndarray:
        if sp.issparse(self.vectors):
            return np.asarray((self.vectors @ qvec.T).todense()).ravel()
        return self.vectors @ qvec

    def search(self, qvec, k: int):
        s = self.scores(qvec)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]

    def save(self, directory: str, meta: dict):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if sp.issparse(self.vectors):
            sp.save_npz(d / "vectors.npz", self.vectors.tocsr())
        else:
            np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str):
        d = Path(directory)
        vectors = sp.load_npz(d / "vectors.npz") if (d / "vectors.npz").exists() else np.load(d / "vectors.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(vectors, chunks), json.loads((d / "meta.json").read_text())
