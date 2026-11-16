import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rag.bm25 import BM25, rrf
from rag.chunking import Chunk
from rag.entity import RestaurantLinker


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """embed(query) -> top-N densos [+ BM25 por RRF] [+ prioriza o restaurante citado] -> (rerank) -> top-k."""

    def __init__(self, index, embedder, reranker, candidates_k: int = 20, bm25: Optional[BM25] = None,
                 linker: Optional[RestaurantLinker] = None):
        self.index, self.embedder, self.reranker = index, embedder, reranker
        self.candidates_k, self.bm25, self.linker = candidates_k, bm25, linker
        self._by_restaurant = {}
        if linker is not None:
            for i, c in enumerate(index.chunks):
                self._by_restaurant.setdefault(c.restaurant_id, []).append(i)

    @property
    def hybrid(self) -> bool:
        return self.bm25 is not None

    def _candidates(self, query, qvec, n):
        idxs, scores = self.index.search(qvec, n)
        if self.bm25 is not None:
            b_idx, _ = self.bm25.search(query, n)
            idxs, scores = rrf([idxs, b_idx], n)
        if self.linker is not None:
            rid = self.linker.link(query)
            if rid is not None:  # o usuário citou um restaurante: seus chunks vêm primeiro (ordem do denso)
                full = self.index.scores(qvec)
                own = np.array(self._by_restaurant[rid])
                own = own[np.argsort(-full[own], kind="stable")]
                own_set = set(own.tolist())
                idxs = np.array(own.tolist() + [i for i in idxs.tolist() if i not in own_set])
                idxs = idxs[:max(n, len(own))]
                scores = full[idxs]
        return idxs, scores

    def _finish(self, query, idxs, scores, k):
        cands = [self.index.chunks[i] for i in idxs]
        order, new_scores = self.reranker.rerank(query, cands, list(scores))
        return [Hit(cands[i], float(s), r) for r, (i, s) in enumerate(zip(order, new_scores))][:k]

    def search(self, query: str, k: int = 5):
        n = max(self.candidates_k, k)
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([query])[0]
        t1 = time.perf_counter()
        idxs, scores = self._candidates(query, qvec, n)
        t2 = time.perf_counter()
        hits = self._finish(query, idxs, scores, k)
        t3 = time.perf_counter()
        return hits, {"embed_ms": (t1 - t0) * 1e3, "search_ms": (t2 - t1) * 1e3,
                      "rerank_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}

    def search_batch(self, queries, k: int = 10):
        """Versão em lote (avaliação de qualidade): codifica todas as queries de uma vez."""
        n = max(self.candidates_k, k)
        qvecs = self.embedder.encode_queries(list(queries))
        return [self._finish(q, *self._candidates(q, qvecs[i], n), k) for i, q in enumerate(queries)]
