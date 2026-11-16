"""Métricas de retrieval + estatística (bootstrap) + concordância do juiz.

Convenção: `rel` é uma lista de bool, na ordem do ranking, indicando se cada
chunk recuperado contém evidência relevante (ver build_dataset/chunking).
"""
import math
from typing import Optional, Sequence

import numpy as np


def hit_at_k(rel: Sequence[bool], k: int) -> bool:
    return any(rel[:k])


def reciprocal_rank(rel: Sequence[bool], k: Optional[int] = None) -> float:
    for i, r in enumerate(rel[:k] if k else rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_dish_ids: Sequence[Sequence[str]], gold: set, k: int) -> float:
    """Fração dos pratos-gold cobertos pelos top-k chunks (importa quando há vários relevantes)."""
    if not gold:
        return 0.0
    covered = set()
    for ids in ranked_dish_ids[:k]:
        covered |= set(ids) & gold
    return len(covered) / len(gold)


def ndcg_at_k(rel: Sequence[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
    n_rel = min(sum(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg else 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """IC percentil (95%) para a média, reamostrando perguntas (vetorizado)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0):
    """Compara dois sistemas nas MESMAS perguntas.
    Retorna (diff médio a-b, IC95% da diferença, P(diff<=0) ~ p-valor unilateral)."""
    assert len(a) == len(b)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
            float((means <= 0).mean()))


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0
