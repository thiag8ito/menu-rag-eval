import math
from evaluation.metrics import (cohen_kappa, hit_at_k, ndcg_at_k, paired_bootstrap,
                                recall_at_k, reciprocal_rank)


def test_hit_and_rr():
    rel = [False, False, True, False]
    assert not hit_at_k(rel, 2) and hit_at_k(rel, 3)
    assert math.isclose(reciprocal_rank(rel), 1 / 3)
    assert reciprocal_rank([False] * 3) == 0.0


def test_ndcg_perfect_and_worst():
    assert math.isclose(ndcg_at_k([True, False, False], 3), 1.0)
    assert 0 < ndcg_at_k([False, False, True], 3) < 1


def test_recall_multi_gold():
    ranked = [["a", "x"], ["b"], ["c"]]
    assert math.isclose(recall_at_k(ranked, {"a", "b", "z"}, 2), 2 / 3)


def test_paired_bootstrap_detects_gain():
    a = [1.0] * 50 + [0.5] * 50
    b = [0.5] * 50 + [0.5] * 50
    d, (lo, hi), p = paired_bootstrap(a, b)
    assert d > 0 and lo > 0 and p < 0.05


def test_kappa():
    assert math.isclose(cohen_kappa([True, False, True], [True, False, True]), 1.0)
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) <= 0.0
