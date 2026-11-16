import json
import tempfile

import numpy as np

from rag.bm25 import BM25, rrf
from rag.chunking import build_chunks
from rag.embeddings import get_embedder, load_embedder
from rag.index import VectorIndex
from rag.reranker import get_reranker
from rag.retriever import Retriever
from tests.test_dataset import DATASET, DOCS


def build(emb_name="tfidf", strategy="structured", size=1024, header=True, hybrid=False):
    chunks = build_chunks(DOCS, strategy, size, header)
    emb = get_embedder(emb_name)
    index = VectorIndex(emb.encode_docs([c.embed_text for c in chunks]), chunks)
    bm25 = BM25([c.embed_text for c in chunks]) if hybrid else None
    return Retriever(index, emb, get_reranker("none"), 20, bm25), chunks


def hit_rate(retr, qs, k=5):
    ok = 0
    for q in qs:
        hits, _ = retr.search(q["question"], k)
        ok += bool(set(q["gold_dish_ids"]) & {d for h in hits for d in h.chunk.dish_ids})
    return ok / len(qs)


def test_exact_price_questions_are_retrievable():
    retr, _ = build()
    qs = [q for q in DATASET if q["type"] == "price"][:20]
    assert hit_rate(retr, qs) >= 0.9


def test_contextual_header_is_what_makes_discovery_work():
    """Sem o nome/bairro no chunk, discovery (sem nome de restaurante) tem que piorar."""
    qs = [q for q in DATASET if q["type"] == "discovery"]
    with_h = hit_rate(build(header=True, size=512)[0], qs)
    without = hit_rate(build(header=False, size=512)[0], qs)
    assert with_h > without


def test_search_returns_sorted_scores_and_k_items():
    retr, _ = build()
    hits, timings = retr.search("pizza vegana de cogumelos", 4)
    assert len(hits) == 4
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in hits] == [0, 1, 2, 3] and timings["total_ms"] >= 0


def test_search_batch_matches_single_search():
    retr, _ = build(hybrid=True)
    qs = [q["question"] for q in DATASET[:5]]
    batch = retr.search_batch(qs, k=5)
    for q, hits in zip(qs, batch):
        single, _ = retr.search(q, 5)
        assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in single]


def test_bm25_ranks_exact_term_first_and_ignores_accents():
    bm = BM25(["feijoada completa com couve", "pizza de calabresa", "salada fit com frango"])
    assert bm.search("feijoada", 1)[0][0] == 0
    assert bm.search("Calabresa", 1)[0][0] == 1
    assert bm.search("açaí", 3)[1].max() == 0.0  # termo fora do vocabulário => score 0


def test_rrf_prefers_items_ranked_high_by_both():
    idx, sc = rrf([np.array([3, 1, 2]), np.array([1, 3, 0])], k=3)
    assert set(idx[:2]) == {1, 3} and sc[0] >= sc[1] >= sc[2]


def test_hybrid_never_loses_candidates_it_should_keep():
    """O híbrido fundido por RRF mantém o top-1 de qualquer um dos dois retrievers entre os candidatos."""
    retr, _ = build(hybrid=True)
    q = "feijoada completa"
    dense_top = retr.index.search(retr.embedder.encode_queries([q])[0], 1)[0][0]
    bm_top = retr.bm25.search(q, 1)[0][0]
    idxs, _ = retr._candidates(q, retr.embedder.encode_queries([q])[0], 20)
    assert dense_top in idxs and bm_top in idxs


def test_char_ngrams_are_more_robust_to_typos_than_word_tfidf():
    word, _ = build("tfidf", size=512)
    char, _ = build("tfidf-char", size=512)
    qs = [q for q in DATASET if q["type"] in ("price", "ingredients")][:40]

    def noisy_hit(retr):
        ok = 0
        for q in qs:
            hits, _ = retr.search(q["question_noisy"], 5)
            ok += bool(set(q["gold_dish_ids"]) & {d for h in hits for d in h.chunk.dish_ids})
        return ok / len(qs)

    assert noisy_hit(char) >= noisy_hit(word)


def test_lsa_embeddings_are_unit_norm():
    emb = get_embedder("lsa")
    v = emb.encode_docs([c.embed_text for c in build_chunks(DOCS, "structured", 512, True)])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(emb.encode_queries(["pizza vegana"]), axis=1), 1.0, atol=1e-4)


def test_index_roundtrip_gives_identical_results_for_sparse_and_dense():
    for name in ("tfidf", "lsa"):
        retr, _ = build(name)
        with tempfile.TemporaryDirectory() as d:
            retr.index.save(d, {"embedder": name})
            retr.embedder.save(d)
            index2, meta = VectorIndex.load(d)
            emb2 = load_embedder(meta["embedder"], d)
            retr2 = Retriever(index2, emb2, get_reranker("none"), 20)
            for q in ("quanto custa pizza margherita", "opção vegana sem glúten"):
                a = [h.chunk.chunk_id for h in retr.search(q, 5)[0]]
                b = [h.chunk.chunk_id for h in retr2.search(q, 5)[0]]
                assert a == b, name
            assert json.loads(open(f"{d}/meta.json").read())["embedder"] == name
