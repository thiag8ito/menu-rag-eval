from rag.bm25 import BM25
from rag.chunking import build_chunks
from rag.embeddings import get_embedder
from rag.entity import RestaurantLinker
from rag.index import VectorIndex
from rag.reranker import get_reranker
from rag.retriever import Retriever
from tests.test_dataset import DATASET, DOCS

NAMES = {d["restaurant_id"]: d["name"] for d in DOCS}
LINKER = RestaurantLinker(NAMES)


def test_links_exact_name_ignoring_case_and_accents():
    d = DOCS[0]
    assert LINKER.link(f"quanto custa pizza no {d['name']}?") == d["restaurant_id"]
    assert LINKER.link(f"QUANTO CUSTA no {d['name'].upper()}") == d["restaurant_id"]


def test_tolerates_a_typo_in_the_name():
    d = next(x for x in DOCS if len(x["name"]) >= 12)
    broken = d["name"][:4] + d["name"][5:]  # omite 1 letra
    assert LINKER.link(f"preço de pizza no {broken}") == d["restaurant_id"]


def test_discovery_questions_never_link_to_a_restaurant():
    disc = [q for q in DATASET if q["type"] == "discovery"]
    assert disc and all(LINKER.link(q["question"]) is None for q in disc)


def test_precision_is_high_and_recall_decent_on_named_questions():
    named = [q for q in DATASET if q["answerable"] and q["type"] != "discovery"]
    for variant, min_recall in (("question", 0.9), ("question_noisy", 0.7)):
        got = [(LINKER.link(q[variant]), q["gold_restaurant_ids"][0]) for q in named]
        linked = [(r, g) for r, g in got if r]
        precision = sum(r == g for r, g in linked) / len(linked)
        recall = len(linked) / len(named)
        assert precision >= 0.98, (variant, precision)  # vínculo errado é pior que nenhum vínculo
        assert recall >= min_recall, (variant, recall)


def test_ambiguous_mention_returns_none():
    names = {"a": "Forno Express", "b": "Pizza Express"}
    assert RestaurantLinker(names).link("quanto custa no express?") is None


def _retriever(entity):
    chunks = build_chunks(DOCS, "structured", 1024, True)
    emb = get_embedder("tfidf-char")
    idx = VectorIndex(emb.encode_docs([c.embed_text for c in chunks]), chunks)
    linker = RestaurantLinker({c.restaurant_id: c.restaurant_name for c in chunks}) if entity else None
    return Retriever(idx, emb, get_reranker("none"), 20, BM25([c.embed_text for c in chunks]), linker)


def test_entity_linking_puts_the_cited_restaurant_first_and_improves_hit_at_1():
    qs = [q for q in DATASET if q["type"] in ("price", "ingredients")]
    plain, linked = _retriever(False), _retriever(True)

    def hit1(retr):
        return sum(bool(set(q["gold_dish_ids"]) & set(retr.search(q["question"], 1)[0][0].chunk.dish_ids))
                   for q in qs) / len(qs)

    assert hit1(linked) >= hit1(plain)
    for q in qs:  # com vínculo, o top-1 pertence sempre ao restaurante citado
        top = linked.search(q["question"], 1)[0][0]
        assert top.chunk.restaurant_id == q["gold_restaurant_ids"][0]


def test_search_batch_with_linker_matches_single_search():
    retr = _retriever(True)
    qs = [q["question_noisy"] for q in DATASET[:8]]
    for q, hits in zip(qs, retr.search_batch(qs, k=5)):
        assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in retr.search(q, 5)[0]]


def test_regression_common_words_must_not_complete_a_restaurant_name():
    """Bug real: trigramas de 'Tem', 'esperança', 'bairro' e 'brasileira' vinculavam a 'Tempero Brasil'."""
    linker = RestaurantLinker({"a": "Tempero Brasil", "b": "Grill Brasil", "c": "Massa Fina Brasil"})
    q = "Tem alguma opção vegetariana de comida brasileira no bairro Boa Esperança, Marília?"
    assert linker.link(q) is None
    assert linker.link("quanto custa feijoada no tempero brasil?") == "a"
    assert linker.link("quanto custa feijoada no tepero brasil?") == "a"  # typo ainda vincula
