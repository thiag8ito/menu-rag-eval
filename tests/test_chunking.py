import json
from rag.chunking import build_chunks
from rag.generate_menus import build_restaurant, CUISINES
import random


def make_docs(n=8):
    rng, used = random.Random(1), set()
    cs = list(CUISINES)
    return [build_restaurant(i, cs[i % len(cs)], rng, used) for i in range(n)]


def test_structured_never_splits_a_dish():
    for size in (200, 512, 1024):
        docs = make_docs()
        chunks = build_chunks(docs, "structured", size)
        covered = {r: set() for r in {d["restaurant_id"] for d in docs}}
        for c in chunks:
            covered[c.restaurant_id] |= set(c.dish_ids)
        for d in docs:
            # com chunking estruturado, todo prato aparece INTEIRO em ao menos um chunk
            assert covered[d["restaurant_id"]] == {it["dish_id"] for it in d["items"]}, size


def test_fixed_can_split_dishes_at_small_sizes():
    docs = make_docs()
    chunks = build_chunks(docs, "fixed", 120)
    total = sum(len(d["items"]) for d in docs)
    covered = {x for c in chunks for x in c.dish_ids}
    assert len(covered) < total  # chunk pequeno e cego à estrutura perde pratos inteiros


def test_contextual_header_only_affects_embed_text():
    docs = make_docs(2)
    with_h = build_chunks(docs, "structured", 512, True)
    without = build_chunks(docs, "structured", 512, False)
    assert [c.text for c in with_h] == [c.text for c in without]
    assert all(docs[0]["name"] in c.embed_text for c in with_h if c.restaurant_id == "r000")
