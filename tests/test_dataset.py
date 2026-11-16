"""Invariantes do dataset: um gabarito errado invalida TODAS as métricas, então é o que mais vale testar."""
import random
import re

from evaluation.build_dataset import build_dataset
from evaluation.noise import perturb, strip_accents, typo
from rag.generate_menus import CUISINES, build_restaurant


def make_corpus(n=40, seed=3):
    rng, used, cs = random.Random(seed), set(), list(CUISINES)
    return [build_restaurant(i, cs[i % len(cs)], rng, used) for i in range(n)]


DOCS = make_corpus()
DATASET = build_dataset(DOCS, n=200, seed=5)
DISH = {it["dish_id"]: (it, d) for d in DOCS for it in d["items"]}
REST = {d["restaurant_id"]: d for d in DOCS}


def test_all_types_present_and_ids_unique():
    assert {q["type"] for q in DATASET} == {"price", "ingredients", "dietary", "discovery", "unanswerable"}
    assert len({q["qid"] for q in DATASET}) == len(DATASET)
    assert {q["split"] for q in DATASET} == {"dev", "test"}


def test_gold_ids_exist_and_belong_to_gold_restaurants():
    for q in DATASET:
        for g in q["gold_dish_ids"]:
            _, d = DISH[g]
            assert d["restaurant_id"] in q["gold_restaurant_ids"]
        assert bool(q["gold_dish_ids"]) == q["answerable"]


def test_price_question_names_its_restaurant_and_reference_matches_menu():
    for q in (x for x in DATASET if x["type"] == "price"):
        it, d = DISH[q["gold_dish_ids"][0]]
        assert d["name"] in q["question"] and it["name"] in q["question"]
        assert f"{it['price']:.2f}".replace(".", ",") in q["reference_answer"]


def test_discovery_excludes_beverages_and_matches_filters():
    """Regressão: antes, 'opção vegana de pizza' tinha refrigerante no gabarito."""
    disc = [q for q in DATASET if q["type"] == "discovery"]
    assert disc
    for q in disc:
        for g in q["gold_dish_ids"]:
            assert DISH[g][0]["category"] != "Bebidas"
        for r in q["gold_restaurant_ids"]:
            assert REST[r]["neighborhood"] in q["question"] and REST[r]["city"] in q["question"]


def test_discovery_has_no_restaurant_name():
    names = [d["name"] for d in DOCS]
    for q in (x for x in DATASET if x["type"] == "discovery"):
        assert not any(n in q["question"] for n in names)


def test_unanswerable_dish_is_really_absent_from_that_restaurant():
    n_checked = 0
    for q in (x for x in DATASET if x["type"] == "unanswerable"):
        owner = next(d for d in DOCS if d["name"] in q["question"])
        for it in DOCS_ITEMS_IN_QUESTION(q["question"]):
            assert it not in {i["name"] for i in owner["items"]}
            n_checked += 1
    assert n_checked > 0


def DOCS_ITEMS_IN_QUESTION(question):
    """Nomes de prato citados na pergunta, ignorando os que só aparecem dentro do nome do restaurante."""
    all_names = {i["name"] for d in DOCS for i in d["items"]}
    stripped = question
    for d in DOCS:
        stripped = stripped.replace(d["name"], "")
    return [n for n in all_names if n in stripped]


def test_no_grammar_artifacts():
    for q in DATASET:
        t = q["question"]
        assert "vegana vegana" not in t and "vegano vegano" not in t
        assert not re.search(r"hambúrguer (vegana|vegetariana)", t)


def test_clean_and_noisy_are_different_but_same_request():
    diff = sum(q["question"] != q["question_noisy"] for q in DATASET)
    assert diff / len(DATASET) > 0.95
    for q in (x for x in DATASET if x["type"] in ("price", "ingredients")):
        it, d = DISH[q["gold_dish_ids"][0]]
        # perturb() faz typo em no máximo 2 palavras; logo, no máximo 2 palavras (5+ letras) dos
        # nomes de restaurante/prato podem ter sido alteradas: o pedido continua identificável.
        noisy = set(re.findall(r"\w+", strip_accents(q["question_noisy"].lower())))
        words = {strip_accents(w.lower()) for w in re.findall(r"\w+", d["name"] + " " + it["name"]) if len(w) >= 5}
        assert len(words - noisy) <= 2, (q["question_noisy"], words - noisy)


def test_dataset_is_deterministic():
    assert build_dataset(DOCS, n=50, seed=5) == build_dataset(DOCS, n=50, seed=5)
    assert build_dataset(DOCS, n=50, seed=5) != build_dataset(DOCS, n=50, seed=6)


# ---- ruído ----
def test_typo_keeps_first_letter_and_changes_word():
    rng = random.Random(0)
    changed = 0
    for w in ["margherita", "calabresa", "hambúrguer", "vegetariana", "sorvete"]:
        for _ in range(20):
            t = typo(w, rng)
            assert t[0] == w[0]
            changed += t != w
    assert changed > 80  # quase sempre altera (neighbor/dup podem coincidir raramente)


def test_perturb_is_deterministic_per_seed_and_short_words_untouched():
    a = perturb("Quanto custa Pizza Margherita no Forno Express?", random.Random(1))
    b = perturb("Quanto custa Pizza Margherita no Forno Express?", random.Random(1))
    assert a == b
    assert perturb("o x tem?", random.Random(2)).lower().startswith("o x t")


def test_strip_accents():
    assert strip_accents("Açaí é bão") == "Acai e bao"
