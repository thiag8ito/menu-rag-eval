"""Gera o dataset de avaliação com gabarito derivado dos dados ESTRUTURADOS (gold exato).

Tipos de pergunta:
  price / ingredients : 1 prato-gold; o restaurante vem na pergunta (hard negatives: o mesmo prato
                        existe em vários restaurantes com preços diferentes).
  dietary             : todos os pratos do restaurante com a restrição (lista).
  discovery           : SEM nome de restaurante (culinária + bairro + cidade + restrição). Só funciona se
                        o chunk carregar contexto do restaurante. Bebidas ficam fora do gabarito.
  unanswerable        : informação que o corpus não tem -> o sistema deve se abster.

Cada pergunta tem duas versões do MESMO pedido:
  question        template "limpo" (bem escrito)
  question_noisy  reformulação coloquial + ruído de caracteres (evaluation/noise.py)

Split dev/test (50/50): escolha a configuração no dev, reporte no test.
Limitação: ainda é gerado por templates; a variante ruidosa reduz, mas não elimina, o viés lexical.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluation.noise import perturb
from rag.generate_menus import CUISINES, money

# tag -> (masculino, feminino, plural feminino)
TAG_FORMS = {"vegano": ("vegano", "vegana", "veganas"), "vegetariano": ("vegetariano", "vegetariana", "vegetarianas"),
             "sem glúten": ("sem glúten",) * 3, "sem lactose": ("sem lactose",) * 3,
             "picante": ("picante", "picante", "picantes"), "fit": ("fit", "fit", "fit")}
# culinária -> (expressão, gênero)
CUISINE_PHRASE = {"Pizzaria": ("pizza", "f"), "Japonesa": ("comida japonesa", "f"),
                  "Hamburgueria": ("hambúrguer", "m"), "Brasileira": ("comida brasileira", "f"),
                  "Vegana": ("comida vegana", "f"), "Açaí e Sobremesas": ("sobremesa", "f"),
                  "Árabe": ("comida árabe", "f"), "Massas": ("massa", "f")}

CLEAN = {
    "price": ["Quanto custa {d} no {r}?", "Qual o preço de {d} no {r}?",
              "Quero saber o valor do prato {d} do {r}.", "{d} no {r} sai por quanto?"],
    "ingredients": ["O que vem no prato {d} do {r}?", "Quais ingredientes tem {d} no {r}?",
                    "Do que é feito {d} do {r}?", "Me conta a composição de {d} no {r}."],
    "dietary": ["O {r} tem opção {a}?", "Quais opções {p} o {r} serve?", "Tem alguma opção {a} no cardápio do {r}?"],
    "discovery": ["Tem alguma opção {a} de {c} no bairro {b}, {city}?",
                  "Estou em {b}, {city} e quero {c} {a}. O que tem?",
                  "Quais restaurantes de {c} em {b}, {city} têm opção {a}?"],
    "unans_dish": ["Quanto custa {d} no {r}?", "O {r} vende {d}?"],
    "unans_info": ["Qual o telefone do {r}?", "O {r} aceita vale-refeição?",
                   "Qual o horário de funcionamento do {r} aos domingos?", "O {r} tem estacionamento próprio?"],
}
COLLOQ = {
    "price": ["{d} do {r} quanto tá?", "qnt é {d} no {r}", "me diz o valor de {d} lá no {r}", "preço {d} {r}"],
    "ingredients": ["{d} do {r} vem com o quê?", "o que tem em {d} do {r}", "ingredientes {d} {r}", "{d} {r} leva o que?"],
    "dietary": ["{r} tem algo {a}?", "{r}: opções {p}?", "tem {a} no {r}?"],
    "discovery": ["to em {b} ({city}) e quero {c} {a}, tem onde?", "{c} {a} em {b} {city}",
                  "onde acho {c} {a} no bairro {b}, {city}?"],
    "unans_dish": ["{d} no {r} custa quanto?", "tem {d} no {r}?"],
    "unans_info": ["telefone do {r}?", "{r} aceita vale refeição?", "{r} abre domingo que horas?",
                   "{r} tem estacionamento?"],
}
ANSWER_MODE = {"price": "exact", "ingredients": "exact", "dietary": "list", "discovery": "list",
               "unanswerable": "abstain"}


def _fmt(tpl_key, rng, **kw):
    """CLEAN[k] e COLLOQ[k] são o MESMO pedido (mesma ordem nas listas): sorteia um índice só."""
    assert len(CLEAN[tpl_key]) == len(COLLOQ[tpl_key])
    i = rng.randrange(len(CLEAN[tpl_key]))
    return CLEAN[tpl_key][i].format(**kw), COLLOQ[tpl_key][i].format(**kw)


def build_dataset(docs, n: int = 300, seed: int = 7):
    rng, noise_rng = random.Random(seed), random.Random(seed + 1)
    pool_names = {nm for _, pool in CUISINES.values() for (nm, *_r) in pool}
    rest_name = {d["restaurant_id"]: d["name"] for d in docs}

    # índice de discovery: (culinária, bairro, cidade, tag) -> pratos/restaurantes (sem bebidas)
    disc = defaultdict(lambda: {"dishes": [], "rests": set()})
    for d in docs:
        for it in d["items"]:
            if it["category"] == "Bebidas":
                continue
            for t in it["tags"]:
                if t in TAG_FORMS and not (d["cuisine"] == "Vegana" and t in ("vegano", "vegetariano")):
                    k = (d["cuisine"], d["neighborhood"], d["city"], t)
                    disc[k]["dishes"].append(it["dish_id"])
                    disc[k]["rests"].add(d["restaurant_id"])
    disc_keys = sorted(disc)

    weights = {"price": 0.25, "ingredients": 0.20, "dietary": 0.20, "discovery": 0.20, "unanswerable": 0.15}
    types = list(weights)
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 50:
        tries += 1
        t = rng.choices(types, weights=[weights[x] for x in types])[0]
        d = rng.choice(docs)
        if t in ("price", "ingredients"):
            it = rng.choice(d["items"])
            qc, qn = _fmt(t, rng, d=it["name"], r=d["name"])
            q = dict(answerable=True, gold_dish_ids=[it["dish_id"]], gold_restaurant_ids=[d["restaurant_id"]],
                     reference_answer=(f"R$ {money(it['price'])}" if t == "price" else it["description"]))
        elif t == "dietary":
            tag = rng.choice(list(TAG_FORMS))
            gold = [it for it in d["items"] if tag in it["tags"]]
            if not gold:
                continue
            m, f, p = TAG_FORMS[tag]
            qc, qn = _fmt(t, rng, r=d["name"], a=f, p=p)
            q = dict(answerable=True, gold_dish_ids=[g["dish_id"] for g in gold],
                     gold_restaurant_ids=[d["restaurant_id"]], reference_answer="; ".join(g["name"] for g in gold))
        elif t == "discovery":
            cu, nb, city, tag = rng.choice(disc_keys)
            g = disc[(cu, nb, city, tag)]
            phrase, gender = CUISINE_PHRASE[cu]
            qc, qn = _fmt(t, rng, a=TAG_FORMS[tag][0 if gender == "m" else 1], c=phrase, b=nb, city=city)
            names = sorted(rest_name[r] for r in g["rests"])
            q = dict(answerable=True, gold_dish_ids=sorted(g["dishes"]), gold_restaurant_ids=sorted(g["rests"]),
                     reference_answer="Restaurantes: " + "; ".join(names))
        else:
            have = {it["name"] for it in d["items"]}
            key = "unans_dish" if rng.random() < 0.5 else "unans_info"
            kw = dict(r=d["name"], d=rng.choice(sorted(pool_names - have)) if key == "unans_dish" else "")
            qc, qn = _fmt(key, rng, **kw)
            q = dict(answerable=False, gold_dish_ids=[], gold_restaurant_ids=[],
                     reference_answer="(abster-se: informação inexistente)")
        if qc in seen:
            continue
        seen.add(qc)
        q.update(question=qc, question_noisy=perturb(qn, noise_rng), type=t, answer_mode=ANSWER_MODE[t])
        out.append(q)

    rng.shuffle(out)
    for i, q in enumerate(out):
        q["qid"] = f"q{i:04d}"
        q["split"] = "dev" if i % 2 == 0 else "test"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/menus.jsonl")
    ap.add_argument("--out", default="data/eval_dataset.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    out = build_dataset(docs, a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for q in out:
        cnt[q["type"]] += 1
    print(f"{len(out)} perguntas -> {a.out} | {dict(cnt)}")


if __name__ == "__main__":
    main()
