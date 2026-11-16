"""Grid de retrieval: embedder x chunking x tamanho x header x híbrido x reranker.

Protocolo (evita otimismo por seleção):
  1. Cada config é avaliada nas perguntas respondíveis, nas variantes `clean` e `noisy`, em dev e test.
  2. A config é ESCOLHIDA no dev (média do MRR@10 clean/noisy) DENTRO de um orçamento de contexto
     (--ctx-budget, em caracteres enviados ao LLM no top-5). Sem isso, o vencedor degenera para
     chunks gigantes (= recuperar o documento inteiro), que sobem o hit@k à custa de tokens.
  3. O número final é o da config escolhida no TEST, com IC95% por bootstrap e comparação pareada
     contra um baseline ingênuo (fixed-512, sem header, sem híbrido, sem reranker).

Saídas em results/: retrieval_results.json, retrieval_table.csv, selected_per_question.json, report.md
"""
import argparse
import itertools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from evaluation.metrics import (bootstrap_ci, hit_at_k, mean, ndcg_at_k, paired_bootstrap,
                                recall_at_k, reciprocal_rank)
from rag.bm25 import BM25
from rag.build_index import load_docs
from rag.chunking import build_chunks
from rag.embeddings import LOCAL_EMBEDDERS, get_embedder
from rag.entity import RestaurantLinker
from rag.index import VectorIndex
from rag.reranker import get_reranker
from rag.retriever import Retriever

KS = (1, 3, 5, 10)
VARIANTS = ("clean", "noisy")
SPLITS = ("dev", "test")


def load_questions(path):
    qs = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [q for q in qs if q["answerable"]]  # retrieval só faz sentido com gabarito


def qtext(q, variant):
    return q["question_noisy"] if variant == "noisy" else q["question"]


def evaluate(retriever: Retriever, qs, variant):
    kfetch = max(max(KS), retriever.candidates_k)
    all_hits = retriever.search_batch([qtext(q, variant) for q in qs], k=kfetch)
    rows = []
    for q, hits in zip(qs, all_hits):
        gold, gold_r = set(q["gold_dish_ids"]), set(q["gold_restaurant_ids"])
        rel = [bool(gold & set(h.chunk.dish_ids)) for h in hits]
        ids = [h.chunk.dish_ids for h in hits]
        r = {"qid": q["qid"], "type": q["type"], "split": q["split"], "rr": reciprocal_rank(rel, 10),
             "ndcg@10": ndcg_at_k(rel, 10), "hit@cand": float(any(rel)),
             "doc_hit@5": float(any(h.chunk.restaurant_id in gold_r for h in hits[:5])),
             "ctx_chars@5": float(sum(len(h.chunk.text) for h in hits[:5]))}
        for k in KS:
            r[f"hit@{k}"] = float(hit_at_k(rel, k))
            r[f"recall@{k}"] = recall_at_k(ids, gold, k)
        rows.append(r)
    return rows


def summarize(rows):
    s = {m: mean(r[m] for r in rows) for m in rows[0] if m not in ("qid", "type", "split")}
    s["mrr@10"] = s.pop("rr")
    s["n"] = len(rows)
    return s


def measure_latency(retriever, qs, n=30):
    texts = [q["question"] for q in qs[:n]]
    retriever.search(texts[0], 5)  # warm-up
    ts = sorted(retriever.search(t, 5)[1]["total_ms"] for t in texts)
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def select_score(res):
    """Critério de seleção (só dev): média do MRR@10 entre clean e noisy."""
    return mean(res["metrics"][v]["dev"]["mrr@10"] for v in VARIANTS)


def ctx_cost(res):
    """Custo de contexto no dev: caracteres do top-5 enviados ao LLM (média clean/noisy)."""
    return mean(res["metrics"][v]["dev"]["ctx_chars@5"] for v in VARIANTS)


def pareto_front(results):
    """Configs não dominadas em (menor contexto, maior score de dev)."""
    front, best = [], -1.0
    for r in sorted(results, key=lambda r: (ctx_cost(r), -select_score(r))):
        if select_score(r) > best:
            front.append(r)
            best = select_score(r)
    return front


def cfg_name(emb, strat, size, hdr, hyb, ent, rr):
    return f"{emb}|{strat}-{size}|hdr{hdr}|hyb{hyb}|ent{ent}|rr-{rr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/menus.jsonl")
    ap.add_argument("--dataset", default="data/eval_dataset.jsonl")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--strategies", nargs="+", default=["fixed", "structured"])
    ap.add_argument("--embedders", nargs="+", default=["tfidf", "tfidf-char", "lsa"])
    ap.add_argument("--rerankers", nargs="+", default=["none"])
    ap.add_argument("--headers", type=int, nargs="+", default=[0, 1], help="1=header contextual")
    ap.add_argument("--hybrids", type=int, nargs="+", default=[0, 1], help="1=BM25+denso (RRF)")
    ap.add_argument("--entities", type=int, nargs="+", default=[0, 1], help="1=entity linking do restaurante")
    ap.add_argument("--candidates-k", type=int, default=20)
    ap.add_argument("--ctx-budget", type=float, default=5000,
                    help="máx. de caracteres de contexto (top-5) para uma config ser elegível")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    docs, qs = load_docs(a.data), load_questions(a.dataset)
    by_split = {s: [q for q in qs if q["split"] == s] for s in SPLITS}
    print(f"{len(docs)} docs | respondíveis: dev={len(by_split['dev'])} test={len(by_split['test'])}")

    embedders, rerankers = {}, {}
    results, per_q = [], {}
    for strat, size, hdr in itertools.product(a.strategies, a.chunk_sizes, a.headers):
        chunks = build_chunks(docs, strat, size, bool(hdr))
        bm25 = BM25([c.embed_text for c in chunks])
        linker = RestaurantLinker({c.restaurant_id: c.restaurant_name for c in chunks})
        for emb_name in a.embedders:
            # modelos locais são re-treinados a cada chunking; os pré-treinados são carregados uma vez
            emb = get_embedder(emb_name) if emb_name in LOCAL_EMBEDDERS else \
                embedders.setdefault(emb_name, get_embedder(emb_name))
            t0 = time.perf_counter()
            index = VectorIndex(emb.encode_docs([c.embed_text for c in chunks]), chunks)
            index_s = time.perf_counter() - t0
            for hyb, ent, rr_name in itertools.product(a.hybrids, a.entities, a.rerankers):
                rr = rerankers.setdefault(rr_name, get_reranker(rr_name))
                retr = Retriever(index, emb, rr, a.candidates_k, bm25 if hyb else None, linker if ent else None)
                metrics, rows_by = {}, {}
                for v in VARIANTS:
                    rows = evaluate(retr, qs, v)
                    rows_by[v] = rows
                    metrics[v] = {s: summarize([r for r in rows if r["split"] == s]) for s in SPLITS}
                p50, p95 = measure_latency(retr, by_split["dev"])
                name = cfg_name(emb_name, strat, size, hdr, hyb, ent, rr_name)
                res = {"config": name, "embedder": emb_name, "strategy": strat, "chunk_size": size,
                       "header": hdr, "hybrid": hyb, "entity": ent, "reranker": rr_name, "n_chunks": len(chunks),
                       "index_time_s": index_s, "p50_ms": p50, "p95_ms": p95, "metrics": metrics}
                results.append(res)
                per_q[name] = rows_by
                print(f"{name:52s} dev MRR clean={metrics['clean']['dev']['mrr@10']:.3f} "
                      f"noisy={metrics['noisy']['dev']['mrr@10']:.3f}  p50={p50:.1f}ms")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    results.sort(key=lambda r: -select_score(r))
    eligible = [r for r in results if ctx_cost(r) <= a.ctx_budget] or results
    sel = eligible[0]
    (out / "retrieval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "selected_per_question.json").write_text(json.dumps({sel["config"]: per_q[sel["config"]]}, ensure_ascii=False))
    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "metrics"}
        for v, s in itertools.product(VARIANTS, SPLITS):
            for m, x in r["metrics"][v][s].items():
                row[f"{v}_{s}_{m}"] = x
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "retrieval_table.csv", index=False)
    write_report(out / "report.md", results, eligible, per_q, a)
    print(f"\nconfig escolhida no dev: {sel['config']}\nrelatório: {out / 'report.md'}")


def _rows(per_q, name, variant, split):
    return [r for r in per_q[name][variant] if r["split"] == split]


def write_report(path, results, eligible, per_q, a):
    sel, unconstrained = eligible[0], results[0]
    base_name = cfg_name(a.embedders[0], "fixed", 512, 0, 0, 0, "none")
    base = next((r for r in results if r["config"] == base_name), None)
    L = ["# Avaliação de retrieval\n",
         "Seleção no **dev** (média do MRR@10 clean/noisy); resultado final no **test**. "
         "`clean` = perguntas bem escritas; `noisy` = coloquial + typos + sem acento.\n",
         f"## 1. Top-10 configs elegíveis (dev, contexto ≤ {a.ctx_budget:.0f} chars)\n",
         "| config | chunks | clean hit@5 | clean MRR@10 | noisy hit@5 | noisy MRR@10 | ctx chars@5 | p50 ms | p95 ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    def row(r):
        c, n = r["metrics"]["clean"]["dev"], r["metrics"]["noisy"]["dev"]
        return (f"| {r['config']} | {r['n_chunks']} | {c['hit@5']:.3f} | {c['mrr@10']:.3f} | {n['hit@5']:.3f} | "
                f"{n['mrr@10']:.3f} | {ctx_cost(r):.0f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")
    L += [row(r) for r in eligible[:10]]
    hdr = L[-12:-10]  # reaproveita o cabeçalho da tabela
    L += [f"\n**Sem restrição de contexto**, o vencedor seria `{unconstrained['config']}` "
          f"({ctx_cost(unconstrained):.0f} chars): com chunks desse tamanho o top-5 já é quase o menu inteiro "
          "de 5 restaurantes. É recuperação de documento, não de passagem, e custa muito mais tokens.\n",
          "### Fronteira de Pareto (melhor score de dev para cada nível de contexto)\n", *hdr]
    L += [row(r) for r in pareto_front(results)]

    L += [f"\n## 2. Config escolhida no dev, avaliada no test: `{sel['config']}`\n",
          "| variante | n | hit@1 | hit@5 [IC95%] | MRR@10 [IC95%] | recall@5 | hit@cand (teto reranker) |",
          "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        rows = _rows(per_q, sel["config"], v, "test")
        s = sel["metrics"][v]["test"]
        h = bootstrap_ci([r["hit@5"] for r in rows])
        m = bootstrap_ci([r["rr"] for r in rows])
        L.append(f"| {v} | {s['n']} | {s['hit@1']:.3f} | {s['hit@5']:.3f} [{h[0]:.3f}, {h[1]:.3f}] | "
                 f"{s['mrr@10']:.3f} [{m[0]:.3f}, {m[1]:.3f}] | {s['recall@5']:.3f} | {s['hit@cand']:.3f} |")

    if base and base["config"] != sel["config"]:
        L += [f"\n## 3. Selecionada vs baseline ingênuo (`{base_name}`), test, bootstrap pareado\n",
              "| variante | Δ MRR@10 | IC95% | P(Δ≤0) |", "|---|---|---|---|"]
        for v in VARIANTS:
            d, (lo, hi), p = paired_bootstrap([r["rr"] for r in _rows(per_q, sel["config"], v, "test")],
                                              [r["rr"] for r in _rows(per_q, base_name, v, "test")])
            L.append(f"| {v} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")

    L += ["\n## 4. Efeito marginal de cada fator (média sobre as demais configs, dev, MRR@10)\n",
          "| fator | nível | clean | noisy | n configs |", "|---|---|---|---|---|"]
    for f in ("embedder", "strategy", "chunk_size", "header", "hybrid", "entity", "reranker"):
        groups = defaultdict(list)
        for r in results:
            groups[r[f]].append(r)
        for lvl, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            L.append(f"| {f} | {lvl} | {mean(x['metrics']['clean']['dev']['mrr@10'] for x in rs):.3f} | "
                     f"{mean(x['metrics']['noisy']['dev']['mrr@10'] for x in rs):.3f} | {len(rs)} |")

    L += ["\n## 5. Por tipo de pergunta (config escolhida, test)\n",
          "| tipo | n | clean hit@5 | clean MRR | noisy hit@5 | noisy MRR |", "|---|---|---|---|---|---|"]
    types = sorted({r["type"] for r in _rows(per_q, sel["config"], "clean", "test")})
    for t in types:
        c = [r for r in _rows(per_q, sel["config"], "clean", "test") if r["type"] == t]
        n = [r for r in _rows(per_q, sel["config"], "noisy", "test") if r["type"] == t]
        L.append(f"| {t} | {len(c)} | {mean(r['hit@5'] for r in c):.3f} | {mean(r['rr'] for r in c):.3f} | "
                 f"{mean(r['hit@5'] for r in n):.3f} | {mean(r['rr'] for r in n):.3f} |")

    L += ["\n## Como ler\n",
          "- `hit@cand`: algum candidato relevante entre os top-N do 1º estágio = teto do reranker.",
          "- `ctx chars@5`: caracteres enviados ao LLM. Chunks maiores sobem hit@k quase de graça "
          "(o menu inteiro cabe num chunk); leia hit@k junto com este custo.",
          f"- {len(results)} configs avaliadas e a melhor foi escolhida no dev: o dev é otimista por construção; "
          "o test é o número que vale."]
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
