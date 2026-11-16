"""Avaliação ponta a ponta: retrieval -> geração -> LLM-as-judge.

Para cada pergunta: recupera top-k, gera a resposta, julga fidelidade e correção, e classifica falhas:
  retrieval_miss     nenhum chunk com o gabarito chegou ao LLM
  false_abstention   havia evidência no contexto, mas o sistema se absteve
  generation_error   havia evidência, o sistema respondeu e errou
  hallucination      pergunta sem resposta, mas o sistema respondeu algo
Requer ANTHROPIC_API_KEY. Custo ~ 3 chamadas de LLM por pergunta (gerar + 2 do juiz).
"""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.judge import judge_correctness, judge_faithfulness
from evaluation.metrics import bootstrap_ci, mean, recall_at_k
from rag.bm25 import BM25
from rag.build_index import load_docs
from rag.chunking import build_chunks
from rag.embeddings import get_embedder
from rag.entity import RestaurantLinker
from rag.generator import format_context, generate_answer, is_abstention
from rag.index import VectorIndex
from rag.reranker import get_reranker
from rag.retriever import Retriever


def classify_failure(q, retrieval_recall, abstained, correct):
    if correct:
        return None
    if not q["answerable"]:
        return "hallucination"
    if retrieval_recall == 0:
        return "retrieval_miss"
    return "false_abstention" if abstained else "generation_error"


def run_one(q, question_text, hits):
    answer = generate_answer(question_text, hits)
    ctx_ids = [h.chunk.dish_ids for h in hits]
    gold = set(q["gold_dish_ids"])
    rec = recall_at_k(ctx_ids, gold, len(hits)) if q["answerable"] else None
    abstained = is_abstention(answer)
    corr = judge_correctness(question_text, q["reference_answer"], answer, q["answer_mode"])
    faith = judge_faithfulness(answer, format_context(hits))
    return {"qid": q["qid"], "type": q["type"], "question": question_text, "answerable": q["answerable"],
            "answer_mode": q["answer_mode"], "reference": q["reference_answer"], "answer": answer,
            "abstained": abstained, "retrieval_recall": rec, "correct": corr["correct"],
            "correct_reason": corr["reason"], "faithfulness": faith["score"], "claims": faith["claims"],
            "failure": classify_failure(q, rec, abstained, corr["correct"]),
            "context_chunk_ids": [h.chunk.chunk_id for h in hits]}


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    fails = defaultdict(int)
    for r in rows:
        if r["failure"]:
            fails[r["failure"]] += 1
    by_type = {t: mean(r["correct"] for r in rows if r["type"] == t) for t in sorted({r["type"] for r in rows})}
    s = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
         "correctness_answerable": mean(r["correct"] for r in ans) if ans else None,
         "correctness_ci95": bootstrap_ci([float(r["correct"]) for r in ans]) if ans else None,
         "correctness_by_type": by_type,
         "faithfulness_mean": mean(faith) if faith else None,
         "faithfulness_ci95": bootstrap_ci(faith) if faith else None,
         "false_abstention_rate": mean(r["abstained"] for r in ans) if ans else None,
         "failure_breakdown": dict(fails)}
    if unans:
        s["abstention_accuracy_unanswerable_judge"] = mean(r["correct"] for r in unans)
        s["abstention_rate_unanswerable_rule"] = mean(r["abstained"] for r in unans)
        # concordância juiz x regra determinística (sanity check do juiz)
        s["judge_vs_rule_agreement"] = mean(r["correct"] == r["abstained"] for r in unans)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/menus.jsonl")
    ap.add_argument("--dataset", default="data/eval_dataset.jsonl")
    ap.add_argument("--split", default="test", choices=["all", "dev", "test"])
    ap.add_argument("--variant", default="clean", choices=["clean", "noisy"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedder", default="tfidf-char")
    ap.add_argument("--strategy", default="structured")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--entity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reranker", default="none")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/e2e_results.json")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(a.dataset, encoding="utf-8")]
    qs = [q for q in qs if a.split == "all" or q["split"] == a.split][: a.n]
    texts = [q["question_noisy"] if a.variant == "noisy" else q["question"] for q in qs]
    chunks = build_chunks(load_docs(a.data), a.strategy, a.chunk_size, not a.no_header)
    emb = get_embedder(a.embedder)
    index = VectorIndex(emb.encode_docs([c.embed_text for c in chunks]), chunks)
    bm25 = BM25([c.embed_text for c in chunks]) if a.hybrid else None
    linker = RestaurantLinker({c.restaurant_id: c.restaurant_name for c in chunks}) if a.entity else None
    all_hits = Retriever(index, emb, get_reranker(a.reranker), 20, bm25, linker).search_batch(texts, k=a.k)

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda t: run_one(*t), zip(qs, texts, all_hits)))

    summary = {"config": vars(a), **summarize(rows)}
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
