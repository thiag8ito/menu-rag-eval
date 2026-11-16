"""Calibra o juiz contra rótulos humanos (rotulagem CEGA: você não vê o veredito do juiz).

  python -m evaluation.calibrate_judge export --n 40   # gera results/judge_labeling.csv (sem o veredito do juiz)
  (preencha human_correct com 1/0 seguindo a mesma rubrica do juiz; ver evaluation/judge.py)
  python -m evaluation.calibrate_judge score           # acurácia, Cohen's kappa e matriz de confusão

Amostragem estratificada por tipo, para o kappa não ser dominado do tipo mais frequente.
Observação: calibra a CORREÇÃO; a fidelidade (claims) pode ser auditada lendo `claims` no JSON do e2e.
"""
import argparse
import csv
import json
import random
from collections import defaultdict

from evaluation.metrics import cohen_kappa

LABEL_CSV = "results/judge_labeling.csv"


def export(n, src):
    rows = json.load(open(src, encoding="utf-8"))["rows"]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    rng, sample = random.Random(0), []
    per = max(1, n // len(by_type))
    for t in sorted(by_type):
        rng.shuffle(by_type[t])
        sample += by_type[t][:per]
    rng.shuffle(sample)
    with open(LABEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "question", "reference", "answer", "human_correct"])
        for r in sample:
            w.writerow([r["qid"], r["type"], r["question"], r["reference"], r["answer"], ""])
    print(f"{len(sample)} linhas -> {LABEL_CSV}. Rotule human_correct (1/0) SEM olhar o e2e_results.json.")


def score(src):
    judge = {r["qid"]: bool(r["correct"]) for r in json.load(open(src, encoding="utf-8"))["rows"]}
    rows = [r for r in csv.DictReader(open(LABEL_CSV, encoding="utf-8")) if r["human_correct"].strip()]
    j = [judge[r["qid"]] for r in rows]
    h = [bool(int(r["human_correct"])) for r in rows]
    acc = sum(x == y for x, y in zip(j, h)) / len(rows)
    tp, tn = sum(x and y for x, y in zip(j, h)), sum((not x) and (not y) for x, y in zip(j, h))
    fp, fn = sum(x and (not y) for x, y in zip(j, h)), sum((not x) and y for x, y in zip(j, h))
    print(f"n={len(rows)} | concordância={acc:.3f} | Cohen's kappa={cohen_kappa(j, h):.3f}")
    print(f"juiz=correto/humano=correto: {tp} | juiz=errado/humano=errado: {tn} | "
          f"juiz mais permissivo (FP): {fp} | juiz mais rígido (FN): {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--src", default="results/e2e_results.json")
    a = ap.parse_args()
    export(a.n, a.src) if a.cmd == "export" else score(a.src)
