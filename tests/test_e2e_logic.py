"""Lógica do e2e testada com LLM mockado: classificação de falhas e agregação, sem gastar API."""
from unittest.mock import patch

from evaluation import run_e2e
from evaluation.run_e2e import classify_failure, run_one, summarize
from rag.chunking import Chunk
from rag.generator import ABSTAIN, is_abstention
from rag.llm import parse_json
from rag.retriever import Hit

Q_ANS = {"qid": "q1", "type": "price", "answerable": True, "answer_mode": "exact",
         "gold_dish_ids": ["d1"], "reference_answer": "R$ 10,00"}
Q_UNANS = {"qid": "q2", "type": "unanswerable", "answerable": False, "answer_mode": "abstain",
           "gold_dish_ids": [], "reference_answer": "(abster-se)"}


def hit(dish_ids):
    c = Chunk("r0-c0", "r0", "Rest", "texto", "texto", dish_ids)
    return Hit(c, 1.0, 0)


def test_classify_failure_branches():
    assert classify_failure(Q_ANS, 1.0, False, True) is None
    assert classify_failure(Q_ANS, 0.0, False, False) == "retrieval_miss"
    assert classify_failure(Q_ANS, 1.0, True, False) == "false_abstention"
    assert classify_failure(Q_ANS, 1.0, False, False) == "generation_error"
    assert classify_failure(Q_UNANS, None, False, False) == "hallucination"


def run_with(q, hits, answer, correct, faith):
    with patch.object(run_e2e, "generate_answer", return_value=answer), \
         patch.object(run_e2e, "judge_correctness", return_value={"correct": correct, "reason": "r"}), \
         patch.object(run_e2e, "judge_faithfulness", return_value={"score": faith, "claims": []}):
        return run_one(q, q.get("question", "pergunta?"), hits)


def test_run_one_marks_retrieval_miss_when_gold_not_in_context():
    r = run_with(Q_ANS, [hit(["outro"])], "R$ 99", False, 0.0)
    assert r["failure"] == "retrieval_miss" and r["retrieval_recall"] == 0.0


def test_run_one_marks_generation_error_when_gold_in_context():
    r = run_with(Q_ANS, [hit(["d1"])], "R$ 99", False, 0.0)
    assert r["failure"] == "generation_error" and r["retrieval_recall"] == 1.0


def test_run_one_unanswerable_correct_abstention_and_hallucination():
    ok = run_with(Q_UNANS, [hit([])], ABSTAIN, True, None)
    assert ok["failure"] is None and ok["abstained"]
    bad = run_with(Q_UNANS, [hit([])], "O telefone é 1234", False, 0.0)
    assert bad["failure"] == "hallucination" and not bad["abstained"]


def test_summarize_aggregates_rates_and_judge_rule_agreement():
    rows = [run_with(Q_ANS, [hit(["d1"])], "R$ 10", True, 1.0),
            run_with(Q_ANS, [hit(["x"])], "R$ 1", False, 0.5),
            run_with(Q_UNANS, [hit([])], ABSTAIN, True, None),
            run_with(Q_UNANS, [hit([])], "inventei", False, 0.0)]
    s = summarize(rows)
    assert s["n"] == 4 and s["n_answerable"] == 2
    assert s["correctness_answerable"] == 0.5
    assert s["faithfulness_mean"] == (1.0 + 0.5 + 0.0) / 3  # None (abstenção) fica fora da média
    assert s["judge_vs_rule_agreement"] == 1.0
    assert s["failure_breakdown"] == {"retrieval_miss": 1, "hallucination": 1}


def test_is_abstention_and_parse_json():
    assert is_abstention(ABSTAIN) and is_abstention(ABSTAIN.upper())
    assert not is_abstention("A pizza custa R$ 40,00 [1]")
    assert parse_json('```json\n{"correct": true}\n```') == {"correct": True}
    assert parse_json('Aqui está: {"claims": []} ok') == {"claims": []}
