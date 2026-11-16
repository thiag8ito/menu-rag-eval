"""LLM-as-judge.

Fidelidade (faithfulness), no estilo RAGAS: o juiz decompõe a resposta em afirmações atômicas e
verifica cada uma contra o CONTEXTO recuperado. score = sustentadas / total. Mede alucinação,
independente de a resposta estar certa segundo o gabarito.

Correção: compara com o gabarito, com rubrica por modo de resposta:
  exact   (preço, ingredientes): precisa conter a informação do gabarito.
  list    (dietary, discovery): o contexto tem só top-k chunks, então não se exige completude;
          exige-se PRECISÃO (nada fora do gabarito) e ao menos 1 item correto. Completude é
          responsabilidade do recall@k do retrieval, medido separadamente.
  abstain : o correto é dizer que não encontrou, sem inventar.

Boas práticas: temperatura 0, saída JSON, juiz de modelo diferente/maior que o gerador (reduz viés de
autopreferência) e calibração contra rótulos humanos (evaluation/calibrate_judge.py).
"""
from rag.config import get_settings
from rag.llm import complete, parse_json

FAITH_SYSTEM = """Você é um avaliador rigoroso de sistemas RAG. Avalie apenas com base no CONTEXTO dado,
sem usar conhecimento externo. Responda SOMENTE com JSON válido."""

FAITH_USER = """CONTEXTO:
{context}

RESPOSTA A AVALIAR:
{answer}

Tarefa:
1. Liste as afirmações factuais atômicas da RESPOSTA (ignore recusas como "não encontrei", saudações e citações [n]).
2. Para cada afirmação, dê o veredito "supported" se o CONTEXTO a sustenta explicitamente (preços e nomes devem bater
   exatamente); caso contrário "unsupported".

Formato: {{"claims": [{{"claim": "...", "verdict": "supported|unsupported"}}]}}"""

CORRECT_SYSTEM = "Você é um avaliador imparcial. Responda SOMENTE com JSON válido."

CORRECT_RUBRIC = {
    "exact": """A resposta do sistema é correta? Ela deve conter a informação do gabarito e não contradizê-lo
(preços devem coincidir). Recusar-se a responder conta como incorreta.""",
    "list": """O gabarito é a lista COMPLETA de itens corretos, mas o sistema só viu parte do cardápio. Portanto:
considere correta se (a) TODOS os itens que a resposta cita estão no gabarito (sem itens fora dele) e (b) cita ao
menos um item do gabarito. Não exija a lista completa. Recusar-se a responder conta como incorreta.""",
}

CORRECT_ANSWERABLE = """PERGUNTA: {question}
GABARITO: {reference}
RESPOSTA DO SISTEMA: {answer}

{rubric}
Formato: {{"correct": true|false, "reason": "uma frase"}}"""

CORRECT_UNANSWERABLE = """PERGUNTA: {question}
Esta pergunta NÃO pode ser respondida pelos cardápios disponíveis. O comportamento correto é se abster/dizer que
não encontrou. RESPOSTA DO SISTEMA: {answer}

O sistema se absteve corretamente (sem inventar a informação)?
Formato: {{"correct": true|false, "reason": "uma frase"}}"""


def judge_faithfulness(answer: str, context: str, model: str | None = None) -> dict:
    model = model or get_settings().judge_model
    raw = complete(model, FAITH_SYSTEM, FAITH_USER.format(context=context, answer=answer), max_tokens=900)
    claims = parse_json(raw).get("claims", [])
    if not claims:
        return {"score": None, "claims": []}  # nada afirmado (ex.: abstenção) -> fora da média
    ok = sum(1 for c in claims if c.get("verdict") == "supported")
    return {"score": ok / len(claims), "claims": claims}


def judge_correctness(question: str, reference: str, answer: str, answer_mode: str,
                      model: str | None = None) -> dict:
    model = model or get_settings().judge_model
    if answer_mode == "abstain":
        user = CORRECT_UNANSWERABLE.format(question=question, answer=answer)
    else:
        user = CORRECT_ANSWERABLE.format(question=question, reference=reference, answer=answer,
                                         rubric=CORRECT_RUBRIC[answer_mode])
    d = parse_json(complete(model, CORRECT_SYSTEM, user, max_tokens=300))
    return {"correct": bool(d.get("correct")), "reason": d.get("reason", "")}
