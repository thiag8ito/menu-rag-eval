from rag.config import get_settings
from rag.llm import complete

ABSTAIN = "Não encontrei essa informação nos cardápios."

SYSTEM = f"""Você é o assistente de busca de cardápios de restaurantes de delivery.
Responda SOMENTE com base nos trechos de cardápio fornecidos em CONTEXTO.
Regras:
- Seja direto e cite o restaurante e o prato; informe preços exatamente como no contexto.
- Cite as fontes no formato [1], [2] ao lado de cada afirmação.
- Se o contexto não contém a resposta, responda exatamente: "{ABSTAIN}"
- Nunca invente pratos, preços, ingredientes ou restaurantes."""


def format_context(hits) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.chunk.restaurant_name})\n{h.chunk.text}"
                       for i, h in enumerate(hits))


def generate_answer(question: str, hits, model: str | None = None) -> str:
    model = model or get_settings().generator_model
    user = f"CONTEXTO:\n{format_context(hits)}\n\nPERGUNTA: {question}"
    return complete(model, SYSTEM, user, max_tokens=500).strip()


def is_abstention(answer: str) -> bool:
    return ABSTAIN.lower() in answer.lower()
