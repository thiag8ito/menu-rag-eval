"""Rerankers (segundo estágio). Cross-encoders leem (query, chunk) juntos:
mais precisos que bi-encoders, mas custam um forward por candidato."""

RERANKER_REGISTRY = {
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",   # multilíngue, leve
    "bge-reranker": "BAAI/bge-reranker-v2-m3",                # mais forte, mais pesado
}


class NoReranker:
    name = "none"

    def rerank(self, query, chunks, scores):
        return list(range(len(chunks))), list(scores)


class CrossEncoderReranker:
    def __init__(self, key: str):
        from sentence_transformers import CrossEncoder
        self.name = key
        self.model = CrossEncoder(RERANKER_REGISTRY.get(key, key), max_length=512)

    def rerank(self, query, chunks, scores):
        pairs = [(query, c.embed_text) for c in chunks]
        new = self.model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(chunks)), key=lambda i: -new[i])
        return order, [float(new[i]) for i in order]


def get_reranker(name: str):
    if not name or name == "none":
        return NoReranker()
    return CrossEncoderReranker(name)
