"""Estratégias de chunking.

Cada chunk guarda `dish_ids`: os pratos cuja linha COMPLETA está contida no
texto do chunk. É isso que permite medir relevância no nível de evidência
(o chunk contém a resposta?) de forma comparável entre tamanhos de chunk.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    chunk_id: str
    restaurant_id: str
    restaurant_name: str
    text: str                 # texto bruto (usado para exibir e para checar relevância)
    embed_text: str           # texto que vai para o embedder (pode ter header contextual)
    dish_ids: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def doc_header(doc: dict) -> str:
    return f"{doc['name']} — {doc['cuisine']} — {doc['neighborhood']}, {doc['city']}"


def _dish_ids_in(doc: dict, text: str) -> list:
    return [it["dish_id"] for it in doc["items"] if it["line"] in text]


def _split_fixed(text: str, size: int, overlap: int):
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            yield piece
        if start + size >= len(text):
            break


def _split_structured(text: str, size: int, overlap_lines: int = 1):
    """Empacota linhas inteiras (pratos/títulos) até `size` caracteres.
    Nunca corta um prato no meio; repete `overlap_lines` linhas entre chunks."""
    lines = text.split("\n")
    cur: list = []
    cur_len = 0
    for ln in lines:
        add = len(ln) + 1
        if cur and cur_len + add > size:
            yield "\n".join(cur)
            cur = cur[-overlap_lines:] if overlap_lines else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        cur_len += add
    if cur:
        yield "\n".join(cur)


def build_chunks(docs, strategy: str = "structured", size: int = 512,
                 contextual_header: bool = True) -> list:
    chunks = []
    for doc in docs:
        if strategy == "fixed":
            pieces = list(_split_fixed(doc["text"], size, overlap=int(size * 0.15)))
        elif strategy == "structured":
            pieces = list(_split_structured(doc["text"], size))
        else:
            raise ValueError(f"estratégia desconhecida: {strategy}")
        header = doc_header(doc)
        for i, p in enumerate(pieces):
            embed_text = f"{header}\n{p}" if contextual_header else p
            chunks.append(Chunk(
                chunk_id=f"{doc['restaurant_id']}-c{i:02d}",
                restaurant_id=doc["restaurant_id"],
                restaurant_name=doc["name"],
                text=p, embed_text=embed_text,
                dish_ids=_dish_ids_in(doc, p),
            ))
    return chunks
