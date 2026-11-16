import argparse
import json

from rag.chunking import build_chunks
from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.index import VectorIndex


def load_docs(path="data/menus.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{s.data_dir}/menus.jsonl")
    ap.add_argument("--embedder", default=s.embedder)
    ap.add_argument("--strategy", default=s.chunk_strategy)
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--out", default=s.index_dir)
    a = ap.parse_args()

    docs = load_docs(a.data)
    chunks = build_chunks(docs, a.strategy, a.chunk_size, not a.no_header)
    emb = get_embedder(a.embedder)
    vecs = emb.encode_docs([c.embed_text for c in chunks])
    meta = {"embedder": a.embedder, "strategy": a.strategy, "chunk_size": a.chunk_size,
            "contextual_header": not a.no_header, "n_chunks": len(chunks)}
    VectorIndex(vecs, chunks).save(a.out, meta)
    if hasattr(emb, "save"):
        emb.save(a.out)  # embedders treinados no corpus (tfidf/lsa) precisam ser reabertos idênticos na API
    print(f"índice salvo em {a.out}: {meta}")


if __name__ == "__main__":
    main()
