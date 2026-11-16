import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.bm25 import BM25
from rag.embeddings import load_embedder
from rag.entity import RestaurantLinker
from rag.generator import generate_answer, is_abstention
from rag.index import VectorIndex
from rag.reranker import get_reranker
from rag.retriever import Retriever


def load_retriever() -> Retriever:
    s = get_settings()
    index, meta = VectorIndex.load(s.index_dir)
    emb = load_embedder(meta["embedder"], s.index_dir)
    bm25 = BM25([c.embed_text for c in index.chunks]) if s.hybrid else None
    linker = RestaurantLinker({c.restaurant_id: c.restaurant_name for c in index.chunks}) if s.entity_link else None
    return Retriever(index, emb, get_reranker(s.reranker), s.candidates_k, bm25, linker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "retriever", None) is None:
        app.state.retriever = load_retriever()
    yield


log = logging.getLogger("menu_rag")
app = FastAPI(title="Menu RAG", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(5, ge=1, le=20)


class HitOut(BaseModel):
    chunk_id: str
    restaurant_id: str
    restaurant_name: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: list[HitOut]
    latency_ms: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    sources: list[HitOut]
    latency_ms: dict


def _hits_out(hits):
    return [HitOut(chunk_id=h.chunk.chunk_id, restaurant_id=h.chunk.restaurant_id,
                   restaurant_name=h.chunk.restaurant_name, score=h.score, text=h.chunk.text)
            for h in hits]


@app.get("/health")
def health(request: Request):
    r: Retriever = request.app.state.retriever
    return {"status": "ok", "n_chunks": len(r.index.chunks),
            "embedder": r.embedder.name, "reranker": r.reranker.name, "hybrid": r.hybrid,
            "entity_link": r.linker is not None}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request):
    hits, timings = request.app.state.retriever.search(req.query, req.k)
    return SearchResponse(query=req.query, hits=_hits_out(hits), latency_ms=timings)


@app.post("/ask", response_model=AskResponse)
def ask(req: SearchRequest, request: Request):
    hits, timings = request.app.state.retriever.search(req.query, req.k)
    t0 = time.perf_counter()
    try:
        answer = generate_answer(req.query, hits)
    except Exception as e:  # sem chave/rede: degrada de forma explícita (e registra a causa real)
        log.exception("falha na geração")
        raise HTTPException(status_code=503, detail=f"geração indisponível: {type(e).__name__}")
    timings["generate_ms"] = (time.perf_counter() - t0) * 1e3
    return AskResponse(question=req.query, answer=answer, abstained=is_abstention(answer),
                       sources=_hits_out(hits), latency_ms=timings)
