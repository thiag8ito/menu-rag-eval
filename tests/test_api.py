from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from rag import api  # noqa: E402
from rag.bm25 import BM25  # noqa: E402
from rag.build_index import load_docs  # noqa: E402
from rag.chunking import build_chunks  # noqa: E402
from rag.embeddings import get_embedder  # noqa: E402
from rag.entity import RestaurantLinker  # noqa: E402
from rag.generator import ABSTAIN  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.reranker import get_reranker  # noqa: E402
from rag.retriever import Retriever  # noqa: E402


@pytest.fixture(scope="module")
def client():
    chunks = build_chunks(load_docs("data/menus.jsonl"), "structured", 512, True)
    emb = get_embedder("tfidf")
    idx = VectorIndex(emb.encode_docs([c.embed_text for c in chunks]), chunks)
    bm25 = BM25([c.embed_text for c in chunks])
    linker = RestaurantLinker({c.restaurant_id: c.restaurant_name for c in chunks})
    api.app.state.retriever = Retriever(idx, emb, get_reranker("none"), 20, bm25, linker)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["hybrid"] is True and body["entity_link"] is True
    assert body["n_chunks"] > 0


def test_search_returns_k_hits_with_latency(client):
    r = client.post("/search", json={"query": "pizza vegana de cogumelos", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) == 3 and body["latency_ms"]["total_ms"] > 0


def test_validation(client):
    assert client.post("/search", json={"query": "x", "k": 3}).status_code == 422
    assert client.post("/search", json={"query": "pizza", "k": 99}).status_code == 422
    assert client.post("/search", json={"k": 3}).status_code == 422


def test_ask_with_mocked_llm_returns_sources_and_abstention_flag(client):
    with patch.object(api, "generate_answer", return_value=ABSTAIN):
        r = client.post("/ask", json={"query": "qual o telefone do Forno Express?", "k": 3})
    body = r.json()
    assert r.status_code == 200 and body["abstained"] is True and len(body["sources"]) == 3


def test_ask_degrades_to_503_when_llm_unavailable(client):
    with patch.object(api, "generate_answer", side_effect=RuntimeError("sem chave")):
        assert client.post("/ask", json={"query": "pizza margherita", "k": 2}).status_code == 503
