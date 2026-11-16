"""Backends de embedding com interface única (fit / encode_docs / encode_queries).

Offline (só scikit-learn):
  tfidf       palavras 1-2gram. Baseline lexical.
  tfidf-char  n-gramas de caracteres (3-5, dentro de palavras). Robusto a typos.
  lsa         TF-IDF + SVD (semântica latente). "Denso" sem rede.
Com sentence-transformers (baixa modelos do HuggingFace):
  minilm-multi, e5-small, bge-m3   (e5 exige prefixos "query: " / "passage: ").

Os vetores são L2-normalizados => produto interno = cosseno. Os backends
esparsos devolvem scipy.sparse (CSR); os densos, numpy.
"""
import pickle
from pathlib import Path

import numpy as np

MODEL_REGISTRY = {
    "minilm-multi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5-small": "intfloat/multilingual-e5-small",
    "bge-m3": "BAAI/bge-m3",
}
LOCAL_EMBEDDERS = ("tfidf", "tfidf-char", "lsa")  # treinados no corpus => precisam ser persistidos


class TfidfEmbedder:
    def __init__(self, name="tfidf"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.name = name
        if name == "tfidf-char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                       strip_accents="unicode", lowercase=True)
        else:
            self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                       strip_accents="unicode", lowercase=True)

    def encode_docs(self, texts):
        return self.vec.fit_transform(texts)  # fit no corpus; TfidfVectorizer já normaliza (L2)

    def encode_queries(self, texts):
        return self.vec.transform(texts)

    def save(self, directory):
        with open(Path(directory) / "embedder.pkl", "wb") as f:
            pickle.dump(self, f)


class LsaEmbedder(TfidfEmbedder):
    def __init__(self, n_components=128):
        super().__init__("tfidf")
        from sklearn.decomposition import TruncatedSVD
        self.name = "lsa"
        self.n_components = n_components
        self.svd = None
        self._TruncatedSVD = TruncatedSVD

    @staticmethod
    def _norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return (m / np.maximum(n, 1e-9)).astype("float32")

    def encode_docs(self, texts):
        x = self.vec.fit_transform(texts)
        k = max(2, min(self.n_components, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = self._TruncatedSVD(k, random_state=0)
        return self._norm(self.svd.fit_transform(x))

    def encode_queries(self, texts):
        return self._norm(self.svd.transform(self.vec.transform(texts)))


class SentenceTransformerEmbedder:
    def __init__(self, key: str):
        from sentence_transformers import SentenceTransformer
        self.name = key
        self.model_id = MODEL_REGISTRY.get(key, key)
        self.model = SentenceTransformer(self.model_id)
        self.is_e5 = "e5" in self.model_id.lower()

    def _enc(self, texts, prefix=""):
        return self.model.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")

    def encode_docs(self, texts):
        return self._enc(texts, "passage: " if self.is_e5 else "")

    def encode_queries(self, texts):
        return self._enc(texts, "query: " if self.is_e5 else "")


def get_embedder(name: str):
    if name in ("tfidf", "tfidf-char"):
        return TfidfEmbedder(name)
    if name == "lsa":
        return LsaEmbedder()
    return SentenceTransformerEmbedder(name)


def load_embedder(name: str, directory: str):
    """Reabre o embedder de um índice salvo: os locais vêm do pickle (vocabulário/SVD treinados)."""
    if name in LOCAL_EMBEDDERS:
        with open(Path(directory) / "embedder.pkl", "rb") as f:
            return pickle.load(f)
    return get_embedder(name)
