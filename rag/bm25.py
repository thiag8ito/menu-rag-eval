"""BM25 (Okapi) sobre os chunks, para busca híbrida. Sem dependências além do scikit-learn."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.cv = CountVectorizer(strip_accents="unicode", lowercase=True)
        tf = self.cv.fit_transform(texts).tocsr().astype("float32")
        n = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.w = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        q = (self.cv.transform([query]) > 0).astype("float32")
        return np.asarray((self.w @ q.T).todense()).ravel()

    def search(self, query: str, k: int):
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]


def rrf(rankings, k: int, c: int = 60):
    """Reciprocal Rank Fusion: score(d) = sum_i 1/(c + rank_i(d)). Robusto a escalas de score diferentes."""
    fused = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (c + r + 1)
    order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
    return np.array(order), np.array([fused[i] for i in order])
