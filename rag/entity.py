"""Entity linking de restaurante: detecta qual restaurante a pergunta cita.

Motivação (análise de erros em price/ingredients): ~37% dos top-1 erravam por "restaurante errado, MESMO prato".
O mesmo prato existe em muitos cardápios, então a única pista é o nome do restaurante, que se dilui num chunk
de ~1000 caracteres.

Método, em dois estágios:
  1. Ranking: containment de trigramas de caracteres. Para cada restaurante,
         score = |trigramas(nome) ∩ trigramas(pergunta)| / |trigramas(nome)|
     Tolerante a typos e a acentos/caixa. Candidato = o melhor, se score >= min_containment e vencer o 2º
     por >= margin.
  2. Gate: TODA palavra significativa (4+ letras) do nome precisa casar com algum token da pergunta a <= 1 edição
     (<= 2 para palavras com 8+ letras; transposição conta como 1). Sem o gate, trigramas coincidentes
     completavam o nome: "Tem alguma opção ... esperança ... bairro ... comida brasileira" chegava a 0,71 em
     "Tempero Brasil" (" te"/"tem" de "Tem", "per" de esperança, "ro " de bairro, "brasil" de brasileira).
     Bug achado por tests/test_entity.py; um gate por trigramas também falhava (0,57), por isso edição.
Perguntas sem restaurante (discovery) ou ambíguas retornam None e seguem o retrieval normal.
Limiares (0,7 / 0,10) escolhidos varrendo SÓ o split dev.
"""
import re
import unicodedata
from typing import Dict, Optional


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "


def _grams(s: str, n: int = 3) -> set:
    s = _norm(s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Distância de Damerau-Levenshtein restrita (OSA); retorna limit+1 se passar do limite."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


class RestaurantLinker:
    def __init__(self, names_by_id: Dict[str, str], min_containment: float = 0.7, margin: float = 0.10,
                 min_word_len: int = 4):
        self.ids = list(names_by_id)
        self.grams = [_grams(names_by_id[i]) for i in self.ids]
        self.words = [[w for w in _norm(names_by_id[i]).split() if len(w) >= min_word_len] for i in self.ids]
        self.min_containment, self.margin = min_containment, margin

    @staticmethod
    def _word_in_query(word: str, tokens) -> bool:
        limit = 2 if len(word) >= 8 else 1
        return any(_edit_distance(word, t, limit) <= limit for t in tokens)

    def link(self, query: str) -> Optional[str]:
        q = _grams(query)
        scored = sorted(((len(g & q) / len(g), k) for k, g in enumerate(self.grams)), reverse=True)
        (s1, k), s2 = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)
        if s1 < self.min_containment or (s1 - s2) < self.margin:
            return None
        tokens = _norm(query).split()
        if not all(self._word_in_query(w, tokens) for w in self.words[k]):
            return None  # alguma palavra significativa do nome não aparece: provável falso positivo
        return self.ids[k]
