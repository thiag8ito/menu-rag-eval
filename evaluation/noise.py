"""Ruído determinístico para simular o texto de usuários reais.

Aplicado sobre uma reformulação coloquial da pergunta (ver build_dataset). O gabarito
não muda: ruído altera a forma da pergunta, nunca o que ela pede.

Operadores (cada pergunta recebe uma combinação aleatória, com seed fixa):
  - remover acentos ("Açaí" -> "Acai")           - caixa baixa
  - abreviações de chat ("quanto" -> "qnt", "tá") - sem pontuação final
  - typos de teclado em 1-2 palavras com 5+ letras (troca, omissão, duplicação, vizinha de tecla)
"""
import random
import re
import unicodedata

ABBREV = {"quanto": "qnt", "você": "vc", "vocês": "vcs", "porque": "pq", "para": "pra", "está": "tá",
          "estou": "to", "também": "tb", "que": "q", "não": "n", "com": "c/"}
KEY_NEIGHBORS = {"a": "sqwz", "e": "wrd", "i": "uok", "o": "ipl", "u": "yij", "s": "adw", "r": "etf",
                 "t": "ryg", "n": "bmh", "m": "nj", "l": "ko", "c": "xv", "d": "sfe", "p": "ol"}


def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def typo(word: str, rng: random.Random) -> str:
    """Um erro de digitação em `word` (assume len >= 5; preserva a 1ª letra, como em digitação real)."""
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "dup", "neighbor"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    ch = word[i].lower()
    return word[:i] + rng.choice(KEY_NEIGHBORS.get(ch, ch)) + word[i + 1:]


def perturb(text: str, rng: random.Random) -> str:
    out = text
    if rng.random() < 0.6:
        out = strip_accents(out)
    if rng.random() < 0.5:
        out = out.lower()
    toks = re.findall(r"\w+|\W+", out)
    # abreviações (só afetam palavras de template; nomes de prato/restaurante quase nunca batem)
    for i, t in enumerate(toks):
        low = t.lower()
        if low in ABBREV and rng.random() < 0.5:
            toks[i] = ABBREV[low]
    # typos em 1-2 palavras "grandes" (inclui nomes de pratos e restaurantes, como em uso real)
    cand = [i for i, t in enumerate(toks) if t.isalpha() and len(t) >= 5]
    for i in rng.sample(cand, k=min(len(cand), rng.choice([1, 1, 2]))):
        toks[i] = typo(toks[i], rng)
    out = "".join(toks)
    if rng.random() < 0.5:
        out = out.rstrip("?.! ")
    return out
