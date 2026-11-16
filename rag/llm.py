"""Wrapper fino sobre a API da Anthropic (usado pelo gerador e pelo juiz)."""
import json
import re
import time
from functools import lru_cache

# Algumas versões do SDK / alguns modelos não aceitam `temperature`. O SDK pode recusar no cliente
# (TypeError: unexpected keyword argument) ou a API pode recusar (BadRequestError). Nos dois casos
# descobrimos uma vez, lembramos, e seguimos sem o parâmetro (vale para todas as threads do e2e).
_TEMPERATURE_SUPPORTED = True


@lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def _is_temperature_error(exc: Exception) -> bool:
    return "temperature" in str(exc).lower()


def complete(model: str, system: str, user: str, max_tokens: int = 800,
             temperature: float = 0.0, retries: int = 3) -> str:
    global _TEMPERATURE_SUPPORTED
    import anthropic
    client = _client()
    last_exc = None
    attempt = 0
    while attempt < retries:
        kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if _TEMPERATURE_SUPPORTED:
            kwargs["temperature"] = temperature
        try:
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except (TypeError, anthropic.BadRequestError) as e:
            if _TEMPERATURE_SUPPORTED and _is_temperature_error(e):
                _TEMPERATURE_SUPPORTED = False  # refaz a chamada sem temperature, sem gastar tentativa
                continue
            raise
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_exc = e
            time.sleep(2 ** attempt)
            attempt += 1
    raise RuntimeError(f"falha ao chamar a API após {retries} tentativas") from last_exc


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)
