"""Wrapper fino sobre a API da Anthropic (usado pelo gerador e pelo juiz)."""
import json
import re
import time
from functools import lru_cache


@lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def complete(model: str, system: str, user: str, max_tokens: int = 800,
             temperature: float = 0.0, retries: int = 3) -> str:
    import anthropic
    client = _client()
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    use_temp = True
    for attempt in range(retries):
        try:
            if use_temp:
                kwargs["temperature"] = temperature
            else:
                kwargs.pop("temperature", None)
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")
        except anthropic.BadRequestError as e:
            if "temperature" in str(e).lower() and use_temp:
                use_temp = False        # alguns modelos não aceitam temperature
                continue
            raise
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt)
    raise RuntimeError("falha ao chamar a API após retries")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)
