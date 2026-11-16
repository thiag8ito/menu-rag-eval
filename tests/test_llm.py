"""Wrapper do LLM com SDK falso: o SDK real recusava `temperature` com TypeError (bug real, na máquina do usuário).

Os testes injetam um módulo `anthropic` falso em sys.modules; assim rodam com ou sem o SDK instalado.
"""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rag import llm


def fake_anthropic_module():
    m = ModuleType("anthropic")
    for name in ("BadRequestError", "RateLimitError", "APIConnectionError", "InternalServerError"):
        setattr(m, name, type(name, (Exception,), {}))
    return m


class FakeMessages:
    def __init__(self, reject_temperature=False):
        self.calls, self.reject = [], reject_temperature

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.reject and "temperature" in kwargs:
            raise TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])


def call_twice(messages):
    llm._TEMPERATURE_SUPPORTED = True
    with patch.dict(sys.modules, {"anthropic": fake_anthropic_module()}), \
         patch.object(llm, "_client", return_value=SimpleNamespace(messages=messages)):
        out = (llm.complete("m", "sys", "oi"), llm.complete("m", "sys", "de novo"))
    supported_after = llm._TEMPERATURE_SUPPORTED
    llm._TEMPERATURE_SUPPORTED = True
    return out, supported_after


def test_sends_temperature_when_supported():
    fake = FakeMessages()
    (a, b), supported = call_twice(fake)
    assert (a, b) == ("ok", "ok") and supported is True
    assert all(c["temperature"] == 0.0 for c in fake.calls)


def test_typeerror_on_temperature_falls_back_and_remembers():
    fake = FakeMessages(reject_temperature=True)
    (a, b), supported = call_twice(fake)
    assert (a, b) == ("ok", "ok") and supported is False
    assert "temperature" in fake.calls[0]                       # 1ª tentativa ainda tenta com temperature
    assert all("temperature" not in c for c in fake.calls[1:])  # retry e 2ª chamada já vão sem
    assert len(fake.calls) == 3


def test_unrelated_typeerror_is_not_swallowed():
    class Broken:
        def create(self, **kw):
            raise TypeError("algo completamente diferente")

    llm._TEMPERATURE_SUPPORTED = True
    try:
        call_twice(Broken())
    except TypeError as e:
        assert "diferente" in str(e)
    else:
        raise AssertionError("TypeError não relacionado a temperature deveria propagar")
    assert llm._TEMPERATURE_SUPPORTED is True


def test_rate_limit_is_retried_then_succeeds():
    mod = fake_anthropic_module()

    class Flaky:
        n = 0

        def create(self, **kw):
            Flaky.n += 1
            if Flaky.n == 1:
                raise mod.RateLimitError("429")
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="depois do retry")])

    llm._TEMPERATURE_SUPPORTED = True
    with patch.dict(sys.modules, {"anthropic": mod}), patch.object(llm.time, "sleep"), \
         patch.object(llm, "_client", return_value=SimpleNamespace(messages=Flaky())):
        assert llm.complete("m", "s", "u") == "depois do retry"
