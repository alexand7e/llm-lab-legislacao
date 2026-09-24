# -----------------------------------------------------------------------------
# File:     tests/test_llm_client.py
# Purpose:  Unit tests for cache, cost and client (HTTP mocked with httpx2.MockTransport).
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import httpx2
import pytest
from openai import BadRequestError, DefaultHttpxClient

from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.cache import SQLiteCache, make_key
from lab.llm.costs import estimate_cost
from lab.settings import ModelsConfig, Provider, Role

MESSAGES = [{"role": "user", "content": "olá"}]


def make_config(price_in: float = 1.0, price_out: float = 2.0) -> ModelsConfig:
    return ModelsConfig(
        roles={"generator": Role(provider="p", model="m", price_in=price_in, price_out=price_out)},
        providers={"p": Provider(base_url="https://example.test/v1", api_key_env="P_KEY")},
    )


def completion(text: str = "oi", prompt: int = 1000, out: int = 500) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out},
    }


def make_client(**kw) -> LLMClient:
    kw.setdefault("env", {"P_KEY": "secret"})
    kw.setdefault("sleep", lambda _s: None)
    return LLMClient(kw.pop("config", make_config()), **kw)


class FakeServer:
    """Replays queued responses through httpx2.MockTransport and counts calls."""

    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def http_client(self) -> DefaultHttpxClient:
        return DefaultHttpxClient(transport=httpx2.MockTransport(self))


def ok() -> httpx2.Response:
    return httpx2.Response(200, json=completion())


def client_for(server: FakeServer, **kw) -> LLMClient:
    return make_client(http_client=server.http_client(), **kw)


def test_estimate_cost_per_million_tokens():
    usage = estimate_cost(Role(provider="p", model="m", price_in=1.0, price_out=2.0), 1000, 500)
    assert usage.cost_usd == pytest.approx(0.002)


def test_cache_key_is_stable_and_sensitive_to_params():
    a = make_key("m", MESSAGES, {"temperature": 0})
    assert a == make_key("m", MESSAGES, {"temperature": 0})
    assert a != make_key("m", MESSAGES, {"temperature": 1})


def test_chat_returns_text_usage_and_cost():
    result = client_for(FakeServer(ok())).chat("generator", MESSAGES)
    assert result.text == "oi"
    assert result.usage.prompt_tokens == 1000
    assert result.usage.cost_usd == pytest.approx(0.002)


def test_retries_on_429_then_succeeds():
    server = FakeServer(httpx2.Response(429, json={"error": {"message": "slow"}}), ok())
    assert client_for(server).chat("generator", MESSAGES).text == "oi"
    assert server.calls == 2


def test_does_not_retry_on_400():
    server = FakeServer(httpx2.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(BadRequestError):
        client_for(server).chat("generator", MESSAGES)
    assert server.calls == 1


def test_second_identical_call_hits_cache(tmp_path):
    server = FakeServer(ok())
    cache = SQLiteCache(tmp_path)
    client = client_for(server, cache=cache)
    first = client.chat("generator", MESSAGES)
    second = client.chat("generator", MESSAGES)
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text
    assert server.calls == 1
    cache.close()


def test_budget_exceeded_aborts():
    with pytest.raises(BudgetExceededError):
        client_for(FakeServer(ok()), max_usd=0.001).chat("generator", MESSAGES)


def test_base_url_can_come_from_env():
    config = ModelsConfig(
        roles={"generator": Role(provider="p", model="m")},
        providers={"p": Provider(base_url_env="P_URL", api_key_env="P_KEY")},
    )
    server = FakeServer(ok())
    client = LLMClient(
        config,
        env={"P_KEY": "k", "P_URL": "https://example.test/v1"},
        http_client=server.http_client(),
        sleep=lambda _s: None,
    )
    assert client.chat("generator", MESSAGES).text == "oi"


def test_missing_base_url_env_is_a_config_error():
    config = ModelsConfig(
        roles={"generator": Role(provider="p", model="m")},
        providers={"p": Provider(base_url_env="P_URL", api_key_env="P_KEY")},
    )
    with pytest.raises(LLMConfigError, match="P_URL"):
        LLMClient(config, env={"P_KEY": "k"}).chat("generator", MESSAGES)


def test_missing_api_key_error_names_variable_only():
    with pytest.raises(LLMConfigError, match="P_KEY"):
        make_client(env={}).chat("generator", MESSAGES)


def test_unknown_role():
    with pytest.raises(LLMConfigError, match="unknown role"):
        make_client().chat("nope", MESSAGES)
