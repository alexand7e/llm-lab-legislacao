# -----------------------------------------------------------------------------
# File:     tests/test_llm_embeddings.py
# Purpose:  Tests for batched, cached embeddings in the LLM client (no network).
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import httpx2
import pytest
from openai import DefaultHttpxClient

from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.cache import SQLiteCache
from lab.settings import ModelsConfig, Provider, Role


class FakeEmbedder:
    """Endpoint de embeddings falso: vetor = [len(texto), posição], em ordem invertida."""

    def __init__(self, dims: int = 2, status: list[int] | None = None) -> None:
        self.dims = dims
        self.status = list(status or [])
        self.batches: list[list[str]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if self.status:
            code = self.status.pop(0)
            if code != 200:
                return httpx2.Response(code, json={"error": {"message": "falhou"}})
        inputs = json.loads(request.content)["input"]
        self.batches.append(inputs)
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": ([float(len(t)), float(i)] + [0.0] * (self.dims - 2)),
            }
            for i, t in enumerate(inputs)
        ]
        tokens = sum(len(t) for t in inputs)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "emb",
                "data": list(reversed(data)),  # fora de ordem: o cliente reordena por index
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )


def client_for(server: FakeEmbedder, dims: int | None = 2, **kw) -> LLMClient:
    config = ModelsConfig(
        roles={"embedding": Role(provider="p", model="emb", price_in=1.0, dims=dims)},
        providers={"p": Provider(base_url="https://example.test/v1", api_key_env="K")},
    )
    return LLMClient(
        config,
        env={"K": "secret"},
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(server)),
        sleep=lambda _s: None,
        **kw,
    )


TEXTS = ["a", "bb", "ccc", "dddd", "eeeee"]


def test_vectors_follow_input_order_across_batches():
    server = FakeEmbedder()
    result = client_for(server).embed("embedding", TEXTS, batch_size=2)
    assert [v[0] for v in result.vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(b) for b in server.batches] == [2, 2, 1]
    assert result.cached == 0


def test_cost_uses_input_price():
    result = client_for(FakeEmbedder()).embed("embedding", TEXTS)
    assert result.usage.prompt_tokens == 15
    assert result.usage.cost_usd == pytest.approx(15 / 1_000_000)


def test_cache_is_per_text_so_only_new_texts_are_sent(tmp_path: Path):
    server = FakeEmbedder()
    cache = SQLiteCache(tmp_path)
    client = client_for(server, cache=cache)
    first = client.embed("embedding", TEXTS[:3])
    second = client.embed("embedding", TEXTS)
    assert server.batches == [["a", "bb", "ccc"], ["dddd", "eeeee"]]
    assert second.cached == 3 and second.vectors[:3] == first.vectors
    assert second.usage.prompt_tokens == 9  # só os textos novos custam
    cache.close()


def test_empty_input_makes_no_call():
    server = FakeEmbedder()
    result = client_for(server).embed("embedding", [])
    assert result.vectors == [] and server.batches == []


def test_retries_transient_errors():
    server = FakeEmbedder(status=[503, 200])
    assert len(client_for(server).embed("embedding", TEXTS).vectors) == 5


def test_dimension_mismatch_is_a_config_error():
    with pytest.raises(LLMConfigError, match="dims=4"):
        client_for(FakeEmbedder(dims=2), dims=4).embed("embedding", TEXTS)


def test_dimension_is_not_checked_when_not_configured():
    assert len(client_for(FakeEmbedder(dims=3), dims=None).embed("embedding", TEXTS).vectors) == 5


def test_budget_applies_to_embeddings():
    with pytest.raises(BudgetExceededError):
        client_for(FakeEmbedder(), max_usd=0.000001).embed("embedding", TEXTS)
