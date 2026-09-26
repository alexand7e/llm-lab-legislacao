# -----------------------------------------------------------------------------
# File:     tests/test_strategies.py
# Purpose:  Tests for the Strategy interface, citation handling and the baseline.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import httpx2
import pytest
from openai import DefaultHttpxClient
from typer.testing import CliRunner

import lab.cli
from lab.cli import app
from lab.ingest.sources import LawSource
from lab.llm import LLMClient, Parsed
from lab.llm.client import ChatResult
from lab.llm.costs import Usage
from lab.settings import ModelsConfig, Provider, Role
from lab.strategies import BaselineStrategy, load_prompt, normalize_citations
from lab.strategies.baseline import BaselineOutput

ROOT = Path(__file__).parent.parent
LAWS = [
    LawSource(id="lgpd", name="Lei Geral de Proteção de Dados", number="Lei 13.709/2018", url="u"),
    LawSource(id="cdc", name="Código de Defesa do Consumidor", number="Lei 8.078/1990", url="u"),
]
runner = CliRunner()


# -- citações -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["lgpd:art:7"], ["lgpd:art:7"]),
        (["LGPD:art:7º", " cdc : art : 6 "], ["lgpd:art:7", "cdc:art:6"]),
        (["lgpd:art:55-a", "lgpd:art:55-A"], ["lgpd:art:55-A"]),
        (["lgpd:art:007"], ["lgpd:art:7"]),
        (["lgpd:art:7", "lgpd:art:7"], ["lgpd:art:7"]),
        (["marco_civil:art:1"], []),  # norma fora da lista conhecida
        (["art. 7 da LGPD", "lgpd:art:", "lgpd:7"], []),  # formato inválido
        ([], []),
    ],
)
def test_normalize_citations(raw: list[str], expected: list[str]):
    assert normalize_citations(raw, {"lgpd", "cdc"}) == expected


def test_load_prompt_reads_versioned_file():
    text = load_prompt("baseline_v1", ROOT / "prompts")
    assert "{laws}" in text and "cited_articles" in text


def test_load_prompt_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prompt("nao_existe", ROOT / "prompts")


# -- baseline -----------------------------------------------------------------


class FakeServer:
    """Devolve ``content`` como resposta do modelo e guarda as requisições."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": self.content},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )


def strategy_for(server: FakeServer) -> BaselineStrategy:
    config = ModelsConfig(
        roles={"generator": Role(provider="p", model="m", price_in=1.0, price_out=2.0)},
        providers={"p": Provider(base_url="https://example.test/v1", api_key_env="K")},
    )
    client = LLMClient(
        config,
        env={"K": "secret"},
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(server)),
        sleep=lambda _s: None,
    )
    return BaselineStrategy(client, LAWS, prompts_dir=ROOT / "prompts")


def test_baseline_returns_structured_answer():
    server = FakeServer(
        json.dumps(
            {
                "answer": "  Imediatamente.  ",
                "cited_articles": ["LGPD:art:19", "lgpd:art:19", "xyz:art:1"],
                "confidence": 0.7,
            }
        )
    )
    answer = strategy_for(server).answer("Qual o prazo?")
    assert answer.text == "Imediatamente."
    assert answer.cited_articles == ["lgpd:art:19"]
    assert answer.retrieved_ids == []  # baseline não recupera nada
    assert answer.confidence == 0.7
    assert answer.usage.prompt_tokens == 100
    assert answer.usage.cost_usd == pytest.approx(0.0002)
    assert answer.parse_error is None


def test_baseline_sends_system_prompt_with_laws_and_question():
    server = FakeServer(json.dumps({"answer": "ok"}))
    strategy_for(server).answer("Qual o prazo?")
    body = server.requests[0]
    system, user = body["messages"]
    assert system["role"] == "system" and user == {"role": "user", "content": "Qual o prazo?"}
    assert "Lei 13.709/2018 (lgpd)" in system["content"]
    assert "Código de Defesa do Consumidor" in system["content"]
    assert "{laws}" not in system["content"]
    assert body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"


def test_answer_outside_the_schema_keeps_text_and_cost():
    server = FakeServer("Não sei responder a isso.")
    answer = strategy_for(server).answer("Pergunta?")
    assert answer.text == "Não sei responder a isso."
    assert answer.cited_articles == []
    assert answer.parse_error
    assert answer.usage.prompt_tokens == 100  # o custo da resposta ruim não se perde


def test_baseline_satisfies_the_strategy_interface():
    strategy = strategy_for(FakeServer(json.dumps({"answer": "ok"})))
    assert strategy.name == "baseline"


# -- comando ------------------------------------------------------------------


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, config, *, cache=None, max_usd=0.0, env=None) -> None:
        self.calls: list[tuple[str, list[dict[str, str]], dict]] = []
        FakeClient.instances.append(self)

    def chat_parsed(self, role, messages, schema, **params):
        self.calls.append((role, messages, params))
        value = BaselineOutput(answer="Em 15 dias.", cited_articles=["lgpd:art:17"], confidence=0.9)
        result = ChatResult(
            text="{}",
            usage=Usage(prompt_tokens=7, completion_tokens=3, cost_usd=0.25),
            latency_ms=42,
        )
        return Parsed(value=value, result=result)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAB_MODELS_CONFIG", str(ROOT / "config" / "models.yaml"))
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(ROOT / "config" / "corpus.yaml"))
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "baseline_v1.md").write_text("Normas:\n{laws}\n", encoding="utf-8")
    monkeypatch.setattr(FakeClient, "instances", [])
    monkeypatch.setattr(lab.cli, "LLMClient", FakeClient)
    return FakeClient


def test_ask_prints_answer_and_details(cli_env):
    result = runner.invoke(app, ["ask", "Qual o prazo?"])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "Em 15 dias."
    assert "lgpd:art:17" in result.stderr
    assert "confiança=0.90" in result.stderr
    assert "in=7 out=3 cost=$0.250000 (42 ms)" in result.stderr
    role, messages, _ = cli_env.instances[0].calls[0]
    assert role == "generator" and messages[1]["content"] == "Qual o prazo?"


def test_ask_missing_prompt_is_a_clean_error(cli_env):
    result = runner.invoke(app, ["ask", "?", "--prompt", "nao_existe"])
    assert result.exit_code == 1
    assert "arquivo não encontrado" in result.output
