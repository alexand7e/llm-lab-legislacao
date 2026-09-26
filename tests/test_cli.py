# -----------------------------------------------------------------------------
# File:     tests/test_cli.py
# Purpose:  Unit tests for the lab CLI (LLMClient replaced by a fake).
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lab.cli
from lab.cli import app
from lab.llm import BudgetExceededError, LLMConfigError
from lab.llm.client import ChatResult
from lab.llm.costs import Usage

REPO_MODELS = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

runner = CliRunner()


class FakeClient:
    """Substitui LLMClient: registra a chamada e devolve ``result`` ou levanta ``error``."""

    instances: list["FakeClient"] = []
    result = ChatResult(text="oi", usage=Usage(prompt_tokens=3, completion_tokens=1, cost_usd=0.5))
    error: Exception | None = None

    def __init__(self, config, *, cache=None, max_usd=0.0, env=None) -> None:
        self.cache = cache
        self.env = env
        self.max_usd = max_usd
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        FakeClient.instances.append(self)

    def chat(self, role, messages):
        self.calls.append((role, messages))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAB_MODELS_CONFIG", str(REPO_MODELS))
    monkeypatch.setattr(FakeClient, "instances", [])
    monkeypatch.setattr(lab.cli, "LLMClient", FakeClient)
    return FakeClient


def test_chat_prints_answer_tokens_and_cost(fake):
    result = runner.invoke(app, ["chat", "olá"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "oi"
    assert "[generator] in=3 out=1 cost=$0.500000" in result.stderr
    (client,) = fake.instances
    assert client.calls == [("generator", [{"role": "user", "content": "olá"}])]
    assert client.cache is not None


def test_chat_uses_role_and_reports_cache_hit(fake, monkeypatch):
    monkeypatch.setattr(fake, "result", fake.result.model_copy(update={"cached": True}))
    result = runner.invoke(app, ["chat", "--role", "judge", "olá"])
    assert result.exit_code == 0
    assert "[judge]" in result.stderr and "(cache)" in result.stderr
    assert fake.instances[0].calls[0][0] == "judge"


def test_no_cache_disables_cache(fake):
    assert runner.invoke(app, ["chat", "--no-cache", "olá"]).exit_code == 0
    assert fake.instances[0].cache is None


def test_max_usd_comes_from_settings(fake, monkeypatch):
    monkeypatch.setenv("LAB_MAX_USD_PER_RUN", "0.25")
    assert runner.invoke(app, ["chat", "olá"]).exit_code == 0
    assert fake.instances[0].max_usd == 0.25


@pytest.mark.parametrize(
    "error", [LLMConfigError("missing env var P_KEY"), BudgetExceededError("over budget")]
)
def test_known_errors_exit_1_on_stderr(fake, monkeypatch, error):
    monkeypatch.setattr(fake, "error", error)
    result = runner.invoke(app, ["chat", "olá"])
    assert result.exit_code == 1
    assert f"error: {error}" in result.stderr
    assert result.stdout == ""


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "chat" in result.output


def test_chat_passes_dotenv_keys_to_the_client(fake, monkeypatch):
    """Regressão: as chaves dos provedores vêm do .env, não só do ambiente."""
    monkeypatch.delenv("PRIMARY_API_KEY", raising=False)
    Path(".env").write_text("PRIMARY_API_KEY=abc\n", encoding="utf-8")
    result = runner.invoke(app, ["chat", "olá"])
    assert result.exit_code == 0, result.output
    assert fake.instances[0].env["PRIMARY_API_KEY"] == "abc"
