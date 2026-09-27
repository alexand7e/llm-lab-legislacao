# -----------------------------------------------------------------------------
# File:     tests/test_eval_runner.py
# Purpose:  Tests for the eval runner, run registry and the `lab eval` command.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import lab.cli
from lab.cli import app
from lab.eval.dataset import Question
from lab.eval.registry import (
    GitState,
    RunConfig,
    RunMeta,
    RunMetrics,
    compute_metrics,
    git_state,
    run_dir_name,
    write_run,
)
from lab.eval.runner import run_eval
from lab.llm import BudgetExceededError, LLMConfigError, Parsed
from lab.llm.cache import SQLiteCache
from lab.llm.client import ChatResult
from lab.llm.costs import Usage
from lab.strategies.base import Answer
from lab.strategies.baseline import BaselineOutput

ROOT = Path(__file__).parent.parent
runner = CliRunner()


def make_question(n: int, **over) -> Question:
    base = {
        "id": f"q{n:03d}",
        "question": f"Pergunta número {n} sobre a lei?",
        "reference_answer": "Resposta.",
        "gold_articles": ["lgpd:art:19"],
        "category": "factual",
        "laws": ["lgpd"],
    }
    return Question(**{**base, **over})


class FakeStrategy:
    """Estratégia falsa: responde citando o gabarito; comportamento ajustável por pergunta."""

    name = "fake"

    def __init__(self, delays: dict[str, float] | None = None, fail: dict | None = None) -> None:
        self.delays = delays or {}
        self.fail = fail or {}
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def answer(self, question: str) -> Answer:
        with self._lock:
            self.calls.append(question)
        time.sleep(self.delays.get(question, 0))
        if question in self.fail:
            raise self.fail[question]
        return Answer(
            text="Imediatamente.",
            cited_articles=["lgpd:art:19"],
            usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.5),
            latency_ms=100,
        )


# -- runner -------------------------------------------------------------------


def test_results_follow_question_order_even_if_finished_out_of_order():
    qs = [make_question(1), make_question(2), make_question(3)]
    strategy = FakeStrategy(delays={qs[0].question: 0.15, qs[1].question: 0.05})
    seen: list[str] = []
    results = run_eval(strategy, qs, workers=3, on_result=lambda r: seen.append(r.question_id))
    assert [r.question_id for r in results] == ["q001", "q002", "q003"]
    assert seen[0] == "q003" and sorted(seen) == ["q001", "q002", "q003"]  # progresso por conclusão


def test_result_carries_answer_and_scores():
    (result,) = run_eval(FakeStrategy(), [make_question(1)], workers=1)
    assert result.error is None and result.answer is not None and result.scores is not None
    assert result.scores.citation_hit == 1.0


def test_one_failing_question_does_not_stop_the_run():
    qs = [make_question(1), make_question(2), make_question(3)]
    strategy = FakeStrategy(fail={qs[1].question: RuntimeError("API fora do ar")})
    results = run_eval(strategy, qs, workers=2)
    assert [bool(r.error) for r in results] == [False, True, False]
    assert "RuntimeError: API fora do ar" in (results[1].error or "")
    assert results[1].answer is None and results[1].scores is None


@pytest.mark.parametrize("error", [BudgetExceededError("estourou"), LLMConfigError("sem chave")])
def test_budget_and_config_errors_abort_the_whole_run(error: Exception):
    qs = [make_question(n) for n in range(1, 30)]
    # As demais perguntas demoram um pouco: sem isso o worker esgota a fila antes de o
    # cancelamento da rodada acontecer, e o teste ficaria dependente de timing.
    slow = {q.question: 0.05 for q in qs[1:]}
    strategy = FakeStrategy(delays=slow, fail={qs[0].question: error})
    with pytest.raises(type(error)):
        run_eval(strategy, qs, workers=1)
    assert len(strategy.calls) < len(qs)  # o que não começou foi cancelado


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers"):
        run_eval(FakeStrategy(), [make_question(1)], workers=0)


def test_cache_supports_concurrent_writers(tmp_path: Path):
    cache = SQLiteCache(tmp_path)
    errors: list[Exception] = []

    def work(n: int) -> None:
        try:
            for i in range(20):
                cache.set(f"k{n}-{i}", {"v": i})
                assert cache.get(f"k{n}-{i}") == {"v": i}
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cache.close()
    assert errors == []


# -- registro -----------------------------------------------------------------

T0 = datetime(2026, 9, 26, 9, 47, 0)


def run_config() -> RunConfig:
    return RunConfig(
        strategy="fake",
        prompt="p_v1",
        roles={"generator": "m"},
        questions_file="q.jsonl",
        questions_sha256="sha256:abc",
        questions_count=2,
        workers=2,
        max_usd=2.0,
        use_cache=True,
    )


def write(tmp_path: Path, qs, results, git: GitState) -> Path:
    return write_run(
        tmp_path,
        config=run_config(),
        questions=qs,
        results=results,
        started=T0,
        finished=T0 + timedelta(seconds=12.5),
        git=git,
    )


def test_run_dir_name():
    assert run_dir_name(T0, "baseline") == "20260926-0947-baseline"


def test_write_run_creates_the_four_files(tmp_path: Path):
    qs = [make_question(1), make_question(2)]
    results = run_eval(FakeStrategy(), qs, workers=1)
    run_dir = write(tmp_path, qs, results, GitState(commit="abc123", dirty=False))
    assert run_dir.name == "20260926-0947-fake"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "answers.jsonl",
        "config.yaml",
        "meta.json",
        "metrics.json",
    ]
    assert yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))["roles"] == {
        "generator": "m"
    }
    lines = (run_dir / "answers.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["question_id"] for line in lines] == ["q001", "q002"]
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text("utf-8"))
    assert (meta.commit, meta.questions, meta.errors, meta.duration_s) == ("abc123", 2, 0, 12.5)
    assert meta.total_cost_usd == 1.0 and meta.official
    metrics = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics.overall.n == 2 and metrics.by_category["factual"].n == 2


@pytest.mark.parametrize(
    ("git", "question_over", "fail", "official"),
    [
        (GitState(commit="abc", dirty=False), {}, False, True),
        (GitState(commit="abc", dirty=True), {}, False, False),  # árvore suja
        (GitState(commit=None, dirty=True), {}, False, False),  # sem commit rastreável
        (GitState(commit="abc", dirty=False), {"origin": "dev_draft"}, False, False),
        (GitState(commit="abc", dirty=False), {}, True, False),  # houve erro
    ],
)
def test_official_only_for_clean_commit_real_set_and_no_errors(
    tmp_path: Path, git: GitState, question_over: dict, fail: bool, official: bool
):
    q = make_question(1, **question_over)
    strategy = FakeStrategy(fail={q.question: RuntimeError("x")} if fail else {})
    results = run_eval(strategy, [q], workers=1)
    run_dir = write(tmp_path, [q], results, git)
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text("utf-8"))
    assert meta.official is official


def test_two_runs_in_the_same_minute_do_not_overwrite(tmp_path: Path):
    qs = [make_question(1)]
    results = run_eval(FakeStrategy(), qs, workers=1)
    git = GitState(commit="abc", dirty=False)
    first, second = write(tmp_path, qs, results, git), write(tmp_path, qs, results, git)
    assert first != second and second.name == "20260926-0947-fake-2"


def test_metrics_skip_failed_questions():
    qs = [make_question(1), make_question(2)]
    strategy = FakeStrategy(fail={qs[1].question: RuntimeError("x")})
    metrics = compute_metrics(qs, run_eval(strategy, qs, workers=1))
    assert metrics.overall.n == 1


def test_git_state_in_this_repo_and_outside(tmp_path: Path):
    inside = git_state(ROOT)
    assert inside.commit is not None and len(inside.commit) == 40
    assert git_state(tmp_path) == GitState(commit=None, dirty=True)


# -- comando ------------------------------------------------------------------


class FakeClient:
    def __init__(self, config, *, cache=None, max_usd=0.0, env=None) -> None:
        pass

    def chat_parsed(self, role, messages, schema, **params):
        question = messages[-1]["content"]
        cited = ["lgpd:art:19"] if "confirmação" in question else ["lgpd:art:1"]
        value = BaselineOutput(answer="Resposta.", cited_articles=cited, confidence=0.5)
        usage = Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
        return Parsed(value=value, result=ChatResult(text="{}", usage=usage, latency_ms=10))


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAB_MODELS_CONFIG", str(ROOT / "config" / "models.yaml"))
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(ROOT / "config" / "corpus.yaml"))
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "baseline_v1.md").write_text("Normas:\n{laws}\n", encoding="utf-8")
    monkeypatch.setattr(lab.cli, "LLMClient", FakeClient)
    return tmp_path


ARGS = [
    "eval",
    "--questions",
    str(ROOT / "data" / "eval" / "questions.dev.jsonl"),
    "--articles",
    str(ROOT / "data" / "processed" / "articles.jsonl"),
]


def test_eval_command_writes_run_and_prints_table(cli_env: Path):
    result = runner.invoke(app, [*ARGS, "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert "categoria" in result.stdout and "TOTAL" in result.stdout
    assert "sem_resposta" in result.stdout and "multi_lei" in result.stdout
    assert "perguntas de rascunho" in result.stderr  # dev_draft: run não oficial
    (run_dir,) = list((cli_env / "results").iterdir())
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text("utf-8"))
    assert meta.questions == 16 and meta.errors == 0 and not meta.official
    assert len((run_dir / "answers.jsonl").read_text("utf-8").splitlines()) == 16


def test_eval_command_limit(cli_env: Path):
    result = runner.invoke(app, [*ARGS, "--limit", "3"])
    assert result.exit_code == 0, result.output
    (run_dir,) = list((cli_env / "results").iterdir())
    assert len((run_dir / "answers.jsonl").read_text("utf-8").splitlines()) == 3


def test_eval_command_rejects_a_question_set_that_contradicts_the_corpus(cli_env: Path):
    bad = make_question(1, gold_articles=["lgpd:art:999"])
    path = cli_env / "bad.jsonl"
    path.write_text(bad.model_dump_json() + "\n", encoding="utf-8")
    result = runner.invoke(app, ["eval", "--questions", str(path), "--articles", ARGS[4]])
    assert result.exit_code == 1
    assert "lgpd:art:999 não existe" in result.output
    assert not (cli_env / "results").exists()  # nada foi gasto nem gravado


def test_eval_command_missing_questions_file(cli_env: Path):
    result = runner.invoke(app, ["eval", "--questions", str(cli_env / "nope.jsonl")])
    assert result.exit_code == 1
    assert "arquivo não encontrado" in result.output
