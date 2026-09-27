# -----------------------------------------------------------------------------
# File:     tests/test_eval_judge.py
# Purpose:  Tests for the LLM judge: rubric prompt, verdicts, failures and independence.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx2
import pytest
from openai import DefaultHttpxClient

from lab.eval.dataset import Question
from lab.eval.judge import Judge, JudgeError, Verdict, check_judge_independence
from lab.eval.registry import GitState, RunConfig, RunMeta, write_run
from lab.eval.runner import run_eval
from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.costs import Usage
from lab.settings import ModelsConfig, Provider, Role
from lab.strategies.base import Answer

ROOT = Path(__file__).parent.parent


def question(**over) -> Question:
    base = {
        "id": "q001",
        "question": "Qual o prazo para confirmar o tratamento em formato simplificado?",
        "reference_answer": "Imediatamente.",
        "gold_articles": ["lgpd:art:19"],
        "category": "factual",
        "laws": ["lgpd"],
    }
    return Question(**{**base, **over})


def answer(text: str = "Em 15 dias.") -> Answer:
    return Answer(text=text, usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.5))


def config(judge_model: str = "juiz", generator_model: str = "gerador") -> ModelsConfig:
    return ModelsConfig(
        roles={
            "generator": Role(provider="p", model=generator_model),
            "judge": Role(provider="p", model=judge_model, price_in=1.0, price_out=2.0),
        },
        providers={"p": Provider(base_url="https://example.test/v1", api_key_env="K")},
    )


class FakeServer:
    """Responde ``content`` como saída do modelo; ``status`` != 200 simula falha."""

    def __init__(self, content: str, status: int = 200) -> None:
        self.content = content
        self.status = status
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "falhou"}})
        return httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "juiz",
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


def judge_for(server: FakeServer, **kw) -> Judge:
    client = LLMClient(
        config(),
        env={"K": "secret"},
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(server)),
        sleep=lambda _s: None,
        max_retries=0,
        **kw,
    )
    return Judge(client, prompts_dir=ROOT / "prompts")


def verdict_json(score: int, rationale: str = "Diverge da referência.") -> str:
    return json.dumps({"score": score, "rationale": rationale})


# -- rubrica ------------------------------------------------------------------


def test_rubric_prompt_is_versioned_and_covers_every_score():
    text = (ROOT / "prompts" / "judge_v1.md").read_text(encoding="utf-8")
    for marker in ("**2 —", "**1 —", "**0 —", "sem resposta", "rationale"):
        assert marker in text, marker


# -- veredito -----------------------------------------------------------------


@pytest.mark.parametrize(("score", "correctness"), [(0, 0.0), (1, 0.5), (2, 1.0)])
def test_verdict_from_the_model(score: int, correctness: float):
    verdict = judge_for(FakeServer(verdict_json(score, "  Explicação.  "))).evaluate(
        question(), answer()
    )
    assert verdict.score == score and verdict.correctness == correctness
    assert verdict.rationale == "Explicação." and verdict.error is None
    assert verdict.usage.cost_usd == pytest.approx(0.0002)  # custo do juiz, à parte


def test_judge_sees_question_reference_and_answer_with_temperature_zero():
    server = FakeServer(verdict_json(0))
    judge_for(server).evaluate(question(), answer("Em 15 dias."))
    body = server.requests[0]
    system, user = body["messages"]
    assert "rubrica" in system["content"]
    assert "Imediatamente." in user["content"] and "Em 15 dias." in user["content"]
    assert "Qual o prazo" in user["content"]
    assert body["temperature"] == 0 and body["model"] == "juiz"
    assert body["response_format"]["type"] == "json_schema"


def test_unanswerable_question_is_labeled_for_the_judge():
    server = FakeServer(verdict_json(2))
    q = question(category="sem_resposta", gold_articles=[], reference_answer="Fora do corpus.")
    judge_for(server).evaluate(q, answer("Não posso responder."))
    assert "o correto é se abster" in server.requests[0]["messages"][1]["content"]


def test_empty_answer_is_marked_for_the_judge():
    server = FakeServer(verdict_json(0))
    judge_for(server).evaluate(question(), answer("   "))
    assert "(vazia)" in server.requests[0]["messages"][1]["content"]


# -- falhas -------------------------------------------------------------------


@pytest.mark.parametrize("content", ["Nota 2, está certa.", verdict_json(3), '{"score": 1}'])
def test_output_outside_the_schema_becomes_an_error_verdict(content: str):
    verdict = judge_for(FakeServer(content)).evaluate(question(), answer())
    assert verdict.score is None and verdict.correctness is None
    assert "fora do schema" in (verdict.error or "")
    assert verdict.usage.prompt_tokens == 100  # a chamada foi paga e é contada


def test_provider_failure_becomes_an_error_verdict():
    verdict = judge_for(FakeServer("", status=500)).evaluate(question(), answer())
    assert verdict.score is None and verdict.error


def test_budget_is_not_swallowed():
    with pytest.raises(BudgetExceededError):
        judge_for(FakeServer(verdict_json(2)), max_usd=0.0000001).evaluate(question(), answer())


# -- independência (SPEC 4.2) -------------------------------------------------


def test_same_model_for_judge_and_generator_is_refused():
    with pytest.raises(JudgeError, match="mesmo modelo"):
        check_judge_independence(config("m", "m"), judge_role="judge", generator_role="generator")


def test_different_models_are_accepted():
    check_judge_independence(config(), judge_role="judge", generator_role="generator")


def test_unknown_role_is_a_config_error():
    with pytest.raises(LLMConfigError, match="unknown role"):
        check_judge_independence(config(), judge_role="nope", generator_role="generator")


def test_repo_models_have_independent_judge():
    from lab.settings import load_models_config

    models = load_models_config(ROOT / "config" / "models.yaml")
    check_judge_independence(models, judge_role="judge", generator_role="generator")


# -- integração com runner e registro -----------------------------------------


class FakeStrategy:
    name = "fake"

    def answer(self, question: str, history=()) -> Answer:
        return answer("Imediatamente.")


class FixedJudge:
    """Juiz falso: nota fixa, ou erro numa pergunta específica."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def evaluate(self, question: Question, answer: Answer) -> Verdict:
        if question.id == self.fail_on:
            return Verdict(error="juiz fora do schema: x", usage=Usage(cost_usd=0.1))
        return Verdict(score=1, rationale="Parcial.", usage=Usage(cost_usd=0.1))


def test_runner_adds_correctness_and_keeps_the_verdict():
    (result,) = run_eval(FakeStrategy(), [question()], workers=1, judge=FixedJudge())  # type: ignore[arg-type]
    assert result.scores is not None and result.scores.correctness == 0.5
    assert result.verdict is not None and result.verdict.rationale == "Parcial."


def test_run_without_judge_has_no_correctness():
    (result,) = run_eval(FakeStrategy(), [question()], workers=1)
    assert result.scores is not None and result.scores.correctness is None
    assert result.verdict is None


def test_registry_counts_judge_cost_and_errors_and_marks_run_unofficial(tmp_path: Path):
    qs = [question(id="q001"), question(id="q002")]
    results = run_eval(FakeStrategy(), qs, workers=1, judge=FixedJudge(fail_on="q002"))  # type: ignore[arg-type]
    started = datetime(2026, 9, 27, 10, 0)
    run_dir = write_run(
        tmp_path,
        config=RunConfig(
            strategy="fake",
            prompt="p",
            roles={"generator": "g"},
            questions_file="q.jsonl",
            questions_sha256="sha256:x",
            questions_count=2,
            workers=1,
            max_usd=2.0,
            use_cache=True,
            judge_model="juiz",
            judge_prompt="judge_v1",
        ),
        questions=qs,
        results=results,
        started=started,
        finished=started + timedelta(seconds=3),
        git=GitState(commit="abc", dirty=False),
    )
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text("utf-8"))
    assert meta.judge_cost_usd == pytest.approx(0.2)
    assert meta.total_cost_usd == pytest.approx(1.2)  # 2 × 0.5 da estratégia + 0.2 do juiz
    assert meta.judge_errors == 1 and meta.errors == 0
    assert not meta.official  # falha do juiz deixa a métrica de correção incompleta
    answers = [
        json.loads(line) for line in (run_dir / "answers.jsonl").read_text("utf-8").splitlines()
    ]
    assert answers[0]["verdict"]["score"] == 1 and answers[1]["verdict"]["error"]
