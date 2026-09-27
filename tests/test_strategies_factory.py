# -----------------------------------------------------------------------------
# File:     tests/test_strategies_factory.py
# Purpose:  Tests for the strategy factory and the few-shot prompt (no leakage).
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
import re
from pathlib import Path

import pytest

from lab.eval.dataset import load_questions
from lab.ingest.export import read_jsonl
from lab.ingest.sources import LawSource
from lab.strategies.factory import (
    STRATEGIES,
    UnknownStrategyError,
    build_strategy,
    default_prompt,
)

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "prompts"
LAWS = [LawSource(id="lgpd", name="LGPD", number="Lei 13.709/2018", url="u")]


def fewshot_examples() -> list[tuple[str, dict]]:
    text = (PROMPTS / "fewshot_v1.md").read_text(encoding="utf-8")
    pairs = re.findall(r"Pergunta: (.+)\nResposta:\n(\{.+\})", text)
    return [(question, json.loads(answer)) for question, answer in pairs]


def test_every_strategy_has_an_existing_default_prompt():
    for name, prompt in STRATEGIES.items():
        assert default_prompt(name) == prompt
        assert (PROMPTS / f"{prompt}.md").exists(), prompt


def test_unknown_strategy():
    with pytest.raises(UnknownStrategyError, match="opções: baseline"):
        default_prompt("rag")
    with pytest.raises(UnknownStrategyError):
        build_strategy("rag", object(), LAWS, prompts_dir=PROMPTS)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["baseline", "fewshot"])
def test_build_strategy_sets_name_and_prompt(name: str):
    strategy = build_strategy(name, object(), LAWS, prompts_dir=PROMPTS)  # type: ignore[arg-type]
    assert strategy.name == name
    assert "Lei 13.709/2018 (lgpd)" in strategy._system  # type: ignore[attr-defined]


def test_explicit_prompt_overrides_the_default():
    strategy = build_strategy(
        "fewshot",
        object(),
        LAWS,
        prompt="baseline_v1",
        prompts_dir=PROMPTS,  # type: ignore[arg-type]
    )
    assert "Exemplos de perguntas" not in strategy._system  # type: ignore[attr-defined]


def test_fewshot_examples_are_valid_json_answers():
    examples = fewshot_examples()
    assert len(examples) >= 3
    for _, answer in examples:
        assert set(answer) == {"answer", "cited_articles", "confidence"}
    assert any(a["cited_articles"] == [] for _, a in examples)  # ensina a se abster


def test_fewshot_examples_cite_real_articles():
    corpus = {r.id: r.status for r in read_jsonl(ROOT / "data" / "processed" / "articles.jsonl")}
    for _, answer in fewshot_examples():
        for article in answer["cited_articles"]:
            assert corpus.get(article) == "vigente", article


def test_fewshot_examples_do_not_leak_the_evaluation_set():
    """Nenhum exemplo pode repetir pergunta ou artigo de gabarito do conjunto de avaliação."""
    example_questions = {q.lower() for q, _ in fewshot_examples()}
    example_articles = {a for _, ans in fewshot_examples() for a in ans["cited_articles"]}
    for path in sorted((ROOT / "data" / "eval").glob("questions*.jsonl")):
        for question in load_questions(path):
            assert question.question.lower() not in example_questions, question.id
            overlap = example_articles & set(question.gold_articles)
            assert not overlap, f"{path.name}:{question.id} usa {overlap}"
