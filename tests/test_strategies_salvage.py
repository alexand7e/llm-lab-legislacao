# -----------------------------------------------------------------------------
# File:     tests/test_strategies_salvage.py
# Purpose:  Tests for recovering citations when the model ignores the JSON format.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from lab.llm.client import ChatResult
from lab.llm.costs import Usage
from lab.strategies.base import salvage_fields
from lab.strategies.baseline import BaselineStrategy

LAWS = {"lgpd", "cdc", "marco_civil"}

# saída real do soberano-alpha no long_context (q005), quando o formato foi ignorado
REAL = (
    "De acordo com o Código de Defesa do Consumidor, o direito de reclamar por vícios "
    "aparentes em produtos duráveis caduca em **90 (noventa) dias**.\n\n"
    '`cited_articles`: ["cdc:art:26"]\n'
    "`confidence`: 1.0"
)


def test_real_markdown_output_is_recovered():
    text, cited, confidence = salvage_fields(REAL, LAWS)
    assert cited == ["cdc:art:26"] and confidence == 1.0
    assert text.endswith("**90 (noventa) dias**.")
    assert "cited_articles" not in text and "confidence" not in text


@pytest.mark.parametrize(
    "line",
    [
        'cited_articles: ["lgpd:art:19", "LGPD:art:18"]',
        "- **cited_articles**: lgpd:art:19, lgpd:art:18",
        '"cited_articles" = [lgpd:art:19, lgpd:art:18]',
    ],
)
def test_field_label_variants(line: str):
    _, cited, _ = salvage_fields(f"Resposta.\n{line}", LAWS)
    assert cited == ["lgpd:art:19", "lgpd:art:18"]


def test_articles_mentioned_in_the_answer_are_not_citations():
    text, cited, confidence = salvage_fields("Conforme lgpd:art:19, é imediato.", LAWS)
    assert cited == [] and confidence is None
    assert text == "Conforme lgpd:art:19, é imediato."


def test_unknown_laws_are_dropped_and_confidence_is_clamped():
    _, cited, confidence = salvage_fields(
        "x\ncited_articles: [cc:art:186, cdc:art:6]\nconfidence: 95", LAWS
    )
    assert cited == ["cdc:art:6"] and confidence == 1.0
    assert salvage_fields("x\nconfidence: 0,4", LAWS)[2] == 0.4


def test_answer_label_is_removed_but_its_content_kept():
    text, _, _ = salvage_fields('answer: "Em 7 dias."\ncited_articles: [cdc:art:49]', LAWS)
    assert text == "Em 7 dias."


def test_free_text_without_fields_is_left_alone():
    assert salvage_fields("Não sei responder.", LAWS) == ("Não sei responder.", [], None)


def test_baseline_keeps_parse_error_but_recovers_citations():
    strategy = BaselineStrategy.__new__(BaselineStrategy)
    strategy._law_ids = sorted(LAWS)  # type: ignore[attr-defined]
    result = ChatResult(text=REAL, usage=Usage(prompt_tokens=5), latency_ms=10)
    answer = strategy._build_answer(None, "Invalid JSON", result)  # type: ignore[attr-defined]
    assert answer.cited_articles == ["cdc:art:26"] and answer.confidence == 1.0
    assert answer.parse_error == "Invalid JSON"  # a falha de formato continua registrada
    assert "cited_articles" not in answer.text
