# -----------------------------------------------------------------------------
# File:     tests/test_strategies_long_context.py
# Purpose:  Tests for the long-context strategy (full law text in the prompt).
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from datetime import date
from pathlib import Path

import pytest

from lab.ingest.export import ArticleRecord, read_jsonl
from lab.ingest.sources import LawSource, load_corpus
from lab.strategies.factory import build_strategy
from lab.strategies.long_context import LongContextStrategy, render_corpus

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "prompts"
LAWS = [
    LawSource(id="lgpd", name="Lei Geral de Proteção de Dados", number="Lei 13.709/2018", url="u"),
    LawSource(id="cdc", name="Código de Defesa do Consumidor", number="Lei 8.078/1990", url="u"),
]


def record(article_id: str, text: str, status: str = "vigente") -> ArticleRecord:
    law, _, number = article_id.partition(":art:")
    return ArticleRecord(
        id=article_id, law=law, article=number, path=["Título I"], status=status, text=text,
        units=[], references=[], source_hash="sha256:x", collected_at=date(2026, 9, 27),
    )  # fmt: skip


RECORDS = [
    record("cdc:art:2", "Art. 2º Consumidor é..."),
    record("lgpd:art:55-A", "Art. 55-A. Fica criada a ANPD."),
    record("lgpd:art:10", "Art. 10. Legítimo interesse."),
    record("lgpd:art:2", "Art. 2º Fundamentos."),
    record("lgpd:art:3", "Art. 3º (VETADO).", status="vetado"),
    record("lgpd:art:55", "Art. 55. (Revogado).", status="revogado"),
]


def test_render_corpus_groups_by_law_in_corpus_order_with_citation_ids():
    text = render_corpus(LAWS, RECORDS)
    lgpd, cdc = text.index("### Lei Geral"), text.index("### Código de Defesa")
    assert lgpd < cdc  # ordem do corpus, não do arquivo
    assert "### Lei Geral de Proteção de Dados, Lei 13.709/2018 (lgpd)" in text
    assert "[lgpd:art:2] Art. 2º Fundamentos." in text
    assert "[cdc:art:2] Art. 2º Consumidor é..." in text


def test_render_corpus_uses_natural_article_order():
    text = render_corpus(LAWS, RECORDS)
    positions = [text.index(f"[lgpd:art:{n}]") for n in ("2", "10", "55-A")]
    assert positions == sorted(positions)


def test_render_corpus_keeps_only_articles_in_force():
    text = render_corpus(LAWS, RECORDS)
    assert "lgpd:art:3" not in text and "lgpd:art:55]" not in text


def test_render_corpus_requires_every_law():
    with pytest.raises(ValueError, match="cdc"):
        render_corpus(LAWS, [r for r in RECORDS if r.law == "lgpd"])


def test_strategy_prompt_has_laws_and_corpus_and_no_placeholders():
    strategy = LongContextStrategy(object(), LAWS, RECORDS, prompts_dir=PROMPTS)  # type: ignore[arg-type]
    system = strategy._system  # type: ignore[attr-defined]
    assert strategy.name == "long_context"
    assert "{laws}" not in system and "{corpus}" not in system
    assert "Lei 13.709/2018 (lgpd)" in system and "[lgpd:art:55-A]" in system
    assert "texto oficial vigente" in system


def test_factory_builds_long_context_and_requires_the_corpus():
    strategy = build_strategy(
        "long_context",
        object(),
        LAWS,
        prompts_dir=PROMPTS,
        records=RECORDS,  # type: ignore[arg-type]
    )
    assert isinstance(strategy, LongContextStrategy)
    with pytest.raises(ValueError, match="articles.jsonl"):
        build_strategy("long_context", object(), LAWS, prompts_dir=PROMPTS)  # type: ignore[arg-type]


def test_real_corpus_fits_the_budget_we_measured():
    """As três leis vigentes ocupam ~172 mil caracteres (~40 mil tokens medidos no provedor)."""
    laws = load_corpus(ROOT / "config" / "corpus.yaml").laws
    text = render_corpus(laws, read_jsonl(ROOT / "data" / "processed" / "articles.jsonl"))
    assert 150_000 < len(text) < 220_000
    assert text.count("\n### ") == len(laws) - 1
