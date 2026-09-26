# -----------------------------------------------------------------------------
# File:     tests/test_corpus.py
# Purpose:  Invariants of the versioned data/processed/articles.jsonl (M1 acceptance).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import re
from collections import defaultdict
from pathlib import Path

import pytest

from lab.ingest.export import ArticleRecord, read_jsonl
from lab.ingest.sources import load_corpus

ROOT = Path(__file__).parent.parent

# Numeração oficial dos artigos originais (sem os acrescidos, como 55-A): o
# último artigo de cada lei. Conferido contra o texto do Planalto; é a base
# do aceite "número de artigos confere com a contagem oficial".
LAST_BASE_ARTICLE = {"lgpd": 65, "marco_civil": 32, "cdc": 119}


@pytest.fixture(scope="module")
def by_law() -> dict[str, list[ArticleRecord]]:
    grouped: dict[str, list[ArticleRecord]] = defaultdict(list)
    for record in read_jsonl(ROOT / "data" / "processed" / "articles.jsonl"):
        grouped[record.law].append(record)
    return grouped


def test_corpus_has_exactly_the_configured_laws(by_law: dict[str, list[ArticleRecord]]):
    configured = {law.id for law in load_corpus(ROOT / "config" / "corpus.yaml").laws}
    assert set(by_law) == configured == set(LAST_BASE_ARTICLE)


@pytest.mark.parametrize("law", sorted(LAST_BASE_ARTICLE))
def test_base_articles_are_complete_and_contiguous(
    by_law: dict[str, list[ArticleRecord]], law: str
):
    base = sorted({int(re.match(r"\d+", r.article)[0]) for r in by_law[law]})  # type: ignore[index]
    assert base == list(range(1, LAST_BASE_ARTICLE[law] + 1))


@pytest.mark.parametrize("law", sorted(LAST_BASE_ARTICLE))
def test_every_added_article_follows_an_existing_base_article(
    by_law: dict[str, list[ArticleRecord]], law: str
):
    articles = {r.article for r in by_law[law]}
    added = [a for a in articles if "-" in a]
    assert all(a.split("-")[0] in articles for a in added)


@pytest.mark.parametrize("law", sorted(LAST_BASE_ARTICLE))
def test_records_are_well_formed(by_law: dict[str, list[ArticleRecord]], law: str):
    records = by_law[law]
    assert len({r.source_hash for r in records}) == 1  # um HTML de origem por norma
    assert all(r.path for r in records), "artigo fora da hierarquia de títulos"
    assert all(r.text.strip() for r in records if r.status == "vigente")
    assert all(r.references == [] or r.status == "vigente" for r in records)
