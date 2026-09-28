# -----------------------------------------------------------------------------
# File:     tests/test_index_chunking.py
# Purpose:  Tests for corpus chunking (one chunk per article in force).
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from datetime import date
from pathlib import Path

from lab.index.chunking import chunk_by_article, make_chunks
from lab.ingest.export import ArticleRecord, read_jsonl
from lab.ingest.sources import LawSource, load_corpus

ROOT = Path(__file__).parent.parent
LAWS = [
    LawSource(id="lgpd", name="Lei Geral de Proteção de Dados", number="Lei 13.709/2018", url="u"),
    LawSource(id="cdc", name="Código de Defesa do Consumidor", number="Lei 8.078/1990", url="u"),
]


def record(
    article_id: str, status: str = "vigente", path: list[str] | None = None
) -> ArticleRecord:
    law, _, number = article_id.partition(":art:")
    return ArticleRecord(
        id=article_id, law=law, article=number, path=path or ["Capítulo III"], status=status,
        text=f"Art. {number}. Texto.", units=[], references=[], source_hash="sha256:x",
        collected_at=date(2026, 9, 28),
    )  # fmt: skip


def test_one_chunk_per_article_in_force_in_corpus_and_natural_order():
    records = [
        record("cdc:art:2"),
        record("lgpd:art:55-A"),
        record("lgpd:art:10"),
        record("lgpd:art:2"),
        record("lgpd:art:3", status="vetado"),
    ]
    chunks = chunk_by_article(records, LAWS)
    assert [c.id for c in chunks] == ["lgpd:art:2", "lgpd:art:10", "lgpd:art:55-A", "cdc:art:2"]
    assert all(c.id == c.article_id for c in chunks)


def test_embed_text_has_law_and_hierarchy_header_but_text_stays_official():
    (chunk,) = chunk_by_article([record("lgpd:art:19", path=["Capítulo III", "Seção I"])], LAWS)
    assert chunk.embed_text.splitlines()[0] == (
        "Lei Geral de Proteção de Dados (Lei 13.709/2018) — Capítulo III — Seção I"
    )
    assert chunk.embed_text.endswith(chunk.text)
    assert chunk.text == "Art. 19. Texto."


def test_unknown_law_uses_its_id_in_the_header():
    (chunk,) = chunk_by_article([record("xyz:art:1")], LAWS)
    assert chunk.embed_text.startswith("xyz — Capítulo III")


def test_real_corpus_chunks_match_articles_in_force():
    records = read_jsonl(ROOT / "data" / "processed" / "articles.jsonl")
    laws = load_corpus(ROOT / "config" / "corpus.yaml").laws
    chunks = make_chunks(records, laws)
    assert len(chunks) == sum(1 for r in records if r.status == "vigente")
    assert len({c.id for c in chunks}) == len(chunks)
    assert max(len(c.embed_text) for c in chunks) < 32_000  # cabe no bge-m3 (8k tokens)
