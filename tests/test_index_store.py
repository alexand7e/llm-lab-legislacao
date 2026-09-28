# -----------------------------------------------------------------------------
# File:     tests/test_index_store.py
# Purpose:  Tests for the Qdrant collection, index build and `lab index` (no network).
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from datetime import date
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

import lab.cli
from lab.cli import app
from lab.index.build import build_index
from lab.index.chunking import Chunk
from lab.index.store import (
    IndexNotReadyError,
    VectorIndex,
    collection_name,
    open_index,
    point_id,
)
from lab.ingest.export import ArticleRecord
from lab.ingest.sources import LawSource
from lab.llm import Embeddings
from lab.llm.costs import Usage

ROOT = Path(__file__).parent.parent
runner = CliRunner()


def chunk(article_id: str) -> Chunk:
    law, _, number = article_id.partition(":art:")
    return Chunk(
        id=article_id, article_id=article_id, law=law, article=number, path=["Título I"],
        text=f"Art. {number}.", embed_text=f"cabeçalho\nArt. {number}.",
    )  # fmt: skip


CHUNKS = [chunk("lgpd:art:1"), chunk("lgpd:art:2"), chunk("cdc:art:1")]
VECTORS = [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]]


def memory_index() -> VectorIndex:
    return VectorIndex(QdrantClient(":memory:"), "teste")


def test_collection_name_is_safe_and_depends_on_granularity_and_model():
    assert collection_name("article", "BAAI/bge-m3") == "lab_article_bge-m3"
    assert collection_name("unit", "Some Model v2") == "lab_unit_some-model-v2"


def test_point_id_is_a_stable_uuid():
    assert point_id("lgpd:art:1") == point_id("lgpd:art:1") != point_id("lgpd:art:2")
    assert len(point_id("x")) == 36


def test_rebuild_and_search_orders_by_similarity_and_returns_chunks():
    index = memory_index()
    assert not index.exists() and index.count() == 0
    assert index.rebuild(CHUNKS, VECTORS) == 3
    hits = index.search([1.0, 0.1, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["lgpd:art:1", "lgpd:art:2"]
    assert hits[0].score > hits[1].score
    assert hits[0].chunk == CHUNKS[0]  # o payload devolve o chunk inteiro


def test_search_can_filter_by_law():
    index = memory_index()
    index.rebuild(CHUNKS, VECTORS)
    hits = index.search([1.0, 0.0, 0.0], k=3, laws=["cdc"])
    assert [h.chunk.law for h in hits] == ["cdc"]


def test_rebuild_replaces_instead_of_duplicating():
    index = memory_index()
    index.rebuild(CHUNKS, VECTORS)
    index.rebuild(CHUNKS[:2], VECTORS[:2])
    assert index.count() == 2


def test_rebuild_validates_input():
    index = memory_index()
    with pytest.raises(ValueError, match="3 chunks para 2 vetores"):
        index.rebuild(CHUNKS, VECTORS[:2])
    with pytest.raises(ValueError, match="nenhum chunk"):
        index.rebuild([], [])


def test_search_without_collection_explains_what_to_do():
    with pytest.raises(IndexNotReadyError, match="lab index"):
        memory_index().search([1.0, 0.0, 0.0], k=1)


def test_local_index_persists_on_disk(tmp_path: Path):
    with open_index(None, None, tmp_path / "q", "c") as index:
        index.rebuild(CHUNKS, VECTORS)
    with open_index(None, None, tmp_path / "q", "c") as reopened:
        assert reopened.count() == 3 and not reopened.remote


# -- construção ---------------------------------------------------------------

LAWS = [
    LawSource(id="lgpd", name="LGPD", number="Lei 13.709/2018", url="u"),
    LawSource(id="cdc", name="CDC", number="Lei 8.078/1990", url="u"),
]


def record(article_id: str, status: str = "vigente") -> ArticleRecord:
    law, _, number = article_id.partition(":art:")
    return ArticleRecord(
        id=article_id, law=law, article=number, path=["Título I"], status=status,
        text=f"Art. {number}. Texto.", units=[], references=[], source_hash="sha256:x",
        collected_at=date(2026, 9, 28),
    )  # fmt: skip


class FakeEmbedClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, role, texts, *, batch_size=64):
        self.texts += list(texts)
        vectors = [[float(len(t)), 1.0] for t in texts]
        return Embeddings(vectors=vectors, usage=Usage(prompt_tokens=42, cost_usd=0.01), cached=1)


def test_build_index_embeds_chunks_and_reports():
    records = [record("lgpd:art:1"), record("lgpd:art:2", "revogado"), record("cdc:art:1")]
    client = FakeEmbedClient()
    index = memory_index()
    report = build_index(records, LAWS, client, index)  # type: ignore[arg-type]
    assert report.chunks == 2 and report.by_law == {"lgpd": 1, "cdc": 1}
    assert (report.embed_tokens, report.cached) == (42, 1)
    assert all(text.startswith(("LGPD", "CDC")) for text in client.texts)  # usa embed_text
    assert index.count() == 2


# -- comando ------------------------------------------------------------------


def test_index_command_builds_a_local_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAB_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("LAB_MODELS_CONFIG", str(ROOT / "config" / "models.yaml"))
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(ROOT / "config" / "corpus.yaml"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(lab.cli, "LLMClient", lambda *a, **k: FakeEmbedClient())
    articles = str(ROOT / "data" / "processed" / "articles.jsonl")
    result = runner.invoke(app, ["index", "--articles", articles])
    assert result.exit_code == 0, result.output
    assert "223 chunks" in result.stdout and "lab_article_bge-m3" in result.stdout
    assert "local em" in result.stdout
    with open_index(None, None, tmp_path / "qdrant", "lab_article_bge-m3") as index:
        assert index.count() == 223


def test_index_command_without_corpus_suggests_ingest(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["index", "--articles", str(tmp_path / "nada.jsonl")])
    assert result.exit_code == 1 and "lab ingest" in result.stderr
