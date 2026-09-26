# -----------------------------------------------------------------------------
# File:     tests/test_ingest_pipeline.py
# Purpose:  Tests for the ingest pipeline and the `lab ingest` command (no network).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from datetime import date
from pathlib import Path

import httpx2
import pytest
import yaml
from typer.testing import CliRunner

import lab.ingest.pipeline as pipeline
from lab.cli import app
from lab.ingest.collect import RawDocument, html_path, meta_path, sha256_of
from lab.ingest.export import CorpusError, read_jsonl
from lab.ingest.pipeline import run_ingest
from lab.ingest.sources import CorpusConfig, LawSource, load_corpus

FIXTURES = Path(__file__).parent / "fixtures"
LAWS = ("lgpd", "marco_civil", "cdc")
runner = CliRunner()


def corpus_config() -> CorpusConfig:
    return CorpusConfig(
        laws=[
            LawSource(id=i, name=i, number=n, url=f"https://example.test/{i}")
            for i, n in zip(
                LAWS, ("Lei 13.709/2018", "Lei 12.965/2014", "Lei 8.078/1990"), strict=True
            )
        ]
    )


class FakeServer:
    """Serve os HTML das fixtures por URL e conta as chamadas."""

    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.missing = missing

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request.url.path.strip("/"))
        law = request.url.path.strip("/")
        if law in self.missing:
            return httpx2.Response(404)
        return httpx2.Response(200, content=(FIXTURES / f"{law}.html").read_bytes())

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))


def seed_raw(raw_dir: Path, law: str, content: bytes) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path(raw_dir, law).write_bytes(content)
    doc = RawDocument(law=law, url="u", sha256=sha256_of(content), collected_at=date(2026, 9, 24))
    meta_path(raw_dir, law).write_text(doc.model_dump_json(), "utf-8")


def test_collects_missing_laws_then_exports(tmp_path: Path):
    server = FakeServer()
    out = tmp_path / "processed" / "articles.jsonl"
    report = run_ingest(corpus_config(), tmp_path / "raw", out, client=server.client())
    assert server.calls == list(LAWS)
    assert [s.law for s in report.laws] == list(LAWS)
    assert report.total == len(read_jsonl(out)) > 0
    assert all(s.source_hash.startswith("sha256:") for s in report.laws)


def test_reuses_collected_html_without_network(tmp_path: Path):
    raw = tmp_path / "raw"
    for law in LAWS:
        seed_raw(raw, law, (FIXTURES / f"{law}.html").read_bytes())
    server = FakeServer()
    run_ingest(corpus_config(), raw, tmp_path / "a.jsonl", client=server.client())
    assert server.calls == []


def test_fetch_downloads_everything_again(tmp_path: Path):
    raw = tmp_path / "raw"
    for law in LAWS:
        seed_raw(raw, law, (FIXTURES / f"{law}.html").read_bytes())
    server = FakeServer()
    run_ingest(corpus_config(), raw, tmp_path / "a.jsonl", fetch=True, client=server.client())
    assert server.calls == list(LAWS)


def test_output_is_reproducible(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    raw = tmp_path / "raw"
    run_ingest(corpus_config(), raw, a, client=FakeServer().client())
    run_ingest(corpus_config(), raw, b, client=FakeServer().client())  # reusa raw
    assert a.read_bytes() == b.read_bytes()


def test_download_failure_writes_nothing(tmp_path: Path):
    out = tmp_path / "articles.jsonl"
    with pytest.raises(httpx2.HTTPStatusError):
        run_ingest(
            corpus_config(), tmp_path / "raw", out, client=FakeServer(missing=("cdc",)).client()
        )
    assert not out.exists()


def test_tampered_html_is_rejected(tmp_path: Path):
    raw = tmp_path / "raw"
    for law in LAWS:
        seed_raw(raw, law, (FIXTURES / f"{law}.html").read_bytes())
    html_path(raw, "lgpd").write_bytes(b"<p>Art. 1o alterado.</p>")
    with pytest.raises(ValueError, match="sha256"):
        run_ingest(corpus_config(), raw, tmp_path / "a.jsonl", client=FakeServer().client())


def test_law_without_articles_is_an_error(tmp_path: Path):
    raw = tmp_path / "raw"
    for law in LAWS:
        seed_raw(raw, law, (FIXTURES / f"{law}.html").read_bytes())
    seed_raw(raw, "cdc", b"<html><body><p>Sem artigos.</p></body></html>")
    out = tmp_path / "a.jsonl"
    with pytest.raises(CorpusError, match="cdc"):
        run_ingest(corpus_config(), raw, out, client=FakeServer().client())
    assert not out.exists()


# -- comando ------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    server = FakeServer()
    monkeypatch.setattr(pipeline, "make_client", server.client)
    corpus = tmp_path / "corpus.yaml"
    corpus.write_text(yaml.safe_dump(corpus_config().model_dump()), "utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(corpus))
    return tmp_path


def test_cli_ingest_writes_articles_and_prints_summary(cli_env: Path):
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert "lgpd:" in result.stdout and "artigos gravados em" in result.stdout
    assert len(read_jsonl(cli_env / "data" / "processed" / "articles.jsonl")) > 0
    assert load_corpus(cli_env / "corpus.yaml").laws[0].id == "lgpd"


def test_cli_ingest_reports_download_error(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline, "make_client", FakeServer(missing=("lgpd",)).client)
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert not (cli_env / "data" / "processed" / "articles.jsonl").exists()


def test_cli_ingest_reports_missing_corpus(cli_env: Path):
    result = runner.invoke(app, ["ingest", "--corpus", str(cli_env / "nope.yaml")])
    assert result.exit_code == 1
    assert "não encontrado" in result.output
