# -----------------------------------------------------------------------------
# File:     tests/test_ingest_export.py
# Purpose:  Unit tests for articles.jsonl export, validation and ordering.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from datetime import date
from pathlib import Path

import pytest

from lab.ingest.collect import RawDocument
from lab.ingest.export import (
    ArticleRecord,
    CorpusError,
    build_records,
    dangling_references,
    read_jsonl,
    sort_records,
    write_jsonl,
)
from lab.ingest.references import known_laws
from lab.ingest.sources import load_corpus

ROOT = Path(__file__).parent.parent
DAY = date(2026, 9, 24)


def doc(law: str = "x", sha: str = "sha256:abc") -> RawDocument:
    return RawDocument(law=law, url="https://example.test", sha256=sha, collected_at=DAY)


def html(*paragraphs: str) -> str:
    return "<html><body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"


def records(*paragraphs: str, law: str = "x") -> list[ArticleRecord]:
    return build_records(html(*paragraphs), doc(law))


def test_record_follows_spec_schema():
    (r,) = records(
        "CAPÍTULO I DISPOSIÇÕES",
        "Art. 7º O tratamento exige base legal, conforme o art. 11.",
        "I - consentimento;",
        "§ 1º Vale o art. 5º.",
    )
    data = json.loads(r.model_dump_json())
    assert data["id"] == "x:art:7"
    assert data["law"] == "x" and data["article"] == "7"
    assert data["path"] == ["Capítulo I"]
    assert data["status"] == "vigente"
    assert data["text"].startswith("Art. 7º O tratamento exige base legal")
    assert data["units"][0] == {
        "type": "inciso",
        "id": "I",
        "key": "I",
        "parent": None,
        "text": "consentimento;",
        "status": "vigente",
    }
    assert [x["target"] for x in data["references"]] == ["x:art:11", "x:art:5"]
    assert data["source_hash"] == "sha256:abc"
    assert data["collected_at"] == "2026-09-24"


def test_revoked_unit_keeps_status_but_not_text():
    (r,) = records("Art. 1º Texto.", "I - (Revogado).", "II - vigente;")
    assert [u.status for u in r.units] == ["revogado", "vigente"]
    assert "Revogado" not in r.text


def test_write_read_round_trip_is_lossless(tmp_path: Path):
    recs = records("Art. 1º A.", "Art. 2º B, art. 1º.", "Art. 55-A C.")
    path = tmp_path / "processed" / "articles.jsonl"
    write_jsonl(path, recs, ["x"])
    assert read_jsonl(path) == sort_records(recs, ["x"])


def test_output_is_deterministic_and_naturally_ordered(tmp_path: Path):
    recs = records("Art. 10. J.", "Art. 55-A K.", "Art. 2º B.", "Art. 55 C.")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, recs, ["x"])
    write_jsonl(b, list(reversed(recs)), ["x"])
    assert a.read_bytes() == b.read_bytes()
    assert [r.article for r in read_jsonl(a)] == ["2", "10", "55", "55-A"]
    assert a.read_bytes().endswith(b"\n") and b"\r" not in a.read_bytes()


def test_laws_follow_corpus_order():
    both = records("Art. 1º A.") + records("Art. 1º B.", law="y")
    assert [r.law for r in sort_records(both, ["y", "x"])] == ["y", "x"]


def test_duplicate_ids_are_rejected_and_nothing_is_written(tmp_path: Path):
    recs = records("Art. 1º A.", "Art. 1º B.")
    path = tmp_path / "articles.jsonl"
    with pytest.raises(CorpusError, match="x:art:1"):
        write_jsonl(path, recs, ["x"])
    assert not path.exists()


def test_invalid_line_reports_line_number(tmp_path: Path):
    path = tmp_path / "articles.jsonl"
    write_jsonl(path, records("Art. 1º A."), ["x"])
    bad = json.loads(path.read_text("utf-8"))
    bad["id"] = "x:art:999"  # não confere com law/article
    path.write_text(path.read_text("utf-8") + json.dumps(bad) + "\n", "utf-8")
    with pytest.raises(CorpusError, match=r"articles\.jsonl:2"):
        read_jsonl(path)


def test_invalid_source_hash_is_rejected():
    with pytest.raises(ValueError, match="sha256"):
        ArticleRecord(
            id="x:art:1",
            law="x",
            article="1",
            path=[],
            status="vigente",
            text="",
            units=[],
            references=[],
            source_hash="abc",
            collected_at=DAY,
        )


def test_dangling_references_are_reported_not_fatal():
    other_law = records("Art. 1º Vale o art. 9º da Lei nº 1.", "Art. 2º Fim.")
    assert dangling_references(other_law) == []  # art. 9 é de outra lei, não do corpus
    missing = records("Art. 1º Vale o art. 3º.", "Art. 2º Fim.")
    assert dangling_references(missing) == [("x:art:1", "x:art:3")]


def test_real_fixtures_export_and_validate(tmp_path: Path):
    corpus = load_corpus(ROOT / "config" / "corpus.yaml")
    known = known_laws(corpus)
    recs: list[ArticleRecord] = []
    for law in corpus.laws:
        text = (ROOT / "tests" / "fixtures" / f"{law.id}.html").read_text(encoding="utf-8")
        recs += build_records(text, doc(law.id, "sha256:fixture"), known)
    path = tmp_path / "articles.jsonl"
    order = [law.id for law in corpus.laws]
    write_jsonl(path, recs, order)
    assert read_jsonl(path) == sort_records(recs, order)
    assert {r.law for r in recs} == set(order)
