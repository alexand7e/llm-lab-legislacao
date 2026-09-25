# -----------------------------------------------------------------------------
# File:     tests/test_ingest_sources.py
# Purpose:  Unit tests for the config/corpus.yaml loader.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lab.ingest.sources import load_corpus

REPO_CORPUS = Path(__file__).resolve().parents[1] / "config" / "corpus.yaml"


def write(tmp_path: Path, *ids: str) -> Path:
    laws = [{"id": i, "name": "N", "number": "Lei 1/2000", "url": "https://x.test/"} for i in ids]
    path = tmp_path / "corpus.yaml"
    path.write_text(yaml.safe_dump({"laws": laws}), encoding="utf-8")
    return path


def test_repo_corpus_is_valid():
    corpus = load_corpus(REPO_CORPUS)
    assert corpus.laws
    assert all(law.url.startswith("https://") for law in corpus.laws)


def test_by_id_keeps_file_order(tmp_path):
    corpus = load_corpus(write(tmp_path, "b", "a"))
    assert list(corpus.by_id()) == ["b", "a"]


def test_duplicated_ids_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicated law ids: a"):
        load_corpus(write(tmp_path, "a", "b", "a"))


@pytest.mark.parametrize("bad_id", ["LGPD", "lei-1", "a:b"])
def test_id_must_be_a_safe_prefix(tmp_path, bad_id):
    with pytest.raises(ValidationError, match="id"):
        load_corpus(write(tmp_path, bad_id))


def test_empty_corpus_is_rejected(tmp_path):
    path = tmp_path / "corpus.yaml"
    path.write_text("laws: []\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="laws"):
        load_corpus(path)


def test_missing_field_names_the_field(tmp_path):
    path = tmp_path / "corpus.yaml"
    path.write_text("laws:\n  - { id: a, name: N, number: X }\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="url"):
        load_corpus(path)
