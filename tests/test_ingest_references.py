# -----------------------------------------------------------------------------
# File:     tests/test_ingest_references.py
# Purpose:  Unit tests for cross-reference extraction (internal and external).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest

from lab.ingest.parser import parse_html
from lab.ingest.references import article_references, extract_references, known_laws
from lab.ingest.sources import load_corpus

ROOT = Path(__file__).parent.parent
KNOWN = {"lei:8078": "cdc", "lei:12965": "marco_civil"}


def targets(text: str, own: str | None = None) -> list[str]:
    return [r.target for r in extract_references(text, "lgpd", KNOWN, own)]


# -- internas -----------------------------------------------------------------


def test_single_article():
    refs = extract_references("Nos termos do art. 7º, o controlador", "lgpd")
    assert [(r.target, r.raw) for r in refs] == [("lgpd:art:7", "art. 7º")]


def test_list_of_articles():
    assert targets("os arts. 7º, 9º e 11 desta Lei") == [
        "lgpd:art:7",
        "lgpd:art:9",
        "lgpd:art:11",
    ]


def test_added_article_keeps_its_suffix():
    assert targets("conforme o art. 55-A") == ["lgpd:art:55-A"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("arts. 7º a 10", ["7", "8", "9", "10"]),
        ("arts. 55-A a 55-D", ["55-A", "55-B", "55-C", "55-D"]),
    ],
)
def test_ranges_are_expanded(text: str, expected: list[str]):
    assert [t.split(":")[-1] for t in targets(text)] == expected


def test_huge_range_is_not_expanded():
    assert targets("arts. 1º a 500") == ["lgpd:art:1", "lgpd:art:500"]


def test_reference_to_itself_is_dropped():
    assert targets("o caput do art. 12 e o art. 13", own="12") == ["lgpd:art:13"]


def test_repeated_reference_is_reported_once():
    assert targets("art. 7º ... art. 7º") == ["lgpd:art:7"]


# -- externas -----------------------------------------------------------------


def test_article_of_corpus_law_uses_corpus_id():
    refs = extract_references("art. 4º da Lei nº 8.078, de 11 de setembro de 1990", "lgpd", KNOWN)
    assert [(r.target, r.raw) for r in refs] == [("cdc:art:4", "art. 4º da Lei nº 8.078")]


def test_article_of_unknown_law_uses_stable_key():
    assert targets("art. 3º do Decreto nº 8.771, de 2016") == ["decreto:8771:art:3"]


def test_qualifiers_between_article_and_law():
    assert targets("art. 5º, inciso X, da Constituição Federal") == ["cf:art:5"]
    assert targets("art. 6º, § 1º, da Lei nº 8.078") == ["cdc:art:6"]


def test_named_code():
    assert targets("art. 186 do Código Civil") == ["cc:art:186"]
    assert targets("art. 2º do Código de Defesa do Consumidor") == ["cdc:art:2"]


def test_law_without_article():
    assert targets("A Lei nº 12.965, de 23 de abril de 2014, dispõe") == ["marco_civil"]


def test_mention_of_own_law_is_not_a_reference():
    refs = extract_references("Lei nº 13.709, de 2018", "lgpd", {"lei:13709": "lgpd"})
    assert refs == []


# -- o que não é remissão -----------------------------------------------------


def test_quoted_text_is_ignored():
    assert targets('"Art. 4º texto de outra lei"\nvide art. 9º') == ["lgpd:art:9"]


def test_text_without_reference():
    assert targets("O controlador deve informar o titular.") == []


# -- integração ---------------------------------------------------------------


def test_known_laws_from_corpus():
    corpus = load_corpus(ROOT / "config" / "corpus.yaml")
    assert known_laws(corpus) == {
        "lei:13709": "lgpd",
        "lei:12965": "marco_civil",
        "lei:8078": "cdc",
    }


def test_article_references_skip_revoked_units():
    html = (
        "<html><body>"
        "<p>Art. 1º Aplica-se o art. 5º.</p>"
        "<p>I - conforme o art. 6º;</p>"
        "<p>II - (Revogado). Ver art. 7º</p>"
        "</body></html>"
    )
    (article,) = parse_html(html, "x")
    assert [r.target for r in article_references(article)] == ["x:art:5", "x:art:6"]


def test_real_fixtures_produce_expected_references():
    corpus = load_corpus(ROOT / "config" / "corpus.yaml")
    known = known_laws(corpus)
    found: dict[str, set[str]] = {}
    for law in corpus.laws:
        html = (ROOT / "tests" / "fixtures" / f"{law.id}.html").read_text(encoding="utf-8")
        articles = parse_html(html, law.id)
        found[law.id] = {r.target for a in articles for r in article_references(a, known)}
    assert {"lgpd:art:7", "lgpd:art:11"} <= found["lgpd"]  # interna, "arts. 7º e 11"
    assert "lei:9307" in found["lgpd"]  # externa, fora do corpus
    assert {"cf:art:170", "lei:7347:art:1"} <= found["cdc"]  # externas com artigo
