# -----------------------------------------------------------------------------
# File:     tests/test_fixtures.py
# Purpose:  Sanity checks on the parser HTML fixtures, whatever norms they hold.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import re
from pathlib import Path

import pytest
import yaml
from bs4 import BeautifulSoup

FIXTURES = Path(__file__).parent / "fixtures"
HTML_FILES = sorted(FIXTURES.glob("*.html"))


def paragraphs(path: Path) -> list[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    return [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in soup.find_all("p")]


def test_every_selection_has_a_fixture_and_vice_versa():
    selections = yaml.safe_load((FIXTURES / "selections.yaml").read_text(encoding="utf-8"))
    assert {p.stem for p in HTML_FILES} == set(selections)


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda p: p.stem)
def test_fixture_records_origin(path):
    head = path.read_text(encoding="utf-8").splitlines()[0]
    assert "gerado por scripts/make_fixtures.py" in head and "sha256:" in head


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda p: p.stem)
def test_fixture_has_articles_and_signature(path):
    lines = paragraphs(path)
    assert any(re.match(r"Art\. 1\s*[º°]", line) for line in lines)
    assert any(re.match(r"Brasília\s*,", line) for line in lines)


def test_fixtures_include_struck_text():
    # O parser precisa descartar redações riscadas; ao menos um recorte as contém.
    assert any("<strike" in p.read_text(encoding="utf-8") for p in HTML_FILES)
