# -----------------------------------------------------------------------------
# File:     scripts/make_fixtures.py
# Purpose:  Cut small real-HTML excerpts of collected norms into tests/fixtures/.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Gerar as fixtures de HTML do parser a partir de ``data/raw/``.

Cada fixture é uma sequência de parágrafos ``<p>`` reais da norma (marcação
original, incluindo texto riscado e anotações). Os trechos vêm de
``tests/fixtures/selections.yaml``: pares de regex sobre o texto do
parágrafo, início inclusivo e fim exclusivo (nulo = até o fim).

Uso (após coletar as normas em ``data/raw/``):

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Tag

from lab.ingest.collect import load_raw

RAW_DIR = Path("data/raw")
OUT_DIR = Path("tests/fixtures")
SELECTIONS = OUT_DIR / "selections.yaml"


def _text(p: Tag) -> str:
    return re.sub(r"\s+", " ", p.get_text(" ", strip=True))


def excerpt(html: str, selections: list[tuple[str, str | None]]) -> list[Tag]:
    """Parágrafos de ``html`` dentro dos intervalos ``[início, fim)`` pedidos."""
    paragraphs = BeautifulSoup(html, "lxml").find_all("p")
    texts = [_text(p) for p in paragraphs]
    picked: list[Tag] = []
    for start, end in selections:
        i = next(k for k, t in enumerate(texts) if re.search(start, t))
        j = len(texts)
        if end is not None:
            j = next(k for k in range(i + 1, len(texts)) if re.search(end, texts[k]))
        picked.extend(paragraphs[i:j])
    return picked


def render(name: str, paragraphs: list[Tag], source: str) -> str:
    body = "\n".join(str(p) for p in paragraphs)
    return (
        f"<!-- Recorte de {source} gerado por scripts/make_fixtures.py; não editar à mão. -->\n"
        f'<html><head><meta charset="utf-8"><title>{name}</title></head><body>\n'
        f"{body}\n</body></html>\n"
    )


def main() -> None:
    selections: dict[str, list[tuple[str, str | None]]] = yaml.safe_load(
        SELECTIONS.read_text(encoding="utf-8")
    )
    for name, ranges in selections.items():
        html, doc = load_raw(RAW_DIR, name)
        paragraphs = excerpt(html, [(start, end) for start, end in ranges])
        source = f"{doc.url} ({doc.sha256[:19]}…, coletado em {doc.collected_at})"
        out = OUT_DIR / f"{name}.html"
        out.write_text(render(name, paragraphs, source), encoding="utf-8", newline="\n")
        print(f"{out}: {len(paragraphs)} parágrafos")


if __name__ == "__main__":
    main()
