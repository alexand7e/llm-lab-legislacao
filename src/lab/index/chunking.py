# -----------------------------------------------------------------------------
# File:     src/lab/index/chunking.py
# Purpose:  Split the corpus into retrievable chunks (one per article, for now).
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Chunking do corpus para o índice vetorial.

Um :class:`Chunk` é a unidade recuperada pela busca. Cada chunk sabe de que
artigo veio (``article_id``), porque a avaliação mede a recuperação por artigo
(recall@k e MRR contra ``gold_articles``) e a resposta cita artigos.

``text`` é o que vai ao modelo de geração (texto oficial, só vigente).
``embed_text`` é o que vai ao modelo de embedding: o mesmo texto precedido de um
cabeçalho com a norma e a posição na hierarquia ("LGPD — Capítulo III — Art.
19"), para a busca achar o artigo por perguntas que citam a lei ou o tema do
capítulo, mesmo quando o artigo não repete essas palavras.

Estratégias:

- ``article`` (#35): um chunk por artigo vigente.
- ``unit`` (#36): um chunk para o caput (com seus incisos e alíneas) e um para
  cada parágrafo (com os seus). Artigo sem parágrafos vira um chunk só, igual ao
  ``article``. O trecho de parágrafo leva o número do artigo no início ("Art. 19,
  § 1º ..."), para o gerador saber de onde veio.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from lab.ingest.export import ArticleRecord
from lab.ingest.parser import article_key
from lab.ingest.sources import LawSource

Granularity = Literal["article", "unit"]
GRANULARITIES: tuple[Granularity, ...] = ("article", "unit")

_PARAGRAPH = re.compile(r"^(?:§\s*(?P<num>\d+)|(?P<unico>Par[áa]grafo [úu]nico))")


class Chunk(BaseModel):
    """Unidade de recuperação."""

    id: str  # estável: "<article_id>" ou "<article_id>#<unidade>"
    article_id: str
    law: str
    article: str
    path: list[str]
    text: str
    embed_text: str


def law_header(law: LawSource | None, law_id: str) -> str:
    """Rótulo da norma no cabeçalho: nome e número, ou o id se não houver fonte."""
    return f"{law.name} ({law.number})" if law else law_id


def chunk_by_article(
    records: Sequence[ArticleRecord], laws: Sequence[LawSource] = ()
) -> list[Chunk]:
    """Um chunk por artigo vigente, em ordem de norma (corpus) e artigo (natural)."""
    by_id = {law.id: law for law in laws}
    rank = {law.id: i for i, law in enumerate(laws)}
    ordered = sorted(
        (r for r in records if r.status == "vigente"),
        key=lambda r: (rank.get(r.law, len(rank)), r.law, article_key(r.article)),
    )
    chunks: list[Chunk] = []
    for r in ordered:
        header = " — ".join([law_header(by_id.get(r.law), r.law), *r.path])
        chunks.append(
            Chunk(
                id=r.id,
                article_id=r.id,
                law=r.law,
                article=r.article,
                path=r.path,
                text=r.text,
                embed_text=f"{header}\n{r.text}",
            )
        )
    return chunks


def split_units(text: str) -> list[tuple[str, str]]:
    """Dividir o texto de um artigo em ``(chave, trecho)``: caput e cada parágrafo.

    Chaves: ``caput``, ``§1``, ``§2``..., ``unico``. Incisos e alíneas ficam no
    trecho do caput ou do parágrafo a que pertencem (vêm logo depois dele no texto).
    """
    parts: list[tuple[str, list[str]]] = [("caput", [])]
    for line in text.splitlines():
        m = _PARAGRAPH.match(line)
        if m:
            parts.append(("unico" if m["unico"] else f"§{m['num']}", [line]))
        else:
            parts[-1][1].append(line)
    return [(key, "\n".join(lines).strip()) for key, lines in parts if "".join(lines).strip()]


def chunk_by_unit(records: Sequence[ArticleRecord], laws: Sequence[LawSource] = ()) -> list[Chunk]:
    """Um chunk por caput e por parágrafo de cada artigo vigente."""
    chunks: list[Chunk] = []
    for article in chunk_by_article(records, laws):
        units = split_units(article.text)
        if len(units) == 1:
            chunks.append(article)
            continue
        header = article.embed_text.split("\n", 1)[0]
        label = f"Art. {article.article}"
        for key, text in units:
            body = text if key == "caput" else f"{label}, {text}"
            chunks.append(
                article.model_copy(
                    update={
                        "id": f"{article.article_id}#{key}",
                        "text": body,
                        "embed_text": f"{header}\n{body}",
                    }
                )
            )
    return chunks


def make_chunks(
    records: Sequence[ArticleRecord],
    laws: Sequence[LawSource] = (),
    granularity: Granularity = "article",
) -> list[Chunk]:
    """Chunks do corpus na granularidade pedida."""
    if granularity == "unit":
        return chunk_by_unit(records, laws)
    return chunk_by_article(records, laws)


__all__ = [
    "GRANULARITIES",
    "Chunk",
    "Granularity",
    "chunk_by_article",
    "chunk_by_unit",
    "law_header",
    "make_chunks",
    "split_units",
]
