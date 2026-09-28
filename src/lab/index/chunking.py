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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from lab.ingest.export import ArticleRecord
from lab.ingest.parser import article_key
from lab.ingest.sources import LawSource

Granularity = Literal["article"]


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


def make_chunks(
    records: Sequence[ArticleRecord],
    laws: Sequence[LawSource] = (),
    granularity: Granularity = "article",
) -> list[Chunk]:
    """Chunks do corpus na granularidade pedida."""
    return chunk_by_article(records, laws)


__all__ = ["Chunk", "Granularity", "chunk_by_article", "law_header", "make_chunks"]
