# -----------------------------------------------------------------------------
# File:     src/lab/index/build.py
# Purpose:  Build the vector index: chunk the corpus, embed and write the collection.
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Construção do índice (lógica por trás de ``lab index``).

Corpus → chunks → embeddings (papel ``embedding``, com cache por texto) →
coleção no Qdrant. Separado do CLI para ser testado sem terminal nem rede.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from lab.index.chunking import Granularity, make_chunks
from lab.index.store import VectorIndex
from lab.ingest.export import ArticleRecord
from lab.ingest.sources import LawSource
from lab.llm import LLMClient


class BuildReport(BaseModel):
    """Resumo da indexação."""

    collection: str
    granularity: str
    chunks: int
    by_law: dict[str, int]
    embed_tokens: int
    embed_cost_usd: float
    cached: int


def build_index(
    records: Sequence[ArticleRecord],
    laws: Sequence[LawSource],
    client: LLMClient,
    index: VectorIndex,
    *,
    granularity: Granularity = "article",
    role: str = "embedding",
) -> BuildReport:
    """Recriar a coleção de ``index`` com os chunks do corpus."""
    chunks = make_chunks(records, laws, granularity)
    embeddings = client.embed(role, [c.embed_text for c in chunks])
    index.rebuild(chunks, embeddings.vectors)
    by_law: dict[str, int] = {}
    for chunk in chunks:
        by_law[chunk.law] = by_law.get(chunk.law, 0) + 1
    return BuildReport(
        collection=index.collection,
        granularity=granularity,
        chunks=len(chunks),
        by_law=by_law,
        embed_tokens=embeddings.usage.prompt_tokens,
        embed_cost_usd=embeddings.usage.cost_usd,
        cached=embeddings.cached,
    )


__all__ = ["BuildReport", "build_index"]
