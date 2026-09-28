# -----------------------------------------------------------------------------
# File:     src/lab/strategies/rag.py
# Purpose:  Vector RAG strategy: retrieve the top-k articles and answer from them.
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Estratégia ``rag`` (M4): busca vetorial e resposta só com os trechos recuperados.

Para cada pergunta: embedding da pergunta (papel ``embedding``), busca dos
``k`` chunks mais próximos no Qdrant e geração com esses trechos junto da
pergunta. O prompt (``rag_v1``) exige que toda afirmação cite um artigo dos
trechos, com o id que aparece entre colchetes.

``Answer.retrieved_ids`` guarda os artigos recuperados na ordem do ranking, o
que liga as métricas de recuperação (recall@k, MRR). O custo do embedding da
pergunta soma no custo da resposta: é parte do custo da técnica.

As citações do modelo não são filtradas pelos trechos: se ele citar um artigo
que não recebeu, isso aparece na precisão de citação em vez de ser escondido.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lab.index.store import Hit, VectorIndex
from lab.ingest.sources import LawSource
from lab.llm.client import LLMClient
from lab.strategies.baseline import BaselineStrategy, Context


def render_hits(hits: Sequence[Hit]) -> str:
    """Trechos para o prompt, do mais ao menos similar, com o id do artigo."""
    blocks = [f"[{h.chunk.article_id}] {h.chunk.text}" for h in hits]
    return "Trechos recuperados:\n\n" + "\n\n".join(blocks)


def unique_articles(hits: Sequence[Hit]) -> list[str]:
    """Artigos recuperados, na ordem do ranking, sem repetição (vários chunks por artigo)."""
    return list(dict.fromkeys(h.chunk.article_id for h in hits))


class RagStrategy(BaselineStrategy):
    """Baseline + busca vetorial dos ``k`` chunks mais próximos da pergunta.

    :param index: coleção do Qdrant já construída (``lab index``).
    :param k: quantos chunks recuperar.
    :param embedding_role: papel de embedding em ``models.yaml``; deve ser o mesmo
        usado para indexar (a coleção leva o nome do modelo).
    """

    def __init__(
        self,
        client: LLMClient,
        laws: Sequence[LawSource],
        index: VectorIndex,
        *,
        k: int = 5,
        prompt: str = "rag_v1",
        role: str = "generator",
        embedding_role: str = "embedding",
        max_tokens: int = 4000,
        prompts_dir: Path = Path("prompts"),
        name: str = "rag",
    ) -> None:
        if k < 1:
            raise ValueError("k deve ser >= 1")
        super().__init__(
            client,
            laws,
            prompt=prompt,
            role=role,
            max_tokens=max_tokens,
            prompts_dir=prompts_dir,
            name=name,
        )
        self._index = index
        self.k = k
        self._embedding_role = embedding_role

    def retrieve(self, question: str) -> tuple[list[Hit], Context]:
        """Chunks recuperados e o contexto montado para a pergunta."""
        embedded = self._client.embed(self._embedding_role, [question])
        hits = self._index.search(embedded.vectors[0], self.k)
        context = Context(
            text=render_hits(hits), retrieved_ids=unique_articles(hits), usage=embedded.usage
        )
        return hits, context

    def _context(self, question: str) -> Context:
        return self.retrieve(question)[1]


__all__ = ["RagStrategy", "render_hits", "unique_articles"]
