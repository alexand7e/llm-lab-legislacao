# -----------------------------------------------------------------------------
# File:     src/lab/strategies/long_context.py
# Purpose:  Long-context strategy: the full text of every law goes into the prompt.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Estratégia ``long_context``: todas as normas do corpus no prompt (SPEC 10, M3).

Sem recuperação: o texto oficial vigente das três leis (~48 mil tokens) vai
inteiro no system prompt, em todas as perguntas. É o teto do que "ter a lei à
mão" dá sem nenhuma busca, e a referência de custo contra a qual o RAG (M4) se
justifica: o RAG precisa chegar perto dessa correção gastando bem menos tokens.

Entram só artigos vigentes, e o ``text`` de cada artigo já exclui dispositivos
revogados ou vetados (ver ``lab.ingest.export``). Cada artigo é precedido do seu
id de citação (``[lgpd:art:19]``), para o modelo citar no formato que a
avaliação confere. As normas entram todas, sempre: escolher a norma pela
pergunta daria uma dica que as outras estratégias não têm.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lab.ingest.export import ArticleRecord
from lab.ingest.parser import article_key
from lab.ingest.sources import LawSource
from lab.llm.client import LLMClient
from lab.strategies.baseline import BaselineStrategy


def render_corpus(laws: Sequence[LawSource], records: Sequence[ArticleRecord]) -> str:
    """Texto das normas para o prompt: uma seção por norma, artigos vigentes em ordem.

    :raises ValueError: alguma norma do corpus não tem nenhum artigo vigente.
    """
    by_law: dict[str, list[ArticleRecord]] = {}
    for record in records:
        if record.status == "vigente":
            by_law.setdefault(record.law, []).append(record)
    sections: list[str] = []
    for law in laws:
        articles = sorted(by_law.get(law.id, []), key=lambda r: article_key(r.article))
        if not articles:
            raise ValueError(f"nenhum artigo vigente da norma {law.id!r} no corpus")
        body = "\n\n".join(f"[{r.id}] {r.text}" for r in articles)
        sections.append(f"### {law.name}, {law.number} ({law.id})\n\n{body}")
    return "\n\n".join(sections)


class LongContextStrategy(BaselineStrategy):
    """Baseline com o texto integral das normas no system prompt.

    Mesma saída estruturada, histórico e streaming do baseline; só o prompt muda.

    :param records: artigos do corpus (``articles.jsonl``).
    """

    def __init__(
        self,
        client: LLMClient,
        laws: Sequence[LawSource],
        records: Sequence[ArticleRecord],
        *,
        prompt: str = "long_context_v1",
        role: str = "generator",
        max_tokens: int = 4000,
        prompts_dir: Path = Path("prompts"),
        name: str = "long_context",
    ) -> None:
        super().__init__(
            client,
            laws,
            prompt=prompt,
            role=role,
            max_tokens=max_tokens,
            prompts_dir=prompts_dir,
            name=name,
        )
        self._system = self._system.replace("{corpus}", render_corpus(laws, records))


__all__ = ["LongContextStrategy", "render_corpus"]
