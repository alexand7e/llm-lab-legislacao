# -----------------------------------------------------------------------------
# File:     src/lab/strategies/factory.py
# Purpose:  Build a strategy by name, so CLI commands share one list of strategies.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Fábrica de estratégias por nome (``lab eval --strategy``, ``lab ask --strategy``).

Cada estratégia tem um prompt padrão versionado em ``prompts/``; ``prompt``
explícito o substitui, e a versão usada fica registrada na run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lab.index.store import VectorIndex
from lab.ingest.export import ArticleRecord
from lab.ingest.sources import LawSource
from lab.llm.client import LLMClient
from lab.strategies.baseline import BaselineStrategy
from lab.strategies.long_context import LongContextStrategy
from lab.strategies.rag import RagStrategy

# nome -> prompt padrão
STRATEGIES: dict[str, str] = {
    "baseline": "baseline_v1",
    "fewshot": "fewshot_v1",
    "long_context": "long_context_v1",
    "rag": "rag_v1",
}


class UnknownStrategyError(ValueError):
    """Nome de estratégia que a fábrica não conhece."""


def default_prompt(name: str) -> str:
    """Prompt padrão da estratégia.

    :raises UnknownStrategyError: estratégia desconhecida.
    """
    if name not in STRATEGIES:
        raise UnknownStrategyError(
            f"estratégia desconhecida: {name!r} (opções: {', '.join(STRATEGIES)})"
        )
    return STRATEGIES[name]


def build_strategy(
    name: str,
    client: LLMClient,
    laws: Sequence[LawSource],
    *,
    prompt: str | None = None,
    role: str = "generator",
    prompts_dir: Path = Path("prompts"),
    records: Sequence[ArticleRecord] | None = None,
    index: VectorIndex | None = None,
    k: int = 5,
) -> BaselineStrategy:
    """Montar a estratégia ``name``.

    :raises UnknownStrategyError: estratégia desconhecida.
    :param records: artigos do corpus; obrigatório para ``long_context``.
    :raises FileNotFoundError: prompt inexistente.
    :param index: coleção vetorial; obrigatória para ``rag``.
    :param k: chunks recuperados pelo ``rag``.
    :raises ValueError: ``long_context`` sem ``records`` ou ``rag`` sem ``index``.
    """
    chosen = prompt or default_prompt(name)
    default_prompt(name)  # valida o nome mesmo com prompt explícito
    if name == "long_context":
        if records is None:
            raise ValueError("long_context precisa do corpus (articles.jsonl; rode lab ingest)")
        return LongContextStrategy(
            client, laws, records, prompt=chosen, role=role, prompts_dir=prompts_dir
        )
    if name == "rag":
        if index is None:
            raise ValueError("rag precisa do índice vetorial (rode lab index)")
        return RagStrategy(
            client, laws, index, k=k, prompt=chosen, role=role, prompts_dir=prompts_dir
        )
    return BaselineStrategy(
        client, laws, prompt=chosen, role=role, prompts_dir=prompts_dir, name=name
    )


__all__ = ["STRATEGIES", "UnknownStrategyError", "build_strategy", "default_prompt"]
