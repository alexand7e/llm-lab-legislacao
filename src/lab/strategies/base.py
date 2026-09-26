# -----------------------------------------------------------------------------
# File:     src/lab/strategies/base.py
# Purpose:  Common interface of all question-answering strategies (SPEC 3).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Interface comum das estratégias de resposta.

Baseline, RAG, GraphRAG e o modelo com fine-tuning implementam a mesma
interface, para o runner de avaliação tratá-las de forma igual (SPEC 3): uma
estratégia recebe uma pergunta e devolve uma :class:`Answer` com texto, artigos
citados, ids recuperados, custo e latência.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from lab.llm.costs import Usage


class Answer(BaseModel):
    """Resposta de uma estratégia a uma pergunta.

    - ``text``: a resposta em linguagem natural.
    - ``cited_articles``: ids citados (``lgpd:art:7``), normalizados e sem repetição.
    - ``retrieved_ids``: ids recuperados, na ordem do ranking; vazio no baseline.
    - ``confidence``: confiança declarada pelo modelo (0 a 1), se informou.
    - ``usage``: tokens e custo estimado da chamada.
    - ``latency_ms``: tempo da chamada ao modelo.
    - ``parse_error``: motivo, se a resposta do modelo não seguiu o schema
      (então ``text`` é o texto bruto e não há citações).
    """

    text: str
    cited_articles: list[str] = []
    retrieved_ids: list[str] = []
    confidence: float | None = None
    usage: Usage
    latency_ms: int = 0
    parse_error: str | None = None


class Strategy(Protocol):
    """Contrato de toda estratégia: um nome estável e ``answer``."""

    name: str

    def answer(self, question: str) -> Answer: ...


_CITATION = re.compile(
    r"^\s*(?P<law>[a-z0-9_]+)\s*:\s*art\s*:\s*(?P<num>\d+)\s*[º°o]?(?:\s*-\s*(?P<suf>[a-z]{1,3}))?\s*$",
    re.IGNORECASE,
)


def normalize_citations(raw: Iterable[str], laws: Collection[str]) -> list[str]:
    """Citações válidas de uma lista escrita pelo modelo.

    Aceita variações de caixa e espaço (``LGPD:art:7º``), descarta o que não
    segue ``<lei>:art:<número>[-<letras>]`` ou cita norma fora do corpus, e
    remove repetições mantendo a ordem.
    """
    known = set(laws)
    seen: dict[str, None] = {}
    for item in raw:
        m = _CITATION.match(item)
        if m is None or m["law"].lower() not in known:
            continue
        suffix = f"-{m['suf'].upper()}" if m["suf"] else ""
        seen.setdefault(f"{m['law'].lower()}:art:{int(m['num'])}{suffix}", None)
    return list(seen)


def load_prompt(name: str, prompts_dir: Path = Path("prompts")) -> str:
    """Ler ``prompts/<name>.md``. O nome carrega a versão (``baseline_v1``).

    :raises FileNotFoundError: prompt inexistente.
    """
    return (prompts_dir / f"{name}.md").read_text(encoding="utf-8").strip()


__all__ = ["Answer", "Strategy", "load_prompt", "normalize_citations"]
