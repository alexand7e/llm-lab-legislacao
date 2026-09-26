# -----------------------------------------------------------------------------
# File:     src/lab/strategies/baseline.py
# Purpose:  Baseline strategy: the model answers from memory, without retrieval.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Estratégia ``baseline``: o modelo responde só com o que sabe (zero-shot).

É o ponto de partida da comparação (SPEC 10, M3): mede quanto o modelo sabe
de legislação sem recuperação. Sem trechos de lei no prompt, os artigos
citados vêm da memória do modelo, e é esperado que ele erre números e prazos.

A resposta é pedida em JSON (``answer``, ``cited_articles``, ``confidence``).
Se o modelo não seguir o schema, o texto bruto vira a resposta, sem citações,
e ``Answer.parse_error`` registra o motivo: o custo da chamada não se perde.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from lab.ingest.sources import LawSource
from lab.llm.client import LLMClient
from lab.strategies.base import Answer, load_prompt, normalize_citations


class BaselineOutput(BaseModel):
    """Schema JSON que o modelo deve devolver."""

    answer: str
    cited_articles: list[str] = []
    confidence: float | None = None


class BaselineStrategy:
    """Resposta direta do modelo, sem recuperação.

    :param client: cliente LLM já configurado (cache, limite de custo).
    :param laws: normas do corpus, listadas no prompt e usadas para validar citações.
    :param prompt: nome do prompt em ``prompts/`` (a versão faz parte do nome).
    :param role: papel de modelo em ``models.yaml``.
    :param max_tokens: teto de tokens da resposta; modelos com raciocínio gastam
        boa parte dele pensando, então o padrão é generoso.
    """

    name = "baseline"

    def __init__(
        self,
        client: LLMClient,
        laws: Sequence[LawSource],
        *,
        prompt: str = "baseline_v1",
        role: str = "generator",
        max_tokens: int = 4000,
        prompts_dir: Path = Path("prompts"),
    ) -> None:
        self._client = client
        self._law_ids = [law.id for law in laws]
        self._role = role
        self._max_tokens = max_tokens
        listing = "\n".join(f"- {law.name}, {law.number} ({law.id})" for law in laws)
        self._system = load_prompt(prompt, prompts_dir).replace("{laws}", listing)

    def answer(self, question: str) -> Answer:
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": question},
        ]
        parsed = self._client.chat_parsed(
            self._role,
            messages,
            BaselineOutput,
            temperature=0,
            max_tokens=self._max_tokens,
        )
        result = parsed.result
        if parsed.value is None:
            return Answer(
                text=result.text.strip(),
                usage=result.usage,
                latency_ms=result.latency_ms,
                parse_error=parsed.error,
            )
        return Answer(
            text=parsed.value.answer.strip(),
            cited_articles=normalize_citations(parsed.value.cited_articles, self._law_ids),
            confidence=parsed.value.confidence,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )


__all__ = ["BaselineOutput", "BaselineStrategy"]
