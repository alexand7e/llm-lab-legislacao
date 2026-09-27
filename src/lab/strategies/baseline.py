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

from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel

from lab.ingest.sources import LawSource
from lab.llm.client import ChatResult, LLMClient, json_schema_format, parse_structured
from lab.strategies.base import (
    Answer,
    Message,
    StreamChunk,
    StreamItem,
    load_prompt,
    normalize_citations,
)
from lab.strategies.streaming import AnswerExtractor

# Quantas mensagens anteriores da conversa vão ao modelo (custo e contexto limitados).
MAX_HISTORY_MESSAGES = 8


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

    def _messages(self, question: str, history: Sequence[Message]) -> list[dict[str, str]]:
        """System + as últimas mensagens da conversa + a pergunta atual."""
        recent = [m for m in history if m.content.strip()][-MAX_HISTORY_MESSAGES:]
        return [
            {"role": "system", "content": self._system},
            *({"role": m.role, "content": m.content} for m in recent),
            {"role": "user", "content": question},
        ]

    def _build_answer(
        self, value: BaselineOutput | None, error: str | None, result: ChatResult
    ) -> Answer:
        """Answer a partir da saída validada; sem ela, o texto bruto vira a resposta."""
        if value is None:
            return Answer(
                text=result.text.strip(),
                usage=result.usage,
                latency_ms=result.latency_ms,
                parse_error=error,
            )
        return Answer(
            text=value.answer.strip(),
            cited_articles=normalize_citations(value.cited_articles, self._law_ids),
            confidence=value.confidence,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )

    def answer(self, question: str, history: Sequence[Message] = ()) -> Answer:
        parsed = self._client.chat_parsed(
            self._role,
            self._messages(question, history),
            BaselineOutput,
            temperature=0,
            max_tokens=self._max_tokens,
        )
        return self._build_answer(parsed.value, parsed.error, parsed.result)

    def answer_stream(self, question: str, history: Sequence[Message] = ()) -> Iterator[StreamItem]:
        """Como :meth:`answer`, mas emite a resposta enquanto o modelo a gera.

        Produz ``StreamChunk`` nos canais ``reasoning`` (raciocínio do modelo),
        ``raw`` (o JSON bruto) e ``text`` (só o campo ``answer``, já decodificado),
        e por fim a :class:`Answer` completa, igual à de :meth:`answer`.
        """
        extractor = AnswerExtractor()
        stream = self._client.chat_stream(
            self._role,
            self._messages(question, history),
            response_format=json_schema_format(BaselineOutput),
            temperature=0,
            max_tokens=self._max_tokens,
        )
        for item in stream:
            if isinstance(item, ChatResult):
                value, error = parse_structured(item.text, BaselineOutput)
                yield self._build_answer(value, error, item)
                return
            if item.kind == "reasoning":
                yield StreamChunk("reasoning", item.text)
                continue
            yield StreamChunk("raw", item.text)
            text = extractor.feed(item.text)
            if text:
                yield StreamChunk("text", text)


__all__ = ["BaselineOutput", "BaselineStrategy"]
