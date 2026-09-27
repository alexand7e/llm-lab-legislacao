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
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

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


class Message(BaseModel):
    """Uma mensagem anterior da conversa (contexto para a pergunta atual)."""

    role: Literal["user", "assistant"]
    content: str


class Strategy(Protocol):
    """Contrato de toda estratégia: um nome estável e ``answer``.

    ``history`` são as mensagens anteriores da conversa, da mais antiga para a
    mais recente, sem a pergunta atual. A avaliação (``lab eval``) chama com
    pergunta única, sem histórico; o chat da interface passa a conversa, para
    perguntas como "quem fez essa lei?" serem entendidas no contexto.
    """

    name: str

    def answer(self, question: str, history: Sequence[Message] = ()) -> Answer: ...


@dataclass(frozen=True)
class StreamChunk:
    """Pedaço de uma resposta em streaming.

    ``text`` é o texto da resposta em si; ``reasoning`` é o raciocínio do modelo,
    quando ele o emite; ``raw`` é a saída bruta do modelo (o JSON estruturado).
    """

    channel: Literal["text", "reasoning", "raw"]
    text: str


StreamItem = StreamChunk | Answer


@runtime_checkable
class StreamingStrategy(Strategy, Protocol):
    """Estratégia que também responde em streaming.

    Produz :class:`StreamChunk` enquanto o modelo gera e, por último, a
    :class:`Answer` completa (com citações, custo e latência).
    """

    def answer_stream(
        self, question: str, history: Sequence[Message] = ()
    ) -> Iterator[StreamItem]: ...


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


_FIELD_LINE = re.compile(
    r"^\s*[-*>]*\s*[`*_\"']*(?P<field>answer|cited_articles|confidence)[`*_\"']*"
    r"\s*[:=]\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_ARTICLE_ID = re.compile(
    r"[a-z0-9_]+\s*:\s*art\s*:\s*\d+[º°o]?(?:\s*-\s*[a-z]{1,3})?", re.IGNORECASE
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def salvage_fields(text: str, laws: Collection[str]) -> tuple[str, list[str], float | None]:
    """Recuperar resposta, citações e confiança de uma saída fora do formato JSON.

    Alguns modelos ignoram o ``response_format`` e escrevem os campos no fim do
    texto (``cited_articles: ["cdc:art:26"]``). Só linhas rotuladas com o nome do
    campo contam: um artigo mencionado no meio da resposta não vira citação.

    :return: ``(texto sem as linhas de campo, citações normalizadas, confiança)``.
    """
    kept: list[str] = []
    cited: list[str] = []
    confidence: float | None = None
    for line in text.splitlines():
        m = _FIELD_LINE.match(line)
        if m is None:
            kept.append(line)
            continue
        field, value = m["field"].lower(), m["value"]
        if field == "cited_articles":
            cited += _ARTICLE_ID.findall(value)
        elif field == "confidence":
            number = _NUMBER.search(value)
            if number:
                confidence = min(1.0, max(0.0, float(number.group().replace(",", "."))))
        else:  # "answer: ..." — o rótulo sai, o conteúdo fica
            kept.append(value.strip().strip('"'))
    clean = "\n".join(kept).strip()
    return clean, normalize_citations(cited, laws), confidence


def load_prompt(name: str, prompts_dir: Path = Path("prompts")) -> str:
    """Ler ``prompts/<name>.md``. O nome carrega a versão (``baseline_v1``).

    :raises FileNotFoundError: prompt inexistente.
    """
    return (prompts_dir / f"{name}.md").read_text(encoding="utf-8").strip()


__all__ = [
    "Answer",
    "Message",
    "StreamChunk",
    "StreamItem",
    "Strategy",
    "StreamingStrategy",
    "load_prompt",
    "normalize_citations",
    "salvage_fields",
]
