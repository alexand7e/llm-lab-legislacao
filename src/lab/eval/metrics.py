# -----------------------------------------------------------------------------
# File:     src/lab/eval/metrics.py
# Purpose:  Deterministic metrics: citation, recall@k, MRR, abstention, aggregates.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Métricas determinísticas (SPEC 7.3).

Tudo aqui é função pura de ``(Answer, Question)``: sem rede, sem modelo, o
mesmo resultado sempre. As métricas dependentes de juiz (correção, fidelidade)
vivem em outro módulo.

Métrica indefinida é ``None``, não zero: recall@k de uma estratégia que não
recupera nada (baseline) não é 0, é "não se aplica"; a agregação ignora ``None``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel

from lab.eval.dataset import Question
from lab.strategies.base import Answer

# Sinais de que o modelo se absteve ("não sei", "fora do escopo"...). Heurística
# simples e explícita; o juiz LLM refina a abstenção nas perguntas sem resposta.
_ABSTENTION = re.compile(
    r"n[ãa]o\s+(?:sei|posso\s+(?:responder|informar|ajudar)|consigo|tenho\s+(?:certeza|informa)"
    r"|h[áa]\s+(?:previs[ãa]o|informa)|consta|encontr(?:ei|o)|est[áa]\s+(?:prevista|dispon)"
    r"|[ée]\s+poss[íi]vel\s+(?:responder|informar|determinar))"
    r"|fora\s+d[oa]\s+(?:escopo|corpus|[âa]mbito)|desculp|sem\s+informa[çc][õo]es\s+suficientes"
    r"|n[ãa]o\s+(?:disp[õo]e|disponho)",
    re.IGNORECASE,
)
_LOW_CONFIDENCE = 0.2


def is_abstention(answer: Answer) -> bool:
    """A resposta se recusa a responder?

    Sim se o texto tem sinal de recusa, ou se não cita nada e a confiança é baixa.
    """
    if _ABSTENTION.search(answer.text):
        return True
    return (
        not answer.cited_articles
        and answer.confidence is not None
        and answer.confidence <= _LOW_CONFIDENCE
    )


def citation_precision(cited: Sequence[str], gold: Sequence[str]) -> float | None:
    """Fração dos artigos citados que estão no gabarito; ``None`` se não citou nada."""
    if not cited:
        return None
    return len(set(cited) & set(gold)) / len(set(cited))


def citation_recall(cited: Sequence[str], gold: Sequence[str]) -> float | None:
    """Fração do gabarito que foi citada; ``None`` se não há gabarito."""
    if not gold:
        return None
    return len(set(cited) & set(gold)) / len(set(gold))


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float | None:
    """Fração do gabarito entre os ``k`` primeiros recuperados.

    ``None`` sem gabarito ou sem recuperação (estratégia que não recupera).
    """
    if not gold or not retrieved:
        return None
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float | None:
    """1/posição do primeiro artigo do gabarito recuperado (0 se nenhum apareceu)."""
    if not gold or not retrieved:
        return None
    for position, article in enumerate(retrieved, start=1):
        if article in gold:
            return 1 / position
    return 0.0


class Scores(BaseModel):
    """Métricas de uma resposta. ``None`` = não se aplica a esta pergunta/estratégia."""

    citation_hit: float | None = None  # citou ao menos um artigo do gabarito
    citation_precision: float | None = None
    citation_recall: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    abstained: float
    abstention_correct: float | None = None  # só em sem_resposta
    wrongful_refusal: float | None = None  # só nas demais


def score(answer: Answer, question: Question) -> Scores:
    """Métricas determinísticas de uma resposta contra o gabarito."""
    gold, cited = question.gold_articles, answer.cited_articles
    abstained = is_abstention(answer)
    unanswerable = question.category == "sem_resposta"
    return Scores(
        citation_hit=None if unanswerable else float(bool(set(cited) & set(gold))),
        citation_precision=None if unanswerable else citation_precision(cited, gold),
        citation_recall=citation_recall(cited, gold),
        recall_at_5=recall_at_k(answer.retrieved_ids, gold, 5),
        recall_at_10=recall_at_k(answer.retrieved_ids, gold, 10),
        mrr=reciprocal_rank(answer.retrieved_ids, gold),
        abstained=float(abstained),
        abstention_correct=float(abstained) if unanswerable else None,
        wrongful_refusal=None if unanswerable else float(abstained),
    )


# -- agregação ----------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Percentil ``q`` (0 a 100) por interpolação linear; ``nan`` se vazio."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Aggregate(BaseModel):
    """Médias de um grupo de perguntas (``n``), ignorando métricas ``None``."""

    n: int
    means: dict[str, float]
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None


def aggregate(rows: Sequence[tuple[Question, Answer, Scores]]) -> Aggregate:
    """Agregar métricas, custo e latência de um conjunto de respostas."""
    values: dict[str, list[float]] = defaultdict(list)
    for _, _, scores in rows:
        for name, value in scores.model_dump().items():
            if value is not None:
                values[name].append(value)
    latencies = [float(a.latency_ms) for _, a, _ in rows if a.latency_ms > 0]  # 0 = veio do cache
    return Aggregate(
        n=len(rows),
        means={name: mean(v) for name, v in sorted(values.items())},
        cost_usd=sum(a.usage.cost_usd for _, a, _ in rows),
        prompt_tokens=sum(a.usage.prompt_tokens for _, a, _ in rows),
        completion_tokens=sum(a.usage.completion_tokens for _, a, _ in rows),
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
    )


def aggregate_by_category(
    rows: Sequence[tuple[Question, Answer, Scores]],
) -> dict[str, Aggregate]:
    """Um :class:`Aggregate` por categoria presente em ``rows``."""
    groups: dict[str, list[tuple[Question, Answer, Scores]]] = defaultdict(list)
    for row in rows:
        groups[row[0].category].append(row)
    return {category: aggregate(group) for category, group in sorted(groups.items())}


__all__ = [
    "Aggregate",
    "Scores",
    "aggregate",
    "aggregate_by_category",
    "citation_precision",
    "citation_recall",
    "is_abstention",
    "percentile",
    "recall_at_k",
    "reciprocal_rank",
    "score",
]
