# -----------------------------------------------------------------------------
# File:     src/lab/eval/dataset.py
# Purpose:  Schema and validator of the evaluation question set (SPEC 7.1, 7.2).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Conjunto de perguntas de avaliação.

Uma linha de ``questions.jsonl`` é uma :class:`Question`. O validador garante o
que a métrica assume: ids únicos, gabarito coerente com a categoria e, quando o
corpus está disponível, gabarito que aponta para artigos que existem e estão
em vigor. Um conjunto mal formado é recusado antes de qualquer chamada paga.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from lab.ingest.export import ArticleRecord

Category = Literal["factual", "multi_artigo", "multi_lei", "numerica", "sem_resposta"]
CATEGORIES: tuple[Category, ...] = (
    "factual",
    "multi_artigo",
    "multi_lei",
    "numerica",
    "sem_resposta",
)
# Distribuição do conjunto oficial (SPEC 7.2): 80 perguntas.
TARGET_DISTRIBUTION: dict[Category, int] = {
    "factual": 25,
    "multi_artigo": 15,
    "multi_lei": 15,
    "numerica": 10,
    "sem_resposta": 15,
}

_ARTICLE_ID = re.compile(r"^[a-z0-9_]+:art:\d+(?:-[A-Z]{1,3})?$")


class DatasetError(ValueError):
    """Conjunto de avaliação inválido; a mensagem indica arquivo, linha e motivo."""


class Question(BaseModel):
    """Uma pergunta de avaliação com o gabarito.

    - ``id``: ``q001``, ``q002``... estável; runs antigos referenciam por ele.
    - ``reference_answer``: resposta esperada, usada pelo juiz.
    - ``gold_articles``: artigos que sustentam a resposta; vazio só em ``sem_resposta``.
    - ``laws``: normas envolvidas (``multi_lei`` exige ao menos duas).
    - ``origin``: ``manual`` (padrão), ``synthetic_reviewed`` (gerada e revisada)
      ou ``dev_draft`` (rascunho de desenvolvimento, fora do conjunto congelado).
    """

    id: str = Field(pattern=r"^q\d{3,}$")
    question: str = Field(min_length=10)
    reference_answer: str = Field(min_length=1)
    gold_articles: list[str] = []
    category: Category
    laws: list[str] = Field(min_length=1)
    origin: Literal["manual", "synthetic_reviewed", "dev_draft"] = "manual"

    @field_validator("gold_articles")
    @classmethod
    def _valid_article_ids(cls, value: list[str]) -> list[str]:
        bad = [a for a in value if not _ARTICLE_ID.match(a)]
        if bad:
            raise ValueError(f"gold_articles fora do formato <lei>:art:<n>: {bad}")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.category == "sem_resposta" and self.gold_articles:
            raise ValueError("sem_resposta não pode ter gold_articles")
        if self.category != "sem_resposta" and not self.gold_articles:
            raise ValueError(f"{self.category} exige ao menos um gold_articles")
        if self.category == "multi_lei" and len(set(self.laws)) < 2:
            raise ValueError("multi_lei exige ao menos duas normas em laws")
        if self.category == "multi_artigo" and len(self.gold_articles) < 2:
            raise ValueError("multi_artigo exige ao menos dois gold_articles")
        gold_laws = {a.split(":")[0] for a in self.gold_articles}
        if not gold_laws <= set(self.laws):
            raise ValueError("gold_articles cita norma que não está em laws")
        return self


def load_questions(path: Path) -> list[Question]:
    """Ler e validar ``questions.jsonl``.

    :raises DatasetError: JSON inválido, pergunta fora do schema ou id repetido.
    """
    questions: list[Question] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                questions.append(Question.model_validate_json(line))
            except ValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(p) for p in first["loc"])
                raise DatasetError(
                    f"{path.name}:{number}: {where + ': ' if where else ''}{first['msg']}"
                ) from exc
    counts = Counter(q.id for q in questions)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if repeated:
        raise DatasetError(f"{path.name}: ids repetidos: {', '.join(repeated)}")
    if not questions:
        raise DatasetError(f"{path.name}: nenhuma pergunta")
    return questions


def check_against_corpus(questions: Sequence[Question], records: Sequence[ArticleRecord]) -> None:
    """Conferir o gabarito contra o corpus: artigos existem e estão vigentes.

    :raises DatasetError: lista todos os problemas de uma vez.
    """
    status = {r.id: r.status for r in records}
    laws = {r.law for r in records}
    problems: list[str] = []
    for q in questions:
        for law in q.laws:
            if law not in laws:
                problems.append(f"{q.id}: norma {law!r} não está no corpus")
        for article in q.gold_articles:
            if article not in status:
                problems.append(f"{q.id}: {article} não existe no corpus")
            elif status[article] != "vigente":
                problems.append(f"{q.id}: {article} está {status[article]}")
    if problems:
        raise DatasetError("gabarito inconsistente com o corpus:\n- " + "\n- ".join(problems))


def distribution(questions: Sequence[Question]) -> dict[Category, int]:
    """Quantidade de perguntas por categoria (todas as categorias, mesmo com zero)."""
    counts = Counter(q.category for q in questions)
    return {c: counts.get(c, 0) for c in CATEGORIES}


def distribution_gaps(questions: Sequence[Question]) -> dict[Category, int]:
    """Quanto falta (positivo) ou sobra (negativo) por categoria frente ao conjunto oficial."""
    have = distribution(questions)
    return {c: TARGET_DISTRIBUTION[c] - have[c] for c in CATEGORIES}


__all__ = [
    "CATEGORIES",
    "TARGET_DISTRIBUTION",
    "Category",
    "DatasetError",
    "Question",
    "check_against_corpus",
    "distribution",
    "distribution_gaps",
    "load_questions",
]
