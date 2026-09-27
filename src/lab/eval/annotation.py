# -----------------------------------------------------------------------------
# File:     src/lab/eval/annotation.py
# Purpose:  Blind human annotation of answers and agreement with the LLM judge.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Validação do juiz contra anotações humanas (SPEC 7.3, aceite do M2).

Fluxo:

1. :func:`sample_for_annotation` sorteia respostas de uma run (semente fixa,
   proporcional às categorias) e gera itens **sem a nota do juiz**: a anotação é
   cega, para a nota humana não ser influenciada pela do modelo.
2. Uma pessoa preenche ``human_score`` (0, 1 ou 2) em cada linha, com a mesma
   rubrica do juiz (``prompts/judge_v1.md``).
3. :func:`agreement` cruza as notas humanas com as do juiz, lidas da run, e
   calcula concordância exata, concordância ±1 e a matriz de confusão.

O aceite do M2 é concordância exata ≥ 80%; abaixo disso, a rubrica é revisada
antes de qualquer comparação entre estratégias.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from lab.eval.dataset import Question
from lab.eval.runner import QuestionResult

ACCEPTANCE = 0.80
Score = Literal[0, 1, 2]


class AnnotationError(ValueError):
    """Arquivo de anotação ou run inconsistente."""


class AnnotationItem(BaseModel):
    """Uma resposta a anotar. ``human_score`` começa vazio."""

    question_id: str
    category: str
    question: str
    reference_answer: str
    answer: str
    human_score: Score | None = None
    note: str = ""


def sample_for_annotation(
    questions: Sequence[Question],
    results: Sequence[QuestionResult],
    *,
    n: int = 20,
    seed: int = 1,
) -> list[AnnotationItem]:
    """Sortear até ``n`` respostas julgadas, proporcionalmente às categorias.

    Só entram respostas que o juiz conseguiu avaliar (sem nota não há o que
    comparar). Cada categoria presente recebe ao menos uma vaga.
    """
    by_id = {q.id: q for q in questions}
    judged = [
        r
        for r in results
        if r.answer is not None and r.verdict is not None and r.verdict.score is not None
    ]
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in judged:
        groups[r.category].append(r)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    total = len(judged)
    quota = {c: min(len(g), max(1, round(n * len(g) / total))) for c, g in groups.items()}
    while sum(quota.values()) > n:  # arredondamento passou do alvo: tira da maior
        biggest = max((c for c in quota if quota[c] > 1), key=lambda c: quota[c], default=None)
        if biggest is None:
            break
        quota[biggest] -= 1
    while sum(quota.values()) < min(n, total):  # faltou: dá a quem ainda tem sobra
        spare = [c for c in quota if quota[c] < len(groups[c])]
        quota[max(spare, key=lambda c: len(groups[c]) - quota[c])] += 1

    picked = [r for c in sorted(groups) for r in groups[c][: quota[c]]]
    items: list[AnnotationItem] = []
    for r in sorted(picked, key=lambda r: r.question_id):
        q = by_id[r.question_id]
        assert r.answer is not None
        items.append(
            AnnotationItem(
                question_id=r.question_id,
                category=r.category,
                question=q.question,
                reference_answer=q.reference_answer,
                answer=r.answer.text,
            )
        )
    return items


def write_items(path: Path, items: Sequence[AnnotationItem]) -> None:
    """Gravar o arquivo de anotação (JSONL, uma resposta por linha)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[AnnotationItem]:
    """Ler o arquivo de anotação.

    :raises AnnotationError: linha inválida (aponta arquivo e linha) ou id repetido.
    """
    items: list[AnnotationItem] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                items.append(AnnotationItem.model_validate_json(line))
            except ValidationError as exc:
                raise AnnotationError(f"{path.name}:{number}: {exc.errors()[0]['msg']}") from exc
    ids = [i.question_id for i in items]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise AnnotationError(f"{path.name}: ids repetidos: {', '.join(repeated)}")
    return items


def load_judge_scores(run_dir: Path) -> dict[str, int]:
    """Notas do juiz por pergunta, lidas do ``answers.jsonl`` da run."""
    scores: dict[str, int] = {}
    with (run_dir / "answers.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            result = QuestionResult.model_validate_json(line)
            if result.verdict is not None and result.verdict.score is not None:
                scores[result.question_id] = result.verdict.score
    return scores


class AgreementReport(BaseModel):
    """Concordância entre notas humanas e do juiz."""

    compared: int
    pending: int  # itens ainda sem nota humana
    exact: float | None  # fração com a mesma nota
    within_one: float | None  # fração com diferença de no máximo 1
    confusion: list[list[int]]  # [humano][juiz], notas 0..2
    disagreements: list[str]  # ids onde as notas diferem

    @property
    def accepted(self) -> bool:
        """Aceite do M2: todas anotadas e concordância exata ≥ 80%."""
        return self.pending == 0 and self.exact is not None and self.exact >= ACCEPTANCE


def agreement(items: Sequence[AnnotationItem], judge_scores: dict[str, int]) -> AgreementReport:
    """Comparar as notas humanas com as do juiz.

    :raises AnnotationError: item anotado sem nota do juiz na run (run errada?).
    """
    confusion = [[0, 0, 0] for _ in range(3)]
    exact = within = 0
    disagreements: list[str] = []
    pending = 0
    for item in items:
        if item.human_score is None:
            pending += 1
            continue
        if item.question_id not in judge_scores:
            raise AnnotationError(f"{item.question_id}: sem nota do juiz nesta run")
        human, judge = item.human_score, judge_scores[item.question_id]
        confusion[human][judge] += 1
        exact += human == judge
        within += abs(human - judge) <= 1
        if human != judge:
            disagreements.append(item.question_id)
    compared = len(items) - pending
    return AgreementReport(
        compared=compared,
        pending=pending,
        exact=exact / compared if compared else None,
        within_one=within / compared if compared else None,
        confusion=confusion,
        disagreements=disagreements,
    )


__all__ = [
    "ACCEPTANCE",
    "AgreementReport",
    "AnnotationError",
    "AnnotationItem",
    "agreement",
    "load_judge_scores",
    "read_items",
    "sample_for_annotation",
    "write_items",
]
