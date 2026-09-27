# -----------------------------------------------------------------------------
# File:     tests/test_eval_annotation.py
# Purpose:  Tests for blind annotation sampling and human × judge agreement.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from collections import Counter
from pathlib import Path

import pytest

from lab.eval.annotation import (
    AnnotationError,
    AnnotationItem,
    agreement,
    load_judge_scores,
    read_items,
    sample_for_annotation,
    write_items,
)
from lab.eval.dataset import Question
from lab.eval.judge import Verdict
from lab.eval.runner import QuestionResult
from lab.llm.costs import Usage
from lab.strategies.base import Answer

CATEGORIES = ["factual"] * 25 + ["numerica"] * 10 + ["sem_resposta"] * 5


def question(n: int, category: str) -> Question:
    unanswerable = category == "sem_resposta"
    return Question(
        id=f"q{n:03d}",
        question=f"Pergunta número {n} sobre a lei?",
        reference_answer=f"Referência {n}.",
        gold_articles=[] if unanswerable else ["lgpd:art:1"],
        category=category,
        laws=["lgpd"],
    )


def result(q: Question, score: int | None = 2) -> QuestionResult:
    verdict = Verdict(score=score, rationale="r") if score is not None else Verdict(error="x")
    return QuestionResult(
        question_id=q.id,
        category=q.category,
        answer=Answer(text=f"Resposta a {q.id}.", usage=Usage()),
        verdict=verdict,
    )


QUESTIONS = [question(i + 1, c) for i, c in enumerate(CATEGORIES)]
RESULTS = [result(q) for q in QUESTIONS]


# -- amostragem ---------------------------------------------------------------


def test_sample_is_blind_and_carries_what_the_annotator_needs():
    items = sample_for_annotation(QUESTIONS, RESULTS, n=20)
    assert len(items) == 20
    data = items[0].model_dump()
    assert set(data) == {
        "question_id", "category", "question", "reference_answer", "answer", "human_score", "note",
    }  # fmt: skip
    assert all(i.human_score is None for i in items)  # nada da nota do juiz


def test_sample_is_proportional_and_covers_every_category():
    counts = Counter(i.category for i in sample_for_annotation(QUESTIONS, RESULTS, n=20))
    assert counts == {"factual": 13, "numerica": 5, "sem_resposta": 2}  # ~25:10:5 de 40


def test_sample_is_reproducible_and_depends_on_the_seed():
    a = [i.question_id for i in sample_for_annotation(QUESTIONS, RESULTS, n=20, seed=1)]
    b = [i.question_id for i in sample_for_annotation(QUESTIONS, RESULTS, n=20, seed=1)]
    c = [i.question_id for i in sample_for_annotation(QUESTIONS, RESULTS, n=20, seed=2)]
    assert a == b and a != c and a == sorted(a)


def test_sample_skips_answers_the_judge_could_not_grade():
    results = [result(q, score=None if q.category == "numerica" else 2) for q in QUESTIONS]
    items = sample_for_annotation(QUESTIONS, results, n=20)
    assert all(i.category != "numerica" for i in items)


def test_small_runs_return_everything_available():
    items = sample_for_annotation(QUESTIONS[:5], RESULTS[:5], n=20)
    assert len(items) == 5


# -- arquivo ------------------------------------------------------------------


def test_write_and_read_round_trip(tmp_path: Path):
    items = sample_for_annotation(QUESTIONS, RESULTS, n=4)
    path = tmp_path / "eval" / "ann.jsonl"
    write_items(path, items)
    assert read_items(path) == items


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"question_id": "q1"}', r"ann\.jsonl:1"),
        ("{nem json", r"ann\.jsonl:1"),
    ],
)
def test_invalid_annotation_line(tmp_path: Path, line: str, message: str):
    path = tmp_path / "ann.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match=message):
        read_items(path)


def test_human_score_outside_the_rubric_is_rejected(tmp_path: Path):
    item = sample_for_annotation(QUESTIONS, RESULTS, n=1)[0].model_dump()
    item["human_score"] = 3
    path = tmp_path / "ann.jsonl"
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(AnnotationError):
        read_items(path)


def test_repeated_ids_are_rejected(tmp_path: Path):
    item = sample_for_annotation(QUESTIONS, RESULTS, n=1)[0]
    path = tmp_path / "ann.jsonl"
    write_items(path, [item, item])
    with pytest.raises(AnnotationError, match="repetidos"):
        read_items(path)


def test_load_judge_scores_from_a_run(tmp_path: Path):
    lines = [result(QUESTIONS[0], 1), result(QUESTIONS[1], None)]
    (tmp_path / "answers.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in lines) + "\n", encoding="utf-8"
    )
    assert load_judge_scores(tmp_path) == {"q001": 1}


# -- concordância -------------------------------------------------------------


def item(qid: str, human: int | None) -> AnnotationItem:
    return AnnotationItem(
        question_id=qid,
        category="factual",
        question="?",
        reference_answer="r",
        answer="a",
        human_score=human,  # type: ignore[arg-type]
    )


def test_agreement_counts_exact_within_one_and_confusion():
    items = [item("q1", 2), item("q2", 0), item("q3", 1), item("q4", 0)]
    report = agreement(items, {"q1": 2, "q2": 0, "q3": 2, "q4": 2})
    assert (report.compared, report.pending) == (4, 0)
    assert report.exact == 0.5 and report.within_one == 0.75
    assert report.confusion == [[1, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert report.disagreements == ["q3", "q4"]
    assert not report.accepted


def test_acceptance_needs_80_percent_and_no_pending():
    scores = {f"q{i}": 2 for i in range(10)}
    good = [item(f"q{i}", 2) for i in range(8)] + [item("q8", 1), item("q9", 1)]
    assert agreement(good, scores).accepted  # 8/10 = 80%
    pending = [*good[:9], item("q9", None)]
    report = agreement(pending, scores)
    assert report.pending == 1 and not report.accepted


def test_nothing_annotated_yet():
    report = agreement([item("q1", None)], {"q1": 2})
    assert report.compared == 0 and report.exact is None and not report.accepted


def test_annotation_for_a_question_the_run_did_not_grade():
    with pytest.raises(AnnotationError, match="q9"):
        agreement([item("q9", 1)], {"q1": 2})
