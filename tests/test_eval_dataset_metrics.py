# -----------------------------------------------------------------------------
# File:     tests/test_eval_dataset_metrics.py
# Purpose:  Tests for the evaluation set schema/validator and deterministic metrics.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
import math
from datetime import date
from pathlib import Path

import pytest

from lab.eval.dataset import (
    CATEGORIES,
    DatasetError,
    Question,
    check_against_corpus,
    distribution,
    distribution_gaps,
    load_questions,
)
from lab.eval.metrics import (
    aggregate,
    aggregate_by_category,
    citation_precision,
    citation_recall,
    is_abstention,
    percentile,
    recall_at_k,
    reciprocal_rank,
    score,
)
from lab.ingest.export import ArticleRecord, read_jsonl
from lab.llm.costs import Usage
from lab.strategies.base import Answer

ROOT = Path(__file__).parent.parent


def question(**over) -> Question:
    base = {
        "id": "q001",
        "question": "Qual o prazo previsto na lei para isso?",
        "reference_answer": "Imediatamente.",
        "gold_articles": ["lgpd:art:19"],
        "category": "factual",
        "laws": ["lgpd"],
    }
    return Question(**{**base, **over})


def answer(**over) -> Answer:
    base = {
        "text": "Em 15 dias.",
        "usage": Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.5),
    }
    return Answer(**{**base, **over})


# -- schema -------------------------------------------------------------------


def test_valid_question_defaults_to_manual_origin():
    assert question().origin == "manual"


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"id": "1"}, "id"),
        ({"question": "curta"}, "question"),
        ({"gold_articles": ["art. 7"]}, "gold_articles"),
        ({"gold_articles": []}, "exige ao menos um"),
        ({"category": "sem_resposta"}, "sem_resposta não pode"),
        ({"category": "multi_lei"}, "ao menos duas normas"),
        ({"category": "multi_artigo"}, "ao menos dois"),
        ({"gold_articles": ["cdc:art:6"]}, "não está em laws"),
        ({"category": "inexistente"}, "category"),
    ],
)
def test_invalid_question_is_rejected(over: dict, message: str):
    with pytest.raises(ValueError, match=message):
        question(**over)


def test_unanswerable_question_has_no_gold():
    q = question(category="sem_resposta", gold_articles=[])
    assert q.gold_articles == []


def write(path: Path, *rows: dict | str) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), "utf-8")
    return path


def test_load_reports_file_line_and_field(tmp_path: Path):
    good = question().model_dump()
    bad = {**good, "id": "q002", "gold_articles": ["errado"]}
    with pytest.raises(DatasetError, match=r"q\.jsonl:2: gold_articles"):
        load_questions(write(tmp_path / "q.jsonl", good, bad))


def test_load_rejects_duplicates_and_empty_files(tmp_path: Path):
    good = question().model_dump()
    with pytest.raises(DatasetError, match="q001"):
        load_questions(write(tmp_path / "a.jsonl", good, good))
    with pytest.raises(DatasetError, match="nenhuma pergunta"):
        load_questions(write(tmp_path / "b.jsonl", ""))


def test_load_rejects_broken_json(tmp_path: Path):
    with pytest.raises(DatasetError, match=r"c\.jsonl:1"):
        load_questions(write(tmp_path / "c.jsonl", "{isto não é json"))


def record(article_id: str, status: str = "vigente") -> ArticleRecord:
    law, _, number = article_id.split(":")[0], None, article_id.split(":")[-1]
    return ArticleRecord(
        id=article_id, law=law, article=number, path=["Título I"], status=status, text="t",
        units=[], references=[], source_hash="sha256:x", collected_at=date(2026, 9, 26),
    )  # fmt: skip


def test_check_against_corpus_lists_every_problem():
    qs = [
        question(id="q001", gold_articles=["lgpd:art:19"]),
        question(id="q002", gold_articles=["lgpd:art:999"]),
        question(id="q003", gold_articles=["lgpd:art:20"]),
        question(id="q004", laws=["xyz"], gold_articles=["xyz:art:1"]),
    ]
    corpus = [record("lgpd:art:19"), record("lgpd:art:20", "vetado")]
    with pytest.raises(DatasetError) as info:
        check_against_corpus(qs, corpus)
    text = str(info.value)
    assert "q002: lgpd:art:999 não existe" in text
    assert "q003: lgpd:art:20 está vetado" in text
    assert "q004: norma 'xyz'" in text
    assert "q001" not in text


def test_distribution_and_gaps():
    qs = [question(id="q001"), question(id="q002", category="sem_resposta", gold_articles=[])]
    assert distribution(qs) == {**dict.fromkeys(CATEGORIES, 0), "factual": 1, "sem_resposta": 1}
    gaps = distribution_gaps(qs)
    assert gaps["factual"] == 24 and gaps["sem_resposta"] == 14 and sum(gaps.values()) == 78


def test_dev_set_is_valid_and_matches_the_corpus():
    questions = load_questions(ROOT / "data" / "eval" / "questions.dev.jsonl")
    check_against_corpus(questions, read_jsonl(ROOT / "data" / "processed" / "articles.jsonl"))
    assert {q.origin for q in questions} == {"dev_draft"}  # não é o conjunto congelado
    assert set(distribution(questions)) == set(CATEGORIES)
    assert all(count > 0 for count in distribution(questions).values())


# -- métricas -----------------------------------------------------------------


def test_citation_metrics():
    assert citation_precision(["a", "b"], ["a"]) == 0.5
    assert citation_precision([], ["a"]) is None
    assert citation_recall(["a"], ["a", "b"]) == 0.5
    assert citation_recall(["a"], []) is None


def test_retrieval_metrics():
    retrieved = ["x", "g1", "y", "g2"]
    assert recall_at_k(retrieved, ["g1", "g2"], 2) == 0.5
    assert recall_at_k(retrieved, ["g1", "g2"], 10) == 1.0
    assert reciprocal_rank(retrieved, ["g2", "g1"]) == 0.5
    assert reciprocal_rank(retrieved, ["nada"]) == 0.0
    assert recall_at_k([], ["g1"], 5) is None  # sem recuperação: não se aplica
    assert reciprocal_rank(retrieved, []) is None


@pytest.mark.parametrize(
    "text",
    [
        "Não sei responder a isso.",
        "Desculpe, mas não posso responder a essa pergunta.",
        "Essa informação não consta nas normas indicadas.",
        "Isso está fora do escopo das normas.",
    ],
)
def test_abstention_by_text(text: str):
    assert is_abstention(answer(text=text))


def test_abstention_by_low_confidence_without_citation():
    assert is_abstention(answer(text="Talvez.", confidence=0.1))
    assert not is_abstention(answer(text="Talvez.", confidence=0.1, cited_articles=["lgpd:art:1"]))
    assert not is_abstention(answer(text="Em 15 dias.", confidence=0.9))


def test_score_answerable_question():
    s = score(answer(cited_articles=["lgpd:art:19", "lgpd:art:17"]), question())
    assert (s.citation_hit, s.citation_precision, s.citation_recall) == (1.0, 0.5, 1.0)
    assert s.abstained == 0 and s.wrongful_refusal == 0 and s.abstention_correct is None
    assert s.mrr is None  # baseline não recupera


def test_score_wrong_citation_and_wrongful_refusal():
    s = score(answer(cited_articles=["lgpd:art:17"]), question())
    assert s.citation_hit == 0.0 and s.citation_recall == 0.0
    refusal = score(answer(text="Não sei."), question())
    assert refusal.wrongful_refusal == 1.0


def test_score_unanswerable_question():
    q = question(category="sem_resposta", gold_articles=[])
    ok = score(answer(text="Isso está fora do escopo."), q)
    assert ok.abstention_correct == 1.0 and ok.wrongful_refusal is None and ok.citation_hit is None
    bad = score(answer(text="A alíquota é 27,5%.", cited_articles=["lgpd:art:1"]), q)
    assert bad.abstention_correct == 0.0


def test_score_with_retrieval():
    s = score(answer(retrieved_ids=["x", "lgpd:art:19"]), question())
    assert s.recall_at_5 == 1.0 and s.mrr == 0.5


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([5], 95) == 5
    assert math.isnan(percentile([], 50))


def test_aggregate_ignores_none_and_sums_cost():
    q1, q2 = question(id="q001"), question(id="q002", category="sem_resposta", gold_articles=[])
    a1 = answer(cited_articles=["lgpd:art:19"], latency_ms=100)
    a2 = answer(text="Fora do escopo.", latency_ms=300)
    rows = [(q1, a1, score(a1, q1)), (q2, a2, score(a2, q2))]
    total = aggregate(rows)
    assert total.n == 2 and total.cost_usd == 1.0 and total.prompt_tokens == 20
    assert total.means["citation_hit"] == 1.0  # só q1 tem citation_hit
    assert total.means["abstention_correct"] == 1.0  # só q2
    assert "mrr" not in total.means  # nenhuma recuperação: nunca vira 0
    assert total.latency_p50_ms == 200
    by_category = aggregate_by_category(rows)
    assert set(by_category) == {"factual", "sem_resposta"}
    assert by_category["factual"].n == 1


def test_aggregate_latency_skips_cache_hits():
    q = question()
    a = answer(latency_ms=0)  # 0 = veio do cache
    assert aggregate([(q, a, score(a, q))]).latency_p50_ms is None
