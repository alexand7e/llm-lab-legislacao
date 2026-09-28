# -----------------------------------------------------------------------------
# File:     tests/test_strategies_rag.py
# Purpose:  Tests for the vector RAG strategy (in-memory Qdrant, fake LLM client).
# Author:   Alexandre
# Created:  2026-09-28
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from lab.eval.dataset import Question
from lab.eval.metrics import score
from lab.index.chunking import Chunk
from lab.index.store import VectorIndex
from lab.ingest.sources import LawSource
from lab.llm import Embeddings, Parsed
from lab.llm.client import ChatResult, StreamDelta
from lab.llm.costs import Usage
from lab.strategies.base import Answer, StreamChunk
from lab.strategies.baseline import BaselineOutput
from lab.strategies.factory import build_strategy
from lab.strategies.rag import RagStrategy, render_hits, unique_articles

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "prompts"
LAWS = [LawSource(id="lgpd", name="LGPD", number="Lei 13.709/2018", url="u")]


def chunk(article_id: str, unit: str = "") -> Chunk:
    return Chunk(
        id=article_id + unit, article_id=article_id, law="lgpd",
        article=article_id.split(":")[-1], path=["Capítulo III"],
        text=f"Texto do {article_id}{unit}.", embed_text=f"LGPD\nTexto do {article_id}{unit}.",
    )  # fmt: skip


def make_index() -> VectorIndex:
    index = VectorIndex(QdrantClient(":memory:"), "teste")
    chunks = [chunk("lgpd:art:19"), chunk("lgpd:art:18"), chunk("lgpd:art:41")]
    index.rebuild(chunks, [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    return index


class FakeClient:
    """Embedding fixo da pergunta e resposta estruturada do gerador."""

    def __init__(self, query_vector=(1.0, 0.05), output=None) -> None:
        self.query_vector = list(query_vector)
        self.output = output or BaselineOutput(
            answer="Imediatamente.", cited_articles=["lgpd:art:19"], confidence=0.9
        )
        self.messages: list[list[dict]] = []
        self.embedded: list[list[str]] = []

    def embed(self, role, texts, *, batch_size=64):
        self.embedded.append(list(texts))
        return Embeddings(vectors=[self.query_vector], usage=Usage(prompt_tokens=7, cost_usd=0.01))

    def chat_parsed(self, role, messages, schema, **params):
        self.messages.append(messages)
        result = ChatResult(
            text=self.output.model_dump_json(),
            usage=Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.1),
            latency_ms=50,
        )
        return Parsed(value=self.output, result=result)

    def chat_stream(self, role, messages, **params):
        self.messages.append(messages)
        text = self.output.model_dump_json()
        yield StreamDelta("content", text)
        yield ChatResult(
            text=text, usage=Usage(prompt_tokens=100, completion_tokens=20), latency_ms=5
        )


def rag(client: FakeClient, k: int = 2) -> RagStrategy:
    return RagStrategy(client, LAWS, make_index(), k=k, prompts_dir=PROMPTS)  # type: ignore[arg-type]


def test_answer_uses_retrieved_chunks_in_ranking_order():
    client = FakeClient()
    answer = rag(client).answer("Qual o prazo em formato simplificado?")
    assert answer.retrieved_ids == ["lgpd:art:19", "lgpd:art:18"]
    user = client.messages[0][-1]["content"]
    assert user.startswith("Trechos recuperados:")
    assert user.index("[lgpd:art:19]") < user.index("[lgpd:art:18]")
    assert "lgpd:art:41" not in user  # fora do top-k
    assert user.endswith("Pergunta: Qual o prazo em formato simplificado?")
    assert client.embedded == [["Qual o prazo em formato simplificado?"]]


def test_retrieval_cost_is_added_to_the_answer():
    answer = rag(FakeClient()).answer("?")
    assert answer.usage.prompt_tokens == 107 and answer.usage.cost_usd == pytest.approx(0.11)


def test_citations_outside_the_retrieved_set_are_kept_to_be_measured():
    output = BaselineOutput(answer="x", cited_articles=["lgpd:art:99"], confidence=0.5)
    answer = rag(FakeClient(output=output)).answer("?")
    assert answer.cited_articles == ["lgpd:art:99"]


def test_retrieval_metrics_now_apply():
    q = Question(
        id="q001", question="Pergunta de teste sobre a lei?", reference_answer="r",
        gold_articles=["lgpd:art:18"], category="factual", laws=["lgpd"],
    )  # fmt: skip
    scores = score(rag(FakeClient()).answer(q.question), q)
    assert scores.recall_at_5 == 1.0 and scores.mrr == 0.5


def test_streaming_also_retrieves_and_carries_retrieved_ids():
    client = FakeClient()
    *chunks, final = list(rag(client).answer_stream("?"))
    assert isinstance(final, Answer) and final.retrieved_ids == ["lgpd:art:19", "lgpd:art:18"]
    assert any(isinstance(c, StreamChunk) and c.channel == "text" for c in chunks)
    assert client.messages[0][-1]["content"].startswith("Trechos recuperados:")


def test_several_chunks_of_the_same_article_count_once():
    from lab.index.store import Hit

    hits = [
        Hit(chunk=chunk("lgpd:art:19", "#I"), score=0.9),
        Hit(chunk=chunk("lgpd:art:19", "#II"), score=0.8),
    ]
    assert unique_articles(hits) == ["lgpd:art:19"]
    assert render_hits(hits).count("[lgpd:art:19]") == 2


def test_invalid_k():
    with pytest.raises(ValueError, match="k deve ser"):
        rag(FakeClient(), k=0)


def test_prompt_requires_citations_from_the_excerpts():
    system = rag(FakeClient())._system  # type: ignore[attr-defined]
    assert "Toda afirmação precisa de um artigo dos trechos" in system
    assert "{laws}" not in system and "(lgpd)" in system


def test_factory_builds_rag_and_requires_the_index():
    strategy = build_strategy(
        "rag",
        FakeClient(),
        LAWS,
        prompts_dir=PROMPTS,
        index=make_index(),
        k=3,  # type: ignore[arg-type]
    )
    assert isinstance(strategy, RagStrategy) and strategy.k == 3 and strategy.name == "rag"
    with pytest.raises(ValueError, match="lab index"):
        build_strategy("rag", FakeClient(), LAWS, prompts_dir=PROMPTS)  # type: ignore[arg-type]
    assert json.loads(BaselineOutput(answer="x").model_dump_json())["answer"] == "x"
