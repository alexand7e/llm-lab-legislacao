# -----------------------------------------------------------------------------
# File:     tests/test_streaming.py
# Purpose:  Tests for token streaming: JSON extractor, client stream and baseline.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
import random
from collections.abc import Sequence
from pathlib import Path

import httpx2
import pytest
from openai import BadRequestError, DefaultHttpxClient

from lab.ingest.sources import LawSource
from lab.llm import BudgetExceededError, LLMClient
from lab.llm.cache import SQLiteCache
from lab.llm.client import ChatResult, StreamDelta
from lab.settings import ModelsConfig, Provider, Role
from lab.strategies.base import Answer, Message, StreamChunk
from lab.strategies.baseline import MAX_HISTORY_MESSAGES, BaselineStrategy
from lab.strategies.streaming import AnswerExtractor

ROOT = Path(__file__).parent.parent
MESSAGES = [{"role": "user", "content": "olá"}]


# -- extrator do campo "answer" -----------------------------------------------


def extract(chunks: list[str]) -> str:
    extractor = AnswerExtractor()
    return "".join(extractor.feed(c) for c in chunks)


def split_at(text: str, cuts: list[int]) -> list[str]:
    points = [0, *sorted(set(cuts)), len(text)]
    return [text[a:b] for a, b in zip(points, points[1:], strict=False)]


PAYLOADS = [
    {"answer": "Imediatamente.", "cited_articles": ["lgpd:art:19"], "confidence": 0.9},
    {"answer": 'Texto com "aspas", barra \\ e nova\nlinha\ttab.', "cited_articles": []},
    {"answer": "Ação de reparação: art. 43 — “citação” e emoji 😀 fim."},
    {"cited_articles": ["cdc:art:6"], "answer": "Campo depois de outro.", "confidence": 1},
]


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_extractor_matches_json_for_every_single_split(payload: dict, ensure_ascii: bool):
    text = json.dumps(payload, ensure_ascii=ensure_ascii)
    for cut in range(len(text) + 1):  # inclusive cortes no meio de escapes e de \uXXXX
        assert extract(split_at(text, [cut])) == payload["answer"], (cut, text)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_extractor_handles_char_by_char_and_random_splits(payload: dict):
    text = json.dumps(payload, ensure_ascii=True)
    assert extract(list(text)) == payload["answer"]
    rng = random.Random(7)
    for _ in range(50):
        cuts = [rng.randrange(len(text)) for _ in range(rng.randrange(1, 8))]
        assert extract(split_at(text, cuts)) == payload["answer"]


def test_extractor_emits_incrementally_and_reports_finished():
    extractor = AnswerExtractor()
    assert extractor.feed('{"ans') == ""
    assert extractor.feed('wer": "Ol') == "Ol"
    assert not extractor.finished
    assert extractor.feed("á mun") == "á mun"
    assert extractor.feed('do", "cited_articles": []}') == "do"
    assert extractor.finished
    assert extractor.feed("qualquer coisa depois") == ""


def test_extractor_waits_for_incomplete_escape():
    extractor = AnswerExtractor()
    assert extractor.feed('{"answer": "a\\') == "a"  # barra no fim: espera
    assert extractor.feed("n") == "\n"
    assert extractor.feed("\\u00e") == ""  # \uXXXX incompleto
    assert extractor.feed("7") == "ç"


def test_extractor_passes_free_text_through():
    extractor = AnswerExtractor()
    assert extractor.feed("  ") == ""
    assert extractor.feed("Não sei ") == "  Não sei "
    assert extractor.feed("responder.") == "responder."
    assert extractor.is_plain_text


def test_extractor_ignores_json_without_answer_key():
    assert extract(['{"cited_articles": ["lgpd:art:1"]}']) == ""


# -- cliente: chat_stream -----------------------------------------------------


def sse(deltas: Sequence[dict], usage: dict | None = None) -> bytes:
    frames = []
    for delta in deltas:
        chunk = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        frames.append(f"data: {json.dumps(chunk)}\n\n")
    if usage:
        final = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        frames.append(f"data: {json.dumps({**final, 'choices': [], 'usage': usage})}\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def stream_response(
    content_parts: list[str], reasoning: list[str] | None = None
) -> httpx2.Response:
    deltas = [{"role": "assistant", "reasoning_content": r} for r in (reasoning or [])]
    deltas += [{"content": c} for c in content_parts]
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    return httpx2.Response(
        200, content=sse(deltas, usage), headers={"content-type": "text/event-stream"}
    )


class FakeServer:
    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def client(self) -> DefaultHttpxClient:
        return DefaultHttpxClient(transport=httpx2.MockTransport(self))


def make_client(server: FakeServer, **kw) -> LLMClient:
    config = ModelsConfig(
        roles={"generator": Role(provider="p", model="m", price_in=1.0, price_out=2.0)},
        providers={"p": Provider(base_url="https://example.test/v1", api_key_env="K")},
    )
    return LLMClient(
        config, env={"K": "secret"}, http_client=server.client(), sleep=lambda _s: None, **kw
    )


def test_chat_stream_yields_deltas_then_result():
    server = FakeServer(stream_response(["Ol", "á"], reasoning=["Pensando…"]))
    items = list(make_client(server).chat_stream("generator", MESSAGES))
    *deltas, result = items
    assert deltas == [
        StreamDelta("reasoning", "Pensando…"),
        StreamDelta("content", "Ol"),
        StreamDelta("content", "á"),
    ]
    assert isinstance(result, ChatResult)
    assert result.text == "Olá" and not result.cached
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (100, 50)
    assert result.usage.cost_usd == pytest.approx(0.0002)
    body = server.requests[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


def test_chat_stream_uses_and_fills_the_cache(tmp_path: Path):
    server = FakeServer(stream_response(["Ol", "á"]))
    cache = SQLiteCache(tmp_path)
    client = make_client(server, cache=cache)
    first = list(client.chat_stream("generator", MESSAGES))
    second = list(client.chat_stream("generator", MESSAGES))
    assert len(server.requests) == 1
    assert second[0] == StreamDelta("content", "Olá")  # do cache: inteiro, de uma vez
    assert isinstance(second[1], ChatResult) and second[1].cached
    assert second[1].text == first[-1].text  # type: ignore[union-attr]
    cache.close()


def test_chat_stream_retries_before_the_stream_opens():
    server = FakeServer(
        httpx2.Response(429, json={"error": {"message": "slow"}}), stream_response(["ok"])
    )
    *_, result = make_client(server).chat_stream("generator", MESSAGES)
    assert isinstance(result, ChatResult) and result.text == "ok" and len(server.requests) == 2


def test_chat_stream_falls_back_when_response_format_is_rejected():
    server = FakeServer(
        httpx2.Response(400, json={"error": {"message": "no"}}), stream_response(["{}"])
    )
    fmt = {"type": "json_schema", "json_schema": {"name": "X", "schema": {}}}
    *_, result = make_client(server).chat_stream("generator", MESSAGES, response_format=fmt)
    assert isinstance(result, ChatResult) and result.text == "{}"
    assert "response_format" in server.requests[0] and "response_format" not in server.requests[1]


def test_chat_stream_does_not_swallow_a_bad_request_without_format():
    server = FakeServer(httpx2.Response(400, json={"error": {"message": "no"}}))
    with pytest.raises(BadRequestError):
        list(make_client(server).chat_stream("generator", MESSAGES))


def test_chat_stream_enforces_the_budget_after_the_stream():
    server = FakeServer(stream_response(["ok"]))
    client = make_client(server, max_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        list(client.chat_stream("generator", MESSAGES))


def test_chat_stream_reads_reasoning_from_either_field():
    body = sse(
        [{"reasoning": "novo campo"}, {"content": "x"}],
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    server = FakeServer(
        httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    items = list(make_client(server).chat_stream("generator", MESSAGES))
    assert items[0] == StreamDelta("reasoning", "novo campo")


# -- baseline em streaming ----------------------------------------------------

LAWS = [
    LawSource(id="lgpd", name="Lei Geral de Proteção de Dados", number="Lei 13.709/2018", url="u")
]


def baseline_for(server: FakeServer) -> BaselineStrategy:
    return BaselineStrategy(make_client(server), LAWS, prompts_dir=ROOT / "prompts")


def json_parts(payload: dict, size: int = 7) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False)
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_baseline_streams_text_raw_and_final_answer():
    payload = {
        "answer": "Imediatamente, conforme a lei.",
        "cited_articles": ["LGPD:art:19"],
        "confidence": 0.9,
    }
    server = FakeServer(stream_response(json_parts(payload), reasoning=["Vou pensar."]))
    items = list(baseline_for(server).answer_stream("Qual o prazo?"))
    *chunks, answer = items
    assert isinstance(answer, Answer)
    assert answer.text == "Imediatamente, conforme a lei."
    assert answer.cited_articles == ["lgpd:art:19"] and answer.confidence == 0.9
    assert answer.usage.prompt_tokens == 100
    text = "".join(c.text for c in chunks if isinstance(c, StreamChunk) and c.channel == "text")
    assert text == "Imediatamente, conforme a lei."  # só o campo answer, sem JSON em volta
    raw = "".join(c.text for c in chunks if isinstance(c, StreamChunk) and c.channel == "raw")
    assert json.loads(raw) == payload
    assert [c.text for c in chunks if c.channel == "reasoning"] == ["Vou pensar."]  # type: ignore[union-attr]
    text_chunks = [c for c in chunks if c.channel == "text"]  # type: ignore[union-attr]
    assert len(text_chunks) > 1  # incremental de verdade


def test_baseline_stream_with_free_text_keeps_text_and_flags_parse_error():
    server = FakeServer(stream_response(["Não sei ", "responder."]))
    *chunks, answer = baseline_for(server).answer_stream("Pergunta?")
    assert isinstance(answer, Answer)
    assert "".join(c.text for c in chunks if c.channel == "text") == "Não sei responder."  # type: ignore[union-attr]
    assert answer.text == "Não sei responder." and answer.parse_error


def test_baseline_sends_history_between_system_and_question():
    server = FakeServer(stream_response(json_parts({"answer": "ok"})))
    history = [
        Message(role="user", content="Fale da LGPD"),
        Message(role="assistant", content="É a Lei 13.709."),
    ]
    list(baseline_for(server).answer_stream("Quem fez essa lei?", history))
    roles = [m["role"] for m in server.requests[0]["messages"]]
    contents = [m["content"] for m in server.requests[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1:] == ["Fale da LGPD", "É a Lei 13.709.", "Quem fez essa lei?"]


def test_baseline_history_is_limited_and_skips_blank_messages():
    server = FakeServer(stream_response(json_parts({"answer": "ok"})))
    history = [Message(role="user", content=f"m{i}") for i in range(MAX_HISTORY_MESSAGES + 5)]
    history.append(Message(role="assistant", content="   "))
    list(baseline_for(server).answer_stream("Atual?", history))
    sent = server.requests[0]["messages"]
    assert len(sent) == 1 + MAX_HISTORY_MESSAGES + 1
    assert sent[1]["content"] == "m5"  # descartou as mais antigas e a mensagem em branco


def test_history_also_reaches_non_streaming_answer():
    content = json.dumps({"answer": "ok"})
    message = {"role": "assistant", "content": content}
    server = FakeServer(
        httpx2.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    history = [Message(role="user", content="contexto")]
    baseline_for(server).answer("Pergunta?", history)
    assert [m["role"] for m in server.requests[0]["messages"]] == ["system", "user", "user"]
