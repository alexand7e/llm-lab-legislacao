# -----------------------------------------------------------------------------
# File:     tests/test_web.py
# Purpose:  Tests for the chat service, the FastAPI app and `lab serve` (no network).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

import json
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import lab.cli
from lab.cli import app as cli_app
from lab.ingest.export import ArticleRecord
from lab.llm import BudgetExceededError, LLMConfigError
from lab.llm.costs import Usage
from lab.strategies.base import Answer, Message, StreamChunk, StreamingStrategy
from lab.web.app import MAX_QUESTION_CHARS, STATIC_DIR, create_app, sse_frame
from lab.web.service import ChatService, law_label, verify_citations

ROOT = Path(__file__).parent.parent
runner = CliRunner()


def record(
    article_id: str, status: str = "vigente", text: str = "Art. 19. Texto oficial."
) -> ArticleRecord:
    law, _, number = article_id.partition(":art:")
    return ArticleRecord(
        id=article_id, law=law, article=number, path=["Título I", "Capítulo II"], status=status,
        text=text, units=[], references=[], source_hash="sha256:x", collected_at=date(2026, 9, 26),
    )  # fmt: skip


RECORDS = {
    r.id: r
    for r in (
        record("lgpd:art:19", text="Art. 19. Confirmação imediata, em formato simplificado."),
        record("lgpd:art:20", status="revogado", text="Art. 20. (Revogado)."),
    )
}


class FakeStrategy:
    name = "fake"

    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self.answer_ = answer or Answer(
            text="Imediatamente.",
            cited_articles=["lgpd:art:19", "lgpd:art:99"],
            confidence=0.8,
            usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.5),
            latency_ms=1200,
        )
        self.error = error
        self.questions: list[str] = []
        self.histories: list[list[Message]] = []

    def answer(self, question: str, history: Sequence[Message] = ()) -> Answer:
        self.questions.append(question)
        self.histories.append(list(history))
        if self.error is not None:
            raise self.error
        return self.answer_


def service(**kw) -> ChatService:
    return ChatService(FakeStrategy(**kw), RECORDS, model="modelo-x")


def client(**kw) -> TestClient:
    return TestClient(create_app(service(**kw)))


def frames(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# -- citações -----------------------------------------------------------------


def test_law_label():
    assert law_label("lgpd") == "LGPD" and law_label("cdc") == "CDC"
    assert law_label("marco_civil") == "Marco Civil"


def test_verify_citations_marks_existing_missing_and_revoked():
    sources = verify_citations(["lgpd:art:19", "lgpd:art:99", "lgpd:art:20"], RECORDS)
    found, missing, revoked = sources
    assert found["title"] == "LGPD art. 19" and found["exists"] and found["n"] == 1
    assert found["ref"] == "vigente · Título I > Capítulo II"
    assert "Confirmação imediata" in found["snippet"]
    assert not missing["exists"] and missing["ref"] == "não encontrado no corpus"
    assert missing["status"] is None and missing["n"] == 2
    assert revoked["exists"] and revoked["status"] == "revogado"


def test_snippet_is_truncated_and_flattened():
    long = {"lgpd:art:1": record("lgpd:art:1", text="palavra " * 100 + "\nfim")}
    (source,) = verify_citations(["lgpd:art:1"], long)
    assert len(source["snippet"]) <= 221 and source["snippet"].endswith("…")
    assert "\n" not in source["snippet"]


# -- eventos ------------------------------------------------------------------


def test_events_follow_the_chat_reference_protocol():
    events = list(service().events("Qual o prazo?"))
    names = [name for name, _ in events]
    assert names == [
        "meta", "tool_start", "tool_step", "tool_step", "tool_step", "tool_step",
        "tool_done", "markdown", "sources", "done",
    ]  # fmt: skip
    data = dict(events)
    assert data["meta"] == {"model": "modelo-x", "strategy": "fake", "lang": "pt-BR"}
    steps = [d for n, d in events if n == "tool_step"]
    assert [(s["label"], s["state"]) for s in steps] == [
        ("Consultando o modelo", "running"),
        ("Consultando o modelo", "done"),
        ("Verificando citações no corpus", "running"),
        ("Verificando citações no corpus", "done"),
    ]
    assert data["markdown"]["md"] == "Imediatamente.\n\nArtigos citados: [^1] [^2]"
    assert len(data["sources"]["sources"]) == 2


def test_done_carries_stats_including_missing_citations():
    stats = dict(service().events("?"))["done"]["stats"]
    assert stats == {
        "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.5, "latency_ms": 1200,
        "cached": False, "confidence": 0.8, "parse_error": None, "cited": 2, "missing": 1,
    }  # fmt: skip


def test_answer_without_citations_has_no_sources_event():
    answer = Answer(text="Fora do escopo.", usage=Usage(), latency_ms=0)
    events = list(service(answer=answer).events("?"))
    assert "sources" not in [n for n, _ in events]
    data = dict(events)
    assert data["markdown"]["md"] == "Fora do escopo."
    assert data["done"]["stats"]["cached"] is True


@pytest.mark.parametrize("error", [BudgetExceededError("limite"), LLMConfigError("sem chave")])
def test_known_errors_become_tool_error_with_their_message(error: Exception):
    events = list(service(error=error).events("?"))
    assert events[-1] == (
        "tool_error",
        {"title": "Não foi possível responder", "detail": str(error)},
    )
    assert "markdown" not in [n for n, _ in events]


def test_unexpected_error_hides_details():
    events = list(service(error=RuntimeError("segredo: sk-123")).events("?"))
    name, data = events[-1]
    assert name == "tool_error" and "sk-123" not in json.dumps(data)


# -- API ----------------------------------------------------------------------


def test_sse_frame_format():
    assert sse_frame("markdown", {"md": "olá"}) == 'event: markdown\ndata: {"md": "olá"}\n\n'


def test_chat_endpoint_streams_sse():
    response = client().post("/api/chat", json={"question": "  Qual o prazo?  "})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    parsed = frames(response.text)
    assert parsed[0][0] == "meta" and parsed[-1][0] == "done"
    assert dict(parsed)["markdown"]["md"].startswith("Imediatamente.")


def test_chat_endpoint_passes_the_trimmed_question_to_the_strategy():
    svc = service()
    TestClient(create_app(svc)).post("/api/chat", json={"question": "  Qual o prazo?  "})
    assert svc.strategy.questions == ["Qual o prazo?"]  # type: ignore[attr-defined]


@pytest.mark.parametrize("question", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_chat_endpoint_rejects_invalid_questions(question: str):
    assert client().post("/api/chat", json={"question": question}).status_code == 422


def test_chat_endpoint_rejects_missing_body():
    assert client().post("/api/chat", json={}).status_code == 422


def test_health_reports_corpus_and_model_without_secrets():
    body = client().get("/api/health").json()
    assert body == {
        "status": "ok", "articles": 2, "laws": ["lgpd"], "strategy": "fake", "model": "modelo-x",
    }  # fmt: skip


def test_article_lookup():
    found = client().get("/api/articles/lgpd:art:19")
    assert found.status_code == 200
    assert found.json()["id"] == "lgpd:art:19" and "Confirmação" in found.json()["text"]
    assert client().get("/api/articles/lgpd:art:999").status_code == 404


# -- interface estática -------------------------------------------------------


def test_index_and_assets_are_served():
    web = client()
    index = web.get("/")
    assert index.status_code == 200 and "Laboratório de legislação" in index.text
    for asset in ("app.js", "app.css", "chat-renderer.js", "chat-messages.css"):
        assert web.get(f"/{asset}").status_code == 200, asset


@pytest.mark.parametrize("name", ["chat-renderer.js", "chat-messages.css"])
def test_ui_uses_the_chat_reference_files_unchanged(name: str):
    """O chat deve ser o do chat-reference: as cópias não podem divergir."""
    ours = (STATIC_DIR / name).read_bytes().replace(b"\r\n", b"\n")
    reference = (ROOT / "chat-reference" / name).read_bytes().replace(b"\r\n", b"\n")
    assert ours == reference


# -- lab serve ----------------------------------------------------------------


@pytest.fixture
def serve_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    calls: list[dict] = []

    def fake_run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(lab.cli, "build_service", lambda *a, **k: service())
    return calls


def test_serve_runs_uvicorn_on_localhost_by_default(serve_env):
    result = runner.invoke(cli_app, ["serve"])
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8000" in result.stdout
    (call,) = serve_env
    assert (call["host"], call["port"]) == ("127.0.0.1", 8000)


def test_serve_custom_port(serve_env):
    assert runner.invoke(cli_app, ["serve", "--port", "8123"]).exit_code == 0
    assert serve_env[0]["port"] == 8123


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.10", "example.com"])
def test_serve_refuses_non_loopback_without_allow_remote(serve_env, host: str):
    result = runner.invoke(cli_app, ["serve", "--host", host])
    assert result.exit_code == 1
    assert "--allow-remote" in result.stderr
    assert serve_env == []  # nem chegou a subir


def test_serve_allows_remote_when_explicit(serve_env):
    result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0", "--allow-remote"])
    assert result.exit_code == 0, result.output
    assert serve_env[0]["host"] == "0.0.0.0"


def test_serve_missing_articles_suggests_ingest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LAB_MODELS_CONFIG", str(ROOT / "config" / "models.yaml"))
    monkeypatch.setenv("LAB_CORPUS_CONFIG", str(ROOT / "config" / "corpus.yaml"))
    result = runner.invoke(cli_app, ["serve"])
    assert result.exit_code == 1
    assert "lab ingest" in result.stderr


# -- contexto da conversa -----------------------------------------------------


def test_chat_endpoint_passes_history_to_the_strategy():
    svc = service()
    history = [
        {"role": "user", "content": "Fale da LGPD"},
        {"role": "assistant", "content": "É a Lei 13.709."},
    ]
    response = TestClient(create_app(svc)).post(
        "/api/chat", json={"question": "Quem fez essa lei?", "history": history}
    )
    assert response.status_code == 200
    assert svc.strategy.histories == [[Message(**m) for m in history]]  # type: ignore[attr-defined]


def test_history_is_optional():
    svc = service()
    TestClient(create_app(svc)).post("/api/chat", json={"question": "Pergunta sem contexto?"})
    assert svc.strategy.histories == [[]]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "system", "content": "x"}],  # papel inválido
        [{"role": "user"}],  # sem conteúdo
        [{"role": "user", "content": "x" * 8001}],  # mensagem grande demais
        [{"role": "user", "content": "x"}] * 41,  # histórico grande demais
    ],
)
def test_invalid_history_is_rejected(history: list):
    response = client().post("/api/chat", json={"question": "Pergunta válida?", "history": history})
    assert response.status_code == 422


# -- serviço e API com streaming ----------------------------------------------


def stream_record(article_id: str) -> ArticleRecord:
    law, _, number = article_id.partition(":art:")
    return ArticleRecord(
        id=article_id, law=law, article=number, path=["Título I"], status="vigente",
        text="Art. 19. Texto oficial.", units=[], references=[], source_hash="sha256:x",
        collected_at=date(2026, 9, 26),
    )  # fmt: skip


STREAM_RECORDS = {"lgpd:art:19": stream_record("lgpd:art:19")}
FINAL = Answer(
    text="Imediatamente.",
    cited_articles=["lgpd:art:19", "lgpd:art:99"],
    usage=Usage(prompt_tokens=3, completion_tokens=2, cost_usd=0.1),
    latency_ms=900,
)


class StreamingFake:
    name = "fake-stream"

    def __init__(self, items: list | None = None, error: Exception | None = None) -> None:
        self.items = items if items is not None else [
            StreamChunk("reasoning", "Pensando"),
            StreamChunk("raw", '{"answer": "Imedi'),
            StreamChunk("text", "Imedi"),
            StreamChunk("raw", 'atamente."}'),
            StreamChunk("text", "atamente."),
            FINAL,
        ]  # fmt: skip
        self.error = error
        self.histories: list[list[Message]] = []

    def answer(self, question: str, history: Sequence[Message] = ()) -> Answer:
        return FINAL

    def answer_stream(self, question: str, history: Sequence[Message] = ()) -> Iterator:
        self.histories.append(list(history))
        yield from self.items
        if self.error is not None:
            raise self.error


def stream_service(**kw) -> ChatService:
    return ChatService(StreamingFake(**kw), STREAM_RECORDS, model="m")


def test_streaming_fake_is_a_streaming_strategy():
    assert isinstance(StreamingFake(), StreamingStrategy)


def test_service_streams_text_chunks_and_logs():
    events = list(stream_service().events("Qual o prazo?"))
    names = [n for n, _ in events]
    assert names[:3] == ["meta", "tool_start", "tool_step"]
    deltas = [d["delta"] for n, d in events if n == "text_chunk"]
    assert deltas == ["Imedi", "atamente."]
    logs = [(d["channel"], d["delta"]) for n, d in events if n == "log"]
    assert logs[0] == ("reasoning", "Pensando") and [c for c, _ in logs] == [
        "reasoning",
        "raw",
        "raw",
    ]
    steps = [(d["label"], d["state"]) for n, d in events if n == "tool_step"]
    assert steps == [
        ("Consultando o modelo", "running"),
        ("Consultando o modelo", "done"),
        ("Gerando a resposta", "running"),
        ("Gerando a resposta", "done"),
        ("Verificando citações no corpus", "running"),
        ("Verificando citações no corpus", "done"),
    ]
    # o texto não é repetido: só os chips de citação vêm em markdown
    assert dict(events)["markdown"] == {"md": "Artigos citados: [^1] [^2]"}
    assert names[-1] == "done" and "sources" in names


def test_service_chunks_arrive_before_the_final_events():
    names = [n for n, _ in stream_service().events("?")]
    assert names.index("text_chunk") < names.index("tool_done") < names.index("sources")


def test_service_passes_history_to_the_streaming_strategy():
    svc = stream_service()
    history = [Message(role="user", content="antes")]
    list(svc.events("agora?", history))
    assert svc.strategy.histories == [history]  # type: ignore[attr-defined]


def test_stream_without_final_answer_becomes_a_tool_error():
    events = list(stream_service(items=[StreamChunk("text", "x")]).events("?"))
    assert events[-1][0] == "tool_error"


def test_error_mid_stream_keeps_the_partial_text_and_hides_details():
    svc = stream_service(
        items=[StreamChunk("text", "parcial")], error=RuntimeError("segredo sk-999")
    )
    events = list(svc.events("?"))
    assert ("text_chunk", {"delta": "parcial"}) in events
    name, data = events[-1]
    assert name == "tool_error" and "sk-999" not in json.dumps(data)


def test_budget_error_after_the_stream_is_reported():
    svc = stream_service(items=[StreamChunk("text", "x")], error=BudgetExceededError("estourou"))
    assert list(svc.events("?"))[-1] == (
        "tool_error",
        {"title": "Não foi possível responder", "detail": "estourou"},
    )


def test_endpoint_streams_sse_frames_progressively():
    web = TestClient(create_app(stream_service()))
    with web.stream("POST", "/api/chat", json={"question": "Qual o prazo?"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    names = [block.split("\n")[0].removeprefix("event: ") for block in text.strip().split("\n\n")]
    assert names.count("text_chunk") == 2 and names[0] == "meta" and names[-1] == "done"
