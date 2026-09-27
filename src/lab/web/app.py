# -----------------------------------------------------------------------------
# File:     src/lab/web/app.py
# Purpose:  FastAPI app: chat over SSE, health, article lookup and the static UI.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Aplicação web local.

Rotas:

- ``POST /api/chat``: corpo ``{"question": "...", "history": [{"role", "content"}]}``
  (``history`` opcional: a conversa até aqui, para o modelo entender o contexto); responde em SSE
  (``text/event-stream``), um frame ``event:``/``data:`` por evento do
  :class:`~lab.web.service.ChatService`.
- ``GET /api/health``: corpus carregado e modelo em uso (sem segredos).
- ``GET /api/articles/{id}``: texto completo de um artigo (``lgpd:art:19``).
- ``/``: a interface estática (``static/``).

A fábrica :func:`create_app` recebe o serviço pronto, então os testes injetam
uma estratégia falsa e nada toca a rede. A chave do provedor só existe no
processo do servidor; o navegador nunca a vê.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from lab.strategies.base import Message
from lab.web.service import ChatService

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 2000
MAX_HISTORY = 40  # mensagens aceitas; a estratégia usa só as mais recentes
MAX_MESSAGE_CHARS = 8000


class ChatRequest(BaseModel):
    """Corpo de ``POST /api/chat``."""

    question: str = Field(max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default=[], max_length=MAX_HISTORY)

    @field_validator("history")
    @classmethod
    def _bounded_history(cls, value: list[Message]) -> list[Message]:
        for message in value:
            if len(message.content) > MAX_MESSAGE_CHARS:
                raise ValueError(f"mensagem do histórico maior que {MAX_MESSAGE_CHARS} caracteres")
        return value

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a pergunta não pode ser vazia")
        return value


def sse_frame(event: str, data: dict[str, Any]) -> str:
    """Um frame SSE: ``event: <nome>`` e ``data: <json>`` numa linha só."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service: ChatService, *, static_dir: Path = STATIC_DIR) -> FastAPI:
    """Montar a aplicação em torno de ``service``."""
    app = FastAPI(title="llm-lab-legislacao", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        laws = sorted({r.law for r in service.records.values()})
        return {
            "status": "ok",
            "articles": len(service.records),
            "laws": laws,
            "strategy": service.strategy.name,
            "model": service.model,
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        def frames() -> Iterator[str]:
            for event, data in service.events(request.question, request.history):
                yield sse_frame(event, data)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/articles/{article_id:path}")
    def article(article_id: str) -> dict[str, Any]:
        record = service.records.get(article_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"artigo {article_id} não está no corpus")
        return record.model_dump(
            mode="json", include={"id", "law", "article", "path", "status", "text"}
        )

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


__all__ = [
    "MAX_HISTORY",
    "MAX_MESSAGE_CHARS",
    "MAX_QUESTION_CHARS",
    "STATIC_DIR",
    "ChatRequest",
    "create_app",
    "sse_frame",
]
