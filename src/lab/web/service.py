# -----------------------------------------------------------------------------
# File:     src/lab/web/service.py
# Purpose:  Turn a strategy's answer into the chat event stream, with verified citations.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Serviço de chat: pergunta em, eventos do chat-reference fora.

Os eventos seguem o protocolo de ``chat-reference/chat-renderer.js``
(``applyEvent``): ``meta``, ``tool_start``, ``tool_step``, ``tool_done``,
``tool_error``, ``markdown``, ``sources``. O evento ``done`` é da interface:
leva as estatísticas da resposta (tokens, custo, latência, confiança).

**Citações verificadas.** Cada artigo citado pelo modelo vira uma fonte com o
texto oficial vindo do ``articles.jsonl``. Se o artigo não existe no corpus, ou
está revogado ou vetado, a fonte diz isso: é a evidência de que o baseline
responde de memória e erra números de artigo.

Nenhum erro vaza para a tela além de uma mensagem curta: detalhes técnicos ficam
no log do servidor, e chaves nunca aparecem (o cliente LLM não as inclui).
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Mapping, Sequence
from typing import Any

from lab.ingest.export import ArticleRecord
from lab.llm import BudgetExceededError, LLMConfigError
from lab.strategies.base import Answer, Message, Strategy, StreamChunk, StreamingStrategy

logger = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]

SNIPPET_CHARS = 220
_NOT_FOUND = "não encontrado no corpus"


def law_label(law_id: str) -> str:
    """Rótulo legível do id da norma: ``lgpd`` -> ``LGPD``, ``marco_civil`` -> ``Marco Civil``."""
    if len(law_id) <= 4:
        return law_id.upper()
    return law_id.replace("_", " ").title()


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rstrip() + "…"


def verify_citations(
    cited: list[str], records: Mapping[str, ArticleRecord]
) -> list[dict[str, Any]]:
    """Fontes do painel: um item por artigo citado, com o texto oficial ou o aviso.

    ``exists`` é falso quando o artigo não está no corpus; ``status`` é o do
    artigo (``vigente``, ``revogado``, ``vetado``) ou ``None`` se não existe.
    """
    sources: list[dict[str, Any]] = []
    for n, article_id in enumerate(cited, start=1):
        law, _, number = article_id.partition(":art:")
        record = records.get(article_id)
        title = f"{law_label(law)} art. {number}"
        if record is None:
            sources.append(
                {
                    "n": n,
                    "title": title,
                    "ref": _NOT_FOUND,
                    "snippet": "Este artigo não existe nas normas do corpus.",
                    "exists": False,
                    "status": None,
                    "article_id": article_id,
                }
            )
            continue
        where = " > ".join(record.path)
        sources.append(
            {
                "n": n,
                "title": title,
                "ref": f"{record.status} · {where}" if where else record.status,
                "snippet": _snippet(record.text),
                "exists": True,
                "status": record.status,
                "article_id": article_id,
            }
        )
    return sources


def _chips(sources: list[dict[str, Any]]) -> str:
    """Linha de chips de citação (``[^n]``), ligados ao painel de fontes."""
    return "Artigos citados: " + " ".join(f"[^{s['n']}]" for s in sources)


def _markdown(answer: Answer, sources: list[dict[str, Any]]) -> str:
    """Resposta inteira com os chips de citação, para quem não faz streaming."""
    text = answer.text.strip()
    return f"{text}\n\n{_chips(sources)}" if sources else text


def _stats(answer: Answer, sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": answer.usage.prompt_tokens,
        "completion_tokens": answer.usage.completion_tokens,
        "cost_usd": answer.usage.cost_usd,
        "latency_ms": answer.latency_ms,
        "cached": answer.latency_ms == 0,
        "confidence": answer.confidence,
        "parse_error": answer.parse_error,
        "cited": len(sources),
        "missing": sum(1 for s in sources if not s["exists"]),
    }


class ChatService:
    """Responde perguntas com uma :class:`Strategy` e verifica as citações no corpus.

    :param strategy: qualquer estratégia (hoje o baseline; RAG entra sem mexer aqui).
    :param records: corpus indexado por id (``lgpd:art:19``).
    :param model: id do modelo, mostrado na etiqueta da resposta.
    """

    def __init__(
        self, strategy: Strategy, records: Mapping[str, ArticleRecord], *, model: str
    ) -> None:
        self.strategy = strategy
        self.records = records
        self.model = model

    def _stream(self, question: str, history: Sequence[Message]) -> Generator[Event, None, Answer]:
        """Eventos de uma resposta em streaming; devolve a :class:`Answer` final.

        Passos: "Consultando o modelo" até o primeiro texto da resposta, depois
        "Gerando a resposta". Raciocínio e saída bruta do modelo vão como eventos
        ``log`` (a tela os guarda nos logs recolhidos); o texto vai como ``text_chunk``.
        """
        assert isinstance(self.strategy, StreamingStrategy)
        generating = False
        for item in self.strategy.answer_stream(question, history):
            if isinstance(item, Answer):
                label = "Gerando a resposta" if generating else "Consultando o modelo"
                yield "tool_step", {"label": label, "state": "done"}
                return item
            assert isinstance(item, StreamChunk)
            if item.channel == "text":
                if not generating:
                    generating = True
                    yield "tool_step", {"label": "Consultando o modelo", "state": "done"}
                    yield "tool_step", {"label": "Gerando a resposta", "state": "running"}
                yield "text_chunk", {"delta": item.text}
            else:
                yield "log", {"channel": item.channel, "delta": item.text}
        raise RuntimeError("a estratégia terminou sem devolver a resposta")

    def events(self, question: str, history: Sequence[Message] = ()) -> Iterator[Event]:
        """Eventos da resposta a ``question``, na ordem em que a tela os consome.

        ``history`` é a conversa até aqui (sem a pergunta atual). Com uma estratégia
        que sabe fazer streaming a resposta chega em ``text_chunk`` enquanto o modelo
        gera; senão vem inteira num ``markdown`` ao final.
        """
        streaming = isinstance(self.strategy, StreamingStrategy)
        yield "meta", {"model": self.model, "strategy": self.strategy.name, "lang": "pt-BR"}
        yield "tool_start", {"title": "Respondendo à pergunta"}
        yield "tool_step", {"label": "Consultando o modelo", "state": "running"}
        try:
            if streaming:
                answer = yield from self._stream(question, history)
            else:
                answer = self.strategy.answer(question, history)
                yield "tool_step", {"label": "Consultando o modelo", "state": "done"}
        except (BudgetExceededError, LLMConfigError) as exc:
            yield "tool_error", {"title": "Não foi possível responder", "detail": str(exc)}
            return
        except Exception:
            logger.exception("falha ao responder pergunta")
            yield (
                "tool_error",
                {
                    "title": "Erro ao consultar o modelo",
                    "detail": "O provedor não respondeu. Tente de novo em instantes.",
                },
            )
            return
        yield "tool_step", {"label": "Verificando citações no corpus", "state": "running"}
        sources = verify_citations(answer.cited_articles, self.records)
        yield "tool_step", {"label": "Verificando citações no corpus", "state": "done"}
        yield "tool_done", {"result": {"kind": "none"}}
        if streaming:
            if sources:  # o texto já chegou em text_chunk; só faltam os chips de citação
                yield "markdown", {"md": _chips(sources)}
        else:
            yield "markdown", {"md": _markdown(answer, sources)}
        if sources:
            yield "sources", {"sources": sources}
        yield "done", {"stats": _stats(answer, sources)}


__all__ = ["ChatService", "Event", "law_label", "verify_citations"]
