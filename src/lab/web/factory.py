# -----------------------------------------------------------------------------
# File:     src/lab/web/factory.py
# Purpose:  Build the chat service from settings, models.yaml and the corpus.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Montagem do serviço de chat a partir da configuração real.

Mesma configuração de ``lab ask``: ``Settings`` + ``.env`` (via ``load_env``),
``models.yaml`` e ``corpus.yaml``. Fica separado de ``app.py`` para os testes da
API não precisarem de provedor nem de arquivos de configuração.
"""

from __future__ import annotations

from pathlib import Path

from lab.ingest.export import read_jsonl
from lab.ingest.sources import load_corpus
from lab.llm import LLMClient
from lab.llm.cache import SQLiteCache
from lab.settings import Settings, load_env, load_models_config
from lab.strategies import BaselineStrategy
from lab.web.service import ChatService


def build_service(
    settings: Settings,
    articles: Path,
    *,
    cache: SQLiteCache | None,
    models_config: Path | None = None,
    prompt: str = "baseline_v1",
    role: str = "generator",
) -> ChatService:
    """Serviço de chat com a estratégia baseline e o corpus de ``articles``.

    :raises FileNotFoundError: ``articles`` (rode ``lab ingest``), ``models.yaml``,
        ``corpus.yaml`` ou o prompt não existem.
    :raises lab.ingest.export.CorpusError: ``articles`` inválido.
    :raises KeyError: ``role`` não está em ``models.yaml``.
    """
    records = {r.id: r for r in read_jsonl(articles)}
    models = load_models_config(models_config or settings.lab_models_config)
    corpus = load_corpus(settings.lab_corpus_config)
    client = LLMClient(models, cache=cache, max_usd=settings.lab_max_usd_per_run, env=load_env())
    strategy = BaselineStrategy(client, corpus.laws, prompt=prompt, role=role)
    return ChatService(strategy, records, model=models.roles[role].model)


__all__ = ["build_service"]
