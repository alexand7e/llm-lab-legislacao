# -----------------------------------------------------------------------------
# File:     src/lab/llm/cache.py
# Purpose:  SQLite response cache keyed by hash of (model, messages, params).
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Cache em disco (SQLite) de respostas de LLM.

Cada chamada do cliente é cacheada sob a chave
:func:`make_key` — SHA-256 de ``(modelo, mensagens, parâmetros)`` em JSON
com chaves ordenadas. A determinismo da chave é o que torna reruns de
avaliação gratuitos e os resultados reproduzíveis (spec, seção 4.3):
mesmo modelo, mesma conversa e mesmos parâmetros → mesmo resultado,
sem novo gasto de API.

O banco vive em ``<LAB_CACHE_DIR>/cache.sqlite`` (padrão ``.cache/llm``),
que é gitignored. Testes usam um diretório temporário
(``tmp_path``) e nunca tocam o cache real.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def make_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    """Chave estável de cache: SHA-256 de ``(modelo, mensagens, parâmetros)``.

    O JSON é serializado com ``sort_keys=True`` para que a ordem das chaves
    não mude a chave; ``ensure_ascii=False`` preserva acentos e evita
    escapes que divergissem entre plataformas.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    """Tabela ``cache(key, value)`` em SQLite; ``value`` é JSON.

    Uma instância por processo, segura para uso por várias threads (o runner
    de avaliação chama o cliente em paralelo): a conexão é compartilhada e cada
    operação é serializada por um lock. Crie o diretório se não existir e
    chame :meth:`close` ao fim do uso.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(directory / "cache.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retornar o valor cacheado para a chave, ou ``None`` se ausente."""
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Gravar (ou sobrescrever) o valor para a chave."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._db.commit()

    def close(self) -> None:
        """Fechar a conexão com o SQLite."""
        with self._lock:
            self._db.close()
