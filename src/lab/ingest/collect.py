# -----------------------------------------------------------------------------
# File:     src/lab/ingest/collect.py
# Purpose:  Download law HTML from Planalto, recording SHA-256, URL and date.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Coleta do HTML bruto das leis (SPEC 2.2, "versão compilada").

Para cada lei, grava em ``data/raw/`` (fora do git, reprodutível via CLI):

- ``<lei>.html``: os bytes exatamente como vieram do servidor;
- ``<lei>.meta.json``: URL de origem, ``sha256:`` dos bytes e data da coleta.

O hash é dos bytes brutos, então identifica a versão do texto usada em cada
``articles.jsonl``, mesmo que o Planalto altere a página depois.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import httpx2
from pydantic import BaseModel

from lab.ingest.sources import LawSource

# O Planalto recusa a conexão quando o User-Agent não começa com "Mozilla/5.0".
USER_AGENT = (
    "Mozilla/5.0 (compatible; llm-lab-legislacao/0.1; "
    "+https://github.com/alexand7e/llm-lab-legislacao)"
)


class RawDocument(BaseModel):
    """Metadados de um HTML coletado (conteúdo de ``<lei>.meta.json``)."""

    law: str
    url: str
    sha256: str  # "sha256:<hex>" dos bytes brutos
    collected_at: date


def make_client() -> httpx2.Client:
    """Cliente HTTP para o Planalto: User-Agent aceito, redirects e retentativas."""
    return httpx2.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
        transport=httpx2.HTTPTransport(retries=3),
    )


def sha256_of(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def html_path(raw_dir: Path, law: str) -> Path:
    return raw_dir / f"{law}.html"


def meta_path(raw_dir: Path, law: str) -> Path:
    return raw_dir / f"{law}.meta.json"


def collect(
    source: LawSource,
    raw_dir: Path,
    *,
    client: httpx2.Client,
    today: date | None = None,
) -> RawDocument:
    """Baixar a lei, gravar HTML e metadados em ``raw_dir`` e devolver os metadados.

    :raises httpx2.HTTPStatusError: resposta não-2xx (nada é gravado).
    """
    response = client.get(source.url)
    response.raise_for_status()
    content = response.content
    doc = RawDocument(
        law=source.id,
        url=str(response.url),
        sha256=sha256_of(content),
        collected_at=today or date.today(),
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path(raw_dir, source.id).write_bytes(content)
    meta_path(raw_dir, source.id).write_text(doc.model_dump_json(indent=2) + "\n", "utf-8")
    return doc


def decode_html(content: bytes) -> str:
    """Decodificar o HTML do Planalto: UTF-8 se válido, senão cp1252 (o usual lá)."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def load_raw(raw_dir: Path, law: str) -> tuple[str, RawDocument]:
    """Ler um HTML já coletado: texto decodificado + metadados.

    Confere o hash contra ``meta.json`` para não parsear um arquivo alterado à mão.

    :raises FileNotFoundError: a lei ainda não foi coletada.
    :raises ValueError: o HTML não corresponde ao hash registrado.
    """
    content = html_path(raw_dir, law).read_bytes()
    doc = RawDocument.model_validate_json(meta_path(raw_dir, law).read_text("utf-8"))
    if sha256_of(content) != doc.sha256:
        raise ValueError(f"{law}: HTML em {raw_dir} não confere com o sha256 registrado")
    return decode_html(content), doc
