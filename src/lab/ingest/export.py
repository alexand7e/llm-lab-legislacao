# -----------------------------------------------------------------------------
# File:     src/lab/ingest/export.py
# Purpose:  Build, validate and write data/processed/articles.jsonl (SPEC 2.3).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Exportação do corpus para ``articles.jsonl``.

Uma linha por artigo, no schema da SPEC 2.3. Cada registro é validado por um
modelo Pydantic ao ser criado e de novo ao ser lido, então um arquivo que
passou por ``write_jsonl`` sempre passa por ``read_jsonl``.

Extensões ao exemplo da SPEC, para o índice e o grafo não precisarem
reparsear nada: cada unidade traz ``key``, ``parent`` e ``status`` (há
dispositivos revogados dentro de artigos vigentes), e ``text`` do artigo só
contém dispositivos vigentes.

A saída é determinística (ordem das normas do corpus, depois ordem natural
dos artigos), para que o diff do arquivo versionado mostre só mudanças reais.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ValidationError, model_validator

from lab.ingest.collect import RawDocument
from lab.ingest.parser import Article, Status, article_key, parse_html
from lab.ingest.references import Reference, article_references


class CorpusError(ValueError):
    """Corpus inconsistente ou arquivo ``articles.jsonl`` inválido."""


class UnitRecord(BaseModel):
    """Dispositivo dentro do artigo (parágrafo, inciso ou alínea)."""

    type: str
    id: str
    key: str
    parent: str | None = None
    text: str
    status: Status = "vigente"


class ArticleRecord(BaseModel):
    """Uma linha de ``articles.jsonl``."""

    id: str
    law: str
    article: str
    path: list[str]
    status: Status
    text: str
    units: list[UnitRecord]
    references: list[Reference]
    source_hash: str
    collected_at: date

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.id != f"{self.law}:art:{self.article}":
            raise ValueError(f"id {self.id!r} não corresponde a {self.law}/{self.article}")
        article_key(self.article)  # ValueError se o número for inválido
        if not self.source_hash.startswith("sha256:"):
            raise ValueError("source_hash deve começar com sha256:")
        return self


def to_record(
    article: Article, doc: RawDocument, known: dict[str, str] | None = None
) -> ArticleRecord:
    """Converter um artigo parseado no registro exportado."""
    return ArticleRecord(
        id=article.id,
        law=article.law,
        article=article.article,
        path=article.path,
        status=article.status,
        text=article.text,
        units=[
            UnitRecord(
                type=u.type, id=u.id, key=u.key, parent=u.parent, text=u.text, status=u.status
            )
            for u in article.units
        ],
        references=article_references(article, known),
        source_hash=doc.sha256,
        collected_at=doc.collected_at,
    )


def build_records(
    html: str, doc: RawDocument, known: dict[str, str] | None = None
) -> list[ArticleRecord]:
    """Parsear o HTML de uma norma e devolver os registros, na ordem do documento."""
    return [to_record(a, doc, known) for a in parse_html(html, doc.law)]


def check_unique(records: Sequence[ArticleRecord]) -> None:
    """:raises CorpusError: há ids repetidos (o mesmo artigo parseado duas vezes)."""
    counts = Counter(r.id for r in records)
    repeated = sorted(i for i, n in counts.items() if n > 1)
    if repeated:
        raise CorpusError(f"ids repetidos: {', '.join(repeated)}")


def dangling_references(records: Sequence[ArticleRecord]) -> list[tuple[str, str]]:
    """Remissões para artigos de normas do corpus que não existem nele.

    Devolve pares ``(id do artigo de origem, alvo)``. Não é erro: o texto
    oficial pode citar artigos vetados que o parser não vê. Serve de alerta.
    """
    known_ids = {r.id for r in records}
    laws = {r.law for r in records}
    return [
        (r.id, ref.target)
        for r in records
        for ref in r.references
        if ref.target.split(":")[0] in laws
        and ":art:" in ref.target
        and ref.target not in known_ids
    ]


def sort_records(records: Sequence[ArticleRecord], law_order: Sequence[str]) -> list[ArticleRecord]:
    """Ordem canônica: normas na ordem do corpus, artigos em ordem natural."""
    rank = {law: i for i, law in enumerate(law_order)}
    return sorted(
        records, key=lambda r: (rank.get(r.law, len(rank)), r.law, article_key(r.article))
    )


def write_jsonl(path: Path, records: Sequence[ArticleRecord], law_order: Sequence[str]) -> None:
    """Gravar ``articles.jsonl`` (UTF-8, LF, uma linha por artigo, ordem canônica).

    :raises CorpusError: ids repetidos. Nada é gravado nesse caso.
    """
    check_unique(records)
    lines = [r.model_dump_json() for r in sort_records(records, law_order)]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[ArticleRecord]:
    """Ler e validar ``articles.jsonl``.

    :raises CorpusError: linha que não é JSON válido, fora do schema ou id repetido;
        a mensagem indica o número da linha.
    """
    records: list[ArticleRecord] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(ArticleRecord.model_validate_json(line))
            except ValidationError as exc:
                raise CorpusError(f"{path.name}:{number}: {exc.errors()[0]['msg']}") from exc
    check_unique(records)
    return records


__all__ = [
    "ArticleRecord",
    "CorpusError",
    "UnitRecord",
    "build_records",
    "check_unique",
    "dangling_references",
    "read_jsonl",
    "sort_records",
    "to_record",
    "write_jsonl",
]
