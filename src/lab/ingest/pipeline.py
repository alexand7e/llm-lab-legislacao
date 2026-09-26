# -----------------------------------------------------------------------------
# File:     src/lab/ingest/pipeline.py
# Purpose:  Orchestrate collect, parse and export of the whole corpus.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Pipeline de ingestão: coleta (se preciso), parse e exportação.

É a lógica por trás de ``lab ingest``, separada do CLI para ser testável sem
terminal. Para cada norma de ``corpus.yaml``:

1. usa o HTML já coletado em ``raw_dir`` ou o baixa (sempre, com ``fetch``);
2. confere o hash contra ``<lei>.meta.json`` e parseia;
3. junta os registros de todas as normas e grava ``articles.jsonl``.

Nada é gravado em ``articles.jsonl`` se qualquer norma falhar: o arquivo
sempre reflete o corpus inteiro.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import httpx2
from pydantic import BaseModel

from lab.ingest.collect import collect, html_path, load_raw, make_client
from lab.ingest.export import (
    ArticleRecord,
    CorpusError,
    build_records,
    dangling_references,
    write_jsonl,
)
from lab.ingest.references import known_laws
from lab.ingest.sources import CorpusConfig


class LawSummary(BaseModel):
    """Resumo de uma norma ingerida."""

    law: str
    articles: int
    units: int
    by_status: dict[str, int]
    source_hash: str


class IngestReport(BaseModel):
    """Resultado de uma ingestão: por norma e alertas."""

    output: Path
    laws: list[LawSummary]
    dangling: list[tuple[str, str]]

    @property
    def total(self) -> int:
        return sum(s.articles for s in self.laws)


def run_ingest(
    corpus: CorpusConfig,
    raw_dir: Path,
    output: Path,
    *,
    fetch: bool = False,
    client: httpx2.Client | None = None,
) -> IngestReport:
    """Coletar (se preciso), parsear e exportar o corpus inteiro.

    :param fetch: baixar de novo todas as normas, mesmo já coletadas.
    :param client: cliente HTTP (padrão: :func:`lab.ingest.collect.make_client`).
    :raises CorpusError: norma sem artigos, ids repetidos ou arquivo inválido.
    :raises FileNotFoundError: ``raw_dir`` incompleto e a coleta não foi possível.
    :raises ValueError: HTML em ``raw_dir`` não confere com o hash registrado.
    :raises httpx2.HTTPError: falha ao baixar uma norma.
    """
    known = known_laws(corpus)
    records: list[ArticleRecord] = []
    summaries: list[LawSummary] = []
    owned_client = client is None
    http = client
    try:
        for law in corpus.laws:
            if fetch or not html_path(raw_dir, law.id).exists():
                if http is None:
                    http = make_client()
                collect(law, raw_dir, client=http)
            html, doc = load_raw(raw_dir, law.id)
            law_records = build_records(html, doc, known)
            if not law_records:
                raise CorpusError(f"{law.id}: nenhum artigo encontrado no HTML de {law.url}")
            records += law_records
            summaries.append(
                LawSummary(
                    law=law.id,
                    articles=len(law_records),
                    units=sum(len(r.units) for r in law_records),
                    by_status=dict(Counter(r.status for r in law_records)),
                    source_hash=doc.sha256,
                )
            )
    finally:
        if owned_client and http is not None:
            http.close()

    write_jsonl(output, records, [law.id for law in corpus.laws])
    return IngestReport(output=output, laws=summaries, dangling=dangling_references(records))


__all__ = ["IngestReport", "LawSummary", "run_ingest"]
