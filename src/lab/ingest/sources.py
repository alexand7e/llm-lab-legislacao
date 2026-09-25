# -----------------------------------------------------------------------------
# File:     src/lab/ingest/sources.py
# Purpose:  Typed loader for config/corpus.yaml (the norms in the corpus).
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Normas do corpus, lidas de ``config/corpus.yaml``.

O corpus é configuração, não código: acrescentar ou remover uma norma é
editar o YAML. O ``id`` de cada norma é o prefixo dos identificadores em
``articles.jsonl`` (ex.: ``<id>:art:7``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator


class LawSource(BaseModel):
    """Uma norma do corpus.

    - ``id``: prefixo estável dos identificadores (minúsculas, dígitos, ``_``).
    - ``name`` / ``number``: nome e número oficiais, para relatórios.
    - ``url``: página do texto (de preferência a versão compilada).
    """

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    number: str
    url: str


class CorpusConfig(BaseModel):
    """Conteúdo validado de ``config/corpus.yaml``."""

    laws: list[LawSource] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [law.id for law in self.laws]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(f"duplicated law ids: {', '.join(duplicated)}")
        return self

    def by_id(self) -> dict[str, LawSource]:
        return {law.id: law for law in self.laws}


def load_corpus(path: Path) -> CorpusConfig:
    """Carregar e validar ``corpus.yaml``.

    :raises pydantic.ValidationError: campos faltando/errados ou ids repetidos.
    """
    with path.open(encoding="utf-8") as fh:
        return CorpusConfig.model_validate(yaml.safe_load(fh))
