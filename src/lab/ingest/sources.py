# -----------------------------------------------------------------------------
# File:     src/lab/ingest/sources.py
# Purpose:  Registry of the laws in the corpus and their official URLs.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Leis do corpus (SPEC 2.1) e a URL do texto compilado no Planalto.

O ``id`` de cada lei é o prefixo dos identificadores em ``articles.jsonl``
(ex.: ``lgpd:art:7``). Ampliar o corpus é acrescentar uma entrada aqui, via
issue própria.
"""

from __future__ import annotations

from pydantic import BaseModel


class LawSource(BaseModel):
    """Uma lei do corpus.

    - ``id``: prefixo estável dos identificadores (``lgpd``, ``cdc``...).
    - ``name`` / ``number``: nome e número oficiais, para relatórios.
    - ``url``: página do texto compilado no Planalto (quando existe;
      o Marco Civil só tem a versão original, com alterações anotadas).
    """

    id: str
    name: str
    number: str
    url: str


LAWS: dict[str, LawSource] = {
    law.id: law
    for law in (
        LawSource(
            id="lgpd",
            name="Lei Geral de Proteção de Dados",
            number="Lei 13.709/2018",
            url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm",
        ),
        LawSource(
            id="marco_civil",
            name="Marco Civil da Internet",
            number="Lei 12.965/2014",
            url="https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm",
        ),
        LawSource(
            id="cdc",
            name="Código de Defesa do Consumidor",
            number="Lei 8.078/1990",
            url="https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
        ),
    )
}
