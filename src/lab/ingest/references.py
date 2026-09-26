# -----------------------------------------------------------------------------
# File:     src/lab/ingest/references.py
# Purpose:  Extract internal and external cross-references from parsed articles.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Remissões entre dispositivos (SPEC 2.2), como dados estruturados.

Uma remissão liga um artigo a outro: ``{"target": "lgpd:art:11", "raw": "art. 11"}``.
Elas alimentam o GraphRAG (M6).

- **Internas** — "art. 7º", "arts. 7º e 11", "art. 55-A", "art. 7º desta Lei".
  O alvo é ``<lei>:art:<n>`` da própria norma.
- **Externas** — "art. 5º da Lei nº 8.078", "Lei nº 12.965, de 2014". Se a
  norma citada está no corpus (``known``), o alvo usa o id do corpus
  (``cdc:art:5``, ``cdc``); senão, uma chave estável (``lei:8078:art:5``).

Intervalos ("arts. 7º a 11", "55-A a 55-C") são expandidos, até 50 artigos.

Não entram como remissão: anotações de alteração ("Redação dada pela Lei ...",
já removidas do texto limpo), texto entre aspas (que reproduz outra lei) e
dispositivos revogados ou vetados.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from lab.ingest.parser import Article
from lab.ingest.sources import CorpusConfig


class Reference(BaseModel):
    """Remissão: ``target`` estruturado e ``raw``, o trecho como no texto."""

    target: str
    raw: str


# -- normas citadas -----------------------------------------------------------

_KINDS = {
    "lei complementar": "lei-complementar",
    "lei": "lei",
    "decreto-lei": "decreto-lei",
    "decreto": "decreto",
    "medida provisória": "mp",
    "medida provisoria": "mp",
}
_NAMED = {
    "constituição federal": "cf",
    "constituicao federal": "cf",
    "constituição": "cf",
    "constituicao": "cf",
    "código civil": "cc",
    "codigo civil": "cc",
    "código penal": "cp",
    "codigo penal": "cp",
    "código de processo civil": "cpc",
    "codigo de processo civil": "cpc",
    "código de processo penal": "cpp",
    "codigo de processo penal": "cpp",
    "código tributário nacional": "ctn",
    "codigo tributario nacional": "ctn",
    "código de defesa do consumidor": "lei:8078",
    "codigo de defesa do consumidor": "lei:8078",
}
_NORM = (
    r"(?:(?P<kind>Lei\s+Complementar|Lei|Decreto-Lei|Decreto|Medida\s+Provis[óo]ria)"
    r"\s+n[º°o.]*\s*(?P<num>\d+(?:\.\d{3})*(?:-\d+)?)"
    r"|(?P<named>Constitui[çc][ãa]o(?:\s+Federal)?|C[óo]digo\s+Civil|C[óo]digo\s+Penal"
    r"|C[óo]digo\s+de\s+Processo\s+(?:Civil|Penal)|C[óo]digo\s+Tribut[áa]rio\s+Nacional"
    r"|C[óo]digo\s+de\s+Defesa\s+do\s+Consumidor))"
)
_NORM_RE = re.compile(_NORM, re.IGNORECASE)


def _norm_key(m: re.Match[str]) -> str:
    """Chave estável da norma citada: ``lei:8078``, ``decreto:8771``, ``cf``."""
    if m["named"]:
        return _NAMED[re.sub(r"\s+", " ", m["named"].lower())]
    kind = _KINDS[re.sub(r"\s+", " ", m["kind"].lower())]
    return f"{kind}:{m['num'].replace('.', '')}"


def known_laws(corpus: CorpusConfig) -> dict[str, str]:
    """Mapa ``chave da norma -> id do corpus`` (ex.: ``lei:8078 -> cdc``)."""
    known: dict[str, str] = {}
    for law in corpus.laws:
        m = re.match(r"\s*Lei\s+(?P<num>\d+(?:\.\d{3})*)", law.number, re.IGNORECASE)
        if m:
            known[f"lei:{m['num'].replace('.', '')}"] = law.id
    return known


# -- artigos citados ----------------------------------------------------------

_TOKEN = r"\d+\s*[º°]?(?:\s*-\s*[A-Z]{1,3}\b)?"
_ART_GROUP = re.compile(
    rf"\barts?\.?\s*(?P<list>{_TOKEN}(?:\s*(?:,|e|ou|a|à)\s*{_TOKEN})*)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"(?:(?P<sep>,|e|ou|a|à)\s*)?(?P<n>\d+)\s*[º°]?(?:\s*-\s*(?P<s>[A-Z]{1,3})\b)?",
    re.IGNORECASE,
)
_MAX_RANGE = 50  # "arts. 1º a 500" não vira 500 remissões
_QUALIFIER = (
    r"(?:[,\s]|caput|par[áa]grafo\s+[úu]nico|§\s*\d+[º°]?|incisos?\s+[IVXLCDM]+"
    r"|al[íi]neas?\s+\S{1,4}|itens?\s+\d+|e|ou)*"
)
_TAIL = re.compile(rf"^{_QUALIFIER}\bd[aoe]s?\s+{_NORM}", re.IGNORECASE)


def _ident(m: re.Match[str]) -> str:
    return f"{m['n']}-{m['s'].upper()}" if m["s"] else m["n"]


def _idents(article_list: str) -> list[str]:
    """Artigos de uma lista ("7º, 9º e 11", "7º a 11", "55-A a 55-C"), com intervalos expandidos."""
    idents: list[str] = []
    prev: re.Match[str] | None = None
    for tok in _TOKEN_RE.finditer(article_list):
        ident = _ident(tok)
        is_range = (tok["sep"] or "").lower() in ("a", "à") and prev is not None
        if is_range and prev is not None:
            lo, hi = int(prev["n"]), int(tok["n"])
            if prev["s"] and tok["s"] and lo == hi:
                letters = range(ord(prev["s"].upper()), ord(tok["s"].upper()))
                idents += [f"{lo}-{chr(c)}" for c in letters][1:]
            elif not prev["s"] and not tok["s"] and 0 < hi - lo <= _MAX_RANGE:
                idents += [str(n) for n in range(lo + 1, hi)]
        idents.append(ident)
        prev = tok
    return idents


def _unquoted(text: str) -> str:
    """Texto sem as linhas citadas (aspas), que reproduzem outra lei."""
    return "\n".join(
        ln for ln in text.split("\n") if not ln.lstrip().startswith(('"', "“", "'", "‘"))
    )


def extract_references(
    text: str, law: str, known: dict[str, str] | None = None, own_article: str | None = None
) -> list[Reference]:
    """Remissões de um trecho de texto legal, sem repetição, na ordem do texto."""
    known = known or {}
    text = _unquoted(text)
    found: dict[tuple[str, str], Reference] = {}

    def add(target: str, raw: str) -> None:
        found.setdefault((target, raw), Reference(target=target, raw=raw))

    consumed: list[tuple[int, int]] = []
    for m in _ART_GROUP.finditer(text):
        tail = _TAIL.match(text[m.end() : m.end() + 160])
        prefix = law
        end = m.end()
        if tail:
            key = _norm_key(tail)
            prefix = known.get(key, key)
            end += tail.end()
            consumed.append((m.end() + tail.start("kind" if tail["kind"] else "named"), end))
        raw = re.sub(r"\s+", " ", text[m.start() : end]).strip(" ,")
        if prefix == law and tail is None:
            raw = re.sub(r"\s+", " ", m.group(0)).strip(" ,")
        for ident in _idents(m["list"]):
            if prefix == law and ident == own_article:
                continue
            add(f"{prefix}:art:{ident}", raw)

    for m in _NORM_RE.finditer(text):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        key = _norm_key(m)
        target = known.get(key, key)
        if target != law:
            add(target, re.sub(r"\s+", " ", m.group(0)))
    return list(found.values())


def article_references(article: Article, known: dict[str, str] | None = None) -> list[Reference]:
    """Remissões de um artigo: caput e dispositivos vigentes."""
    parts = [article.caput] if article.status == "vigente" else []
    parts += [u.text for u in article.units if u.status == "vigente"]
    return extract_references("\n".join(parts), article.law, known, article.article)


__all__ = ["Reference", "article_references", "extract_references", "known_laws"]
