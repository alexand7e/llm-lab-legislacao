# -----------------------------------------------------------------------------
# File:     src/lab/ingest/parser.py
# Purpose:  Parse the HTML of a Brazilian norm into articles and their units.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Parser de normas no formato das páginas de legislação do Planalto.

O parser não conhece nenhuma norma específica: trata a estrutura comum às
leis brasileiras (SPEC 2.2) em três passos.

1. **Linhas** — um parágrafo ``<p>`` vira uma linha de texto. Redações
   antigas marcadas como riscadas (``<strike>``, ``<s>``, ``<del>``,
   ``text-decoration: line-through``) são descartadas, e anotações de
   alteração ("(Redação dada pela Lei ...)", "(Incluído pela ...)",
   "Vigência" ...) são removidas do texto limpo.
2. **Classificação** — cada linha é título (Livro, Parte, Título, Capítulo,
   Seção, Subseção), artigo, parágrafo, inciso ou alínea. Linhas não
   reconhecidas dentro de um artigo continuam o dispositivo anterior (ex.:
   "Pena - ...", ou texto entre aspas que altera outra lei). O preâmbulo
   antes do primeiro artigo e a assinatura final são ignorados.
3. **Hierarquia** — títulos atualizam o caminho (``path``); dentro do
   artigo, incisos pertencem ao caput ou ao último parágrafo, e alíneas ao
   último inciso.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# -- linhas -------------------------------------------------------------------

_STRUCK_TAGS = ("strike", "s", "del")
_LINE_THROUGH = re.compile(r"line-through", re.IGNORECASE)

# Anotações de alteração entre parênteses; podem conter um nível de parênteses.
_ANNOTATION = re.compile(
    r"\(\s*(?:Reda[çc][ãa]o dada|Inclu[íi]d[oa]|Acrescid[oa]|Acrescentad[oa]|Renumerad[oa]"
    r"|Revogad[oa]s? pel[oa]|Vetad[oa] pel[oa]|Vide\b|Vig[êe]ncia|Regulamento|Promulga"
    r"|Produ[çc][ãa]o de efeito|Rejeitad[oa]|Convertid[oa]|Retificad[oa]|Mantid[oa]"
    r"|Suspens[oa]|Declarad[oa])"
    r"[^()]*(?:\([^()]*\)[^()]*)*\)",
    re.IGNORECASE,
)
_TRAILING_NOTE = re.compile(r"(?:\s+(?:Vig[êe]ncia|Regulamento|Mensagem de veto))+\s*$")
_SIGNATURE = re.compile(r"^(?:Bras[íi]lia\s*,|Este texto não substitui)", re.IGNORECASE)


@dataclass(frozen=True)
class Line:
    """Uma linha do documento: ``raw`` com anotações, ``text`` limpo."""

    raw: str
    text: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def clean(raw: str) -> str:
    """Remover anotações de alteração e normalizar espaços."""
    text = raw
    while True:
        stripped = _ANNOTATION.sub("", text)
        if stripped == text:
            break
        text = stripped
    text = _TRAILING_NOTE.sub("", _normalize(text))
    return re.sub(r"\s+([.,;:])", r"\1", text).strip()


def extract_lines(html: str) -> list[Line]:
    """Linhas de texto do documento, sem redações riscadas.

    Linhas que ficam vazias depois de limpas (só anotação) são descartadas.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_STRUCK_TAGS):
        tag.decompose()
    for tag in soup.find_all(style=_LINE_THROUGH):
        tag.decompose()
    lines: list[Line] = []
    for p in soup.find_all("p"):
        raw = _normalize(p.get_text(" "))
        text = clean(raw)
        if text:
            lines.append(Line(raw=raw, text=text))
    return lines


# -- classificação ------------------------------------------------------------

_ROMAN = r"[IVXLCDM]+"
_ORD = r"(?:\s*[º°]|o)?"

HEADING_LEVELS = ("livro", "parte", "titulo", "capitulo", "secao", "subsecao")
_HEADING_LABELS = {
    "livro": "Livro",
    "parte": "Parte",
    "titulo": "Título",
    "capitulo": "Capítulo",
    "secao": "Seção",
    "subsecao": "Subseção",
}
# Palavra-chave em qualquer caixa; numeral romano só em maiúsculas (evita "Parte civil").
_HEADING = re.compile(
    r"^(?P<kind>(?i:LIVRO|PARTE|T[ÍI]TULO|CAP[ÍI]TULO|SUBSE[ÇC][ÃA]O|SE[ÇC][ÃA]O))\s+"
    rf"(?P<num>{_ROMAN}(?:\s*-\s*[A-Z])?|(?i:[ÚU]NIC[OA]))\b\.?\s*(?P<name>.*)$"
)
_SUFFIX = r"(?:\s*-\s*(?P<suf>[A-Z]{1,3})\b)?"
_ARTICLE = re.compile(
    rf"^Art\.?\s*(?P<num>\d+(?:\s\d+)?){_ORD}{_SUFFIX}\s*[.\-–—]?\s*(?P<body>.*)$"
)
_PARAGRAPH = re.compile(
    rf"^(?:§\s*(?P<num>\d+){_ORD}{_SUFFIX}|(?P<unico>(?i:Par[áa]grafo [úu]nico)))"
    r"\s*[.:\-–—]?\s*(?P<body>.*)$"
)
_INCISO = re.compile(rf"^(?P<num>{_ROMAN}){_SUFFIX}\s*[-–—]\s*(?P<body>.*)$")
_ALINEA = re.compile(r"^(?P<num>[a-z])(?:\s*-\s*(?P<suf>[A-Z]{1,3}))?\s*\)\s*(?P<body>.*)$")


def _ident(num: str, suffix: str | None) -> str:
    base = num.replace(" ", "")
    return f"{base}-{suffix.upper()}" if suffix else base


_ARTICLE_ID = re.compile(r"^(\d+)(?:-([A-Z]+))?$")


def article_key(ident: str) -> tuple[int, int, str]:
    """Chave de ordenação natural de artigos.

    Artigos acrescidos são artigos próprios, entre o número base e o
    seguinte: ``"2" < "10" < "55" < "55-A" < "55-Z" < "55-AA" < "56"``.

    :raises ValueError: identificador fora do formato ``<número>[-<letras>]``.
    """
    m = _ARTICLE_ID.match(ident)
    if m is None:
        raise ValueError(f"invalid article id: {ident!r}")
    suffix = m[2] or ""
    return int(m[1]), len(suffix), suffix


def _heading_kind(word: str) -> str:
    return word.lower().translate(str.maketrans("íçã", "ica"))


# -- status -------------------------------------------------------------------

Status = Literal["vigente", "revogado", "vetado"]

_REVOKED = re.compile(r"^\(?\s*revogad[oa]s?\b", re.IGNORECASE)
_VETOED = re.compile(r"^\(?\s*vetad[oa]s?\b", re.IGNORECASE)


def status_of(body: str, raw: str) -> Status:
    """Status de um dispositivo a partir do texto após o rótulo.

    ``body`` é o texto limpo ("(Revogado).", "(VETADO).", ...). Quando a
    linha era só a anotação ("Art. 9º (Revogado pela Lei nº ...)"), o texto
    limpo fica vazio e o status vem da linha original ``raw``.
    """
    if _REVOKED.match(body):
        return "revogado"
    if _VETOED.match(body):
        return "vetado"
    if not body:
        if re.search(r"\brevogad", raw, re.IGNORECASE):
            return "revogado"
        if re.search(r"\bvetad", raw, re.IGNORECASE):
            return "vetado"
    return "vigente"


# -- modelo -------------------------------------------------------------------


class Heading(BaseModel):
    """Um título estrutural: ``label`` (ex.: "Capítulo II") e o nome dele."""

    level: str
    label: str
    name: str = ""


class Unit(BaseModel):
    """Dispositivo dentro do artigo: parágrafo, inciso ou alínea.

    - ``id``: rótulo do dispositivo ("1", "unico", "I", "V-A", "a").
    - ``key``: caminho único no artigo ("§1", "I", "§3.I", "I.a").
    - ``parent``: ``key`` do dispositivo pai; ``None`` = caput.
    - ``text``: texto limpo, sem o rótulo.
    - ``status``: ``vigente``, ``revogado`` ou ``vetado``. Há dispositivos
      revogados dentro de artigos vigentes, por isso o status é por unidade.
    - ``raw``: linha original (com anotações), usada para status e remissões.
    """

    type: str
    id: str
    key: str
    parent: str | None = None
    text: str
    status: Status = "vigente"
    raw: str = Field(default="", exclude=True)


class Article(BaseModel):
    """Um artigo da norma com sua posição na hierarquia e seus dispositivos.

    ``status`` é o do caput: um artigo revogado ou vetado por inteiro.
    """

    law: str
    article: str
    path: list[str]
    headings: list[Heading]
    caput: str
    units: list[Unit]
    status: Status = "vigente"
    raw: str = Field(default="", exclude=True)

    @property
    def id(self) -> str:
        return f"{self.law}:art:{self.article}"

    @property
    def text(self) -> str:
        """Texto do artigo com os rótulos, um dispositivo por linha.

        Só dispositivos vigentes entram; revogados e vetados ficam em ``units``.
        """
        lines = [f"{_ordinal('Art.', self.article)} {self.caput}".rstrip()]
        for unit in self.units:
            if unit.status != "vigente":
                continue
            lines.append(f"{_unit_label(unit)} {unit.text}".rstrip())
        return "\n".join(lines)


def _ordinal(prefix: str, ident: str) -> str:
    """Rótulo na convenção legal: ordinal até 9 ("Art. 7º"), cardinal depois ("Art. 10.")."""
    if ident.isdigit() and int(ident) < 10:
        return f"{prefix} {ident}º"
    return f"{prefix} {ident}."


def _unit_label(unit: Unit) -> str:
    if unit.type == "paragrafo":
        return "Parágrafo único." if unit.id == "unico" else _ordinal("§", unit.id)
    if unit.type == "inciso":
        return f"{unit.id} -"
    return f"{unit.id})"


# -- montagem -----------------------------------------------------------------


class _Builder:
    def __init__(self, law: str) -> None:
        self.law = law
        self.articles: list[Article] = []
        self.headings: list[Heading] = []
        self.pending_name: Heading | None = None
        self.current: Article | None = None
        self.last_unit: Unit | None = None
        self.paragraph: str | None = None
        self.inciso: str | None = None

    # títulos
    def heading(self, kind: str, num: str, name: str) -> None:
        level = _heading_kind(kind)
        rank = HEADING_LEVELS.index(level)
        self.headings = [h for h in self.headings if HEADING_LEVELS.index(h.level) < rank]
        numeral = re.sub(r"\s+", "", num).upper()
        heading = Heading(level=level, label=f"{_HEADING_LABELS[level]} {numeral}", name=name)
        self.headings.append(heading)
        self.pending_name = None if name else heading
        self.finish()

    # artigos e dispositivos
    def article(self, number: str, body: str, line: Line) -> None:
        self.finish()
        self.current = Article(
            law=self.law,
            article=number,
            path=[h.label for h in self.headings],
            headings=[h.model_copy() for h in self.headings],
            caput=body,
            units=[],
            status=status_of(body, line.raw),
            raw=line.raw,
        )
        self.pending_name = None

    def unit(self, type_: str, ident: str, body: str, line: Line) -> None:
        assert self.current is not None
        if type_ == "paragrafo":
            parent, key = None, f"§{ident}"
            self.paragraph, self.inciso = key, None
        elif type_ == "inciso":
            parent = self.paragraph
            key = f"{parent}.{ident}" if parent else ident
            self.inciso = key
        else:
            parent = self.inciso or self.paragraph
            key = f"{parent}.{ident}" if parent else ident
        unit = Unit(
            type=type_,
            id=ident,
            key=key,
            parent=parent,
            text=body,
            status=status_of(body, line.raw),
            raw=line.raw,
        )
        self.current.units.append(unit)
        self.last_unit = unit

    def continuation(self, line: Line) -> None:
        if self.pending_name is not None:
            self.pending_name.name = line.text
            self.pending_name = None
            return
        if self.current is None:
            return  # preâmbulo
        if self.last_unit is not None:
            self.last_unit.text = f"{self.last_unit.text}\n{line.text}".strip()
            self.last_unit.raw = f"{self.last_unit.raw}\n{line.raw}"
        else:
            self.current.caput = f"{self.current.caput}\n{line.text}".strip()
            self.current.raw = f"{self.current.raw}\n{line.raw}"

    def finish(self) -> None:
        """Fechar o artigo corrente, se houver."""
        if self.current is not None:
            self.articles.append(self.current)
        self.current, self.last_unit, self.paragraph, self.inciso = None, None, None, None


def parse_lines(lines: list[Line], law: str) -> list[Article]:
    """Montar os artigos a partir das linhas já extraídas."""
    b = _Builder(law)
    for line in lines:
        text = line.text
        if _SIGNATURE.match(text):
            break
        if text[0] in "\"“”'‘’":  # texto citado (ex.: alteração de outra lei)
            b.continuation(line)
        elif m := _HEADING.match(text):
            b.heading(m["kind"], m["num"], m["name"])
        elif m := _ARTICLE.match(text):
            b.article(_ident(m["num"], m["suf"]), m["body"], line)
        elif b.current is None:
            b.continuation(line)
        elif m := _PARAGRAPH.match(text):
            ident = "unico" if m["unico"] else _ident(m["num"], m["suf"])
            b.unit("paragrafo", ident, m["body"], line)
        elif m := _INCISO.match(text):
            b.unit("inciso", _ident(m["num"], m["suf"]), m["body"], line)
        elif m := _ALINEA.match(text):
            b.unit("alinea", _ident(m["num"], m["suf"]), m["body"], line)
        else:
            b.continuation(line)
    b.finish()
    return b.articles


def parse_html(html: str, law: str) -> list[Article]:
    """Parsear o HTML de uma norma em artigos, na ordem do documento."""
    return parse_lines(extract_lines(html), law)


__all__ = [
    "Article",
    "Heading",
    "Line",
    "Status",
    "Unit",
    "article_key",
    "clean",
    "extract_lines",
    "parse_html",
    "parse_lines",
    "status_of",
]
