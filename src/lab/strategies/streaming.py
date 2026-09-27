# -----------------------------------------------------------------------------
# File:     src/lab/strategies/streaming.py
# Purpose:  Pull the "answer" text out of a JSON object that is still being streamed.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Extração incremental do campo ``answer`` de um JSON em streaming.

O baseline pede ao modelo um objeto JSON (``answer``, ``cited_articles``,
``confidence``). Em streaming o JSON chega em pedaços quebrados em qualquer
ponto, inclusive no meio de um escape (``\\u00e7``, ``\\n``). O
:class:`AnswerExtractor` recebe os pedaços e devolve só o texto novo do campo
``answer``, já decodificado, para a tela mostrar a resposta crescendo.

Se a saída não começar com ``{`` (o modelo ignorou o schema e respondeu em texto
livre), o texto é repassado como está. Quem chama decide o que fazer com o
resultado final; aqui só se extrai o que já dá para mostrar.
"""

from __future__ import annotations

import re

_KEY = re.compile(r'"answer"\s*:\s*"')
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class AnswerExtractor:
    """Alimente com :meth:`feed`; cada chamada devolve o texto novo de ``answer``."""

    def __init__(self) -> None:
        self._buffer = ""
        self._pos = -1  # -1: ainda não achou o início do valor de "answer"
        self._done = False
        self._plain: bool | None = None  # None: ainda não dá para saber se é JSON

    @property
    def is_plain_text(self) -> bool:
        """A saída não é JSON (o modelo respondeu em texto livre)."""
        return self._plain is True

    @property
    def finished(self) -> bool:
        """O valor de ``answer`` já foi fechado (aspas de fechamento vistas)."""
        return self._done

    def feed(self, chunk: str) -> str:
        """Texto novo do campo ``answer`` (``""`` se o pedaço não trouxe nada dele)."""
        if self._done and not self._plain:
            self._buffer += chunk
            return ""
        self._buffer += chunk
        if self._plain is None:
            stripped = self._buffer.lstrip()
            if not stripped:
                return ""
            self._plain = not stripped.startswith("{")
            if self._plain:
                return self._buffer  # texto livre: tudo o que já chegou
        if self._plain:
            return chunk
        return self._decode()

    def _decode(self) -> str:
        if self._pos < 0:
            m = _KEY.search(self._buffer)
            if m is None:
                return ""
            self._pos = m.end()
        out: list[str] = []
        buf, i = self._buffer, self._pos
        while i < len(buf):
            ch = buf[i]
            if ch == '"':
                self._done = True
                i += 1
                break
            if ch != "\\":
                out.append(ch)
                i += 1
                continue
            if i + 1 >= len(buf):
                break  # escape cortado: espera o próximo pedaço
            nxt = buf[i + 1]
            if nxt == "u":
                if i + 6 > len(buf):
                    break
                code = int(buf[i + 2 : i + 6], 16)
                if 0xD800 <= code < 0xDC00:  # par substituto: precisa do \uXXXX seguinte
                    if i + 12 > len(buf):
                        break
                    low = int(buf[i + 8 : i + 12], 16)
                    out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                    i += 12
                    continue
                out.append(chr(code))
                i += 6
                continue
            out.append(_SIMPLE_ESCAPES.get(nxt, nxt))
            i += 2
        self._pos = i
        return "".join(out)


__all__ = ["AnswerExtractor"]
