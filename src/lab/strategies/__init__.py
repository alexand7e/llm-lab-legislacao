# -----------------------------------------------------------------------------
# File:     src/lab/strategies/__init__.py
# Purpose:  Question-answering strategies (baseline now; RAG, GraphRAG, fine-tuned later).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Estratégias de resposta a perguntas sobre legislação."""

from lab.strategies.base import (
    Answer,
    Message,
    Strategy,
    StreamChunk,
    StreamingStrategy,
    load_prompt,
    normalize_citations,
)
from lab.strategies.baseline import BaselineStrategy

__all__ = [
    "Answer",
    "BaselineStrategy",
    "Message",
    "StreamChunk",
    "Strategy",
    "StreamingStrategy",
    "load_prompt",
    "normalize_citations",
]
