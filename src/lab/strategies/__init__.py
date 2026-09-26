# -----------------------------------------------------------------------------
# File:     src/lab/strategies/__init__.py
# Purpose:  Question-answering strategies (baseline now; RAG, GraphRAG, fine-tuned later).
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Estratégias de resposta a perguntas sobre legislação."""

from lab.strategies.base import Answer, Strategy, load_prompt, normalize_citations
from lab.strategies.baseline import BaselineStrategy

__all__ = ["Answer", "BaselineStrategy", "Strategy", "load_prompt", "normalize_citations"]
