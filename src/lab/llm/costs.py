# -----------------------------------------------------------------------------
# File:     src/lab/llm/costs.py
# Purpose:  Token usage and cost estimation for a model role.
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Uso de tokens e estimativa de custo por papel de modelo.

O custo é estimado localmente a partir dos preços informados em
``models.yaml`` (US$ por 1M de tokens), não do campo ``total_cost``
da resposta — alguns provedores não o enviam ou o calculam com
descontos/promoções que distorcem a comparação entre runs.
"""

from __future__ import annotations

from pydantic import BaseModel

from lab.settings import Role


class Usage(BaseModel):
    """Tokens de uma chamada e o custo estimado em US$."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def estimate_cost(role: Role, prompt_tokens: int, completion_tokens: int) -> Usage:
    """Estimar o custo de uma chamada com os preços do papel.

    ``cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000``.
    """
    cost = (prompt_tokens * role.price_in + completion_tokens * role.price_out) / 1_000_000
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost)
