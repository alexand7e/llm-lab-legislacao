# -----------------------------------------------------------------------------
# File:     src/lab/settings.py
# Purpose:  Runtime settings (env) and typed loader for config/models.yaml.
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Configuração de runtime e tipos da configuração de modelos.

Este módulo tem duas metades, com origens diferentes de valores:

- :class:`Settings` — variáveis de ambiente (``.env`` ou ambiente do
  sistema), via ``pydantic-settings``: limites de custo, caminhos de cache
  e do arquivo de modelos.
- :class:`ModelsConfig` e cia. — a configuração tipada de
  ``config/models.yaml`` (papéis e provedores), carregada por
  :func:`load_models_config` e validada por Pydantic.

Princípio de segredos: nenhum valor sensível (chave de API, URL privada)
vive em arquivo versionado. Provedores podem apontar para variáveis de
ambiente (``base_url_env`` / ``api_key_env``) em vez de literal — é o caso
do provedor ``primary`` em ``config/models.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variáveis de ambiente do laboratório (``.env`` ou ambiente do sistema).

    - ``lab_max_usd_per_run``: limite de gasto acumulado por run, em US$
      (o cliente aborta com ``BudgetExceededError`` ao ultrapassá-lo).
    - ``lab_cache_dir``: diretório do cache SQLite de respostas.
    - ``lab_models_config``: caminho de ``models.yaml``.
    - ``lab_corpus_config``: caminho de ``corpus.yaml`` (normas do corpus).

    ``extra="ignore"``: variáveis não conhecidas (chaves de provedores,
    Qdrant etc.) não causam erro.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    lab_max_usd_per_run: float = 2.00
    lab_cache_dir: Path = Path(".cache/llm")
    lab_models_config: Path = Path("config/models.yaml")
    lab_corpus_config: Path = Path("config/corpus.yaml")


class Provider(BaseModel):
    """Um provedor OpenAI-compatible.

    - ``base_url``: URL literal do endpoint (ex.: ``https://openrouter.ai/api/v1``).
    - ``base_url_env``: nome da variável de ambiente que guarda a URL.
      Usado quando a URL não pode/versionar bem (provedor ``primary``).
      ``base_url`` e ``base_url_env`` são exclusivos; o validador exige
      exatamente um.
    - ``api_key_env``: nome da variável de ambiente que guarda a chave.
      Sempre por ambiente — nunca literal.
    """

    base_url: str | None = None
    base_url_env: str | None = None  # env var holding the URL (keeps it out of git)
    api_key_env: str

    @model_validator(mode="after")
    def _has_one_base_url(self) -> Self:
        if bool(self.base_url) == bool(self.base_url_env):
            raise ValueError("provider needs exactly one of base_url or base_url_env")
        return self


class Role(BaseModel):
    """Um papel de modelo (``generator``, ``judge``, ``embedding``, ...).

    - ``provider``: chave do provedor em ``ModelsConfig.providers``.
    - ``model``: identificador do modelo no provedor.
    - ``price_in`` / ``price_out``: US$ por 1M de tokens de entrada/saída,
      preenchidos manualmente a partir da página de preços do provedor.
    - ``dims``: dimensão dos vetores (somente papel ``embedding``).
    - ``timeout_s``: timeout da chamada, em segundos.
    """

    provider: str
    model: str
    price_in: float = 0.0  # valor por 1M input tokens
    price_out: float = 0.0  # valor por 1M output tokens
    dims: int | None = None
    timeout_s: float = Field(default=60.0, gt=0)


class ModelsConfig(BaseModel):
    """Conteúdo validado de ``config/models.yaml``: papéis + provedores."""

    roles: dict[str, Role]
    providers: dict[str, Provider]


def load_env(env_file: Path = Path(".env")) -> dict[str, str]:
    """Variáveis de ambiente para os provedores: ``.env`` com o ambiente por cima.

    :class:`Settings` só enxerga os campos que declara e ignora o resto, então
    as chaves dos provedores (``PRIMARY_API_KEY`` etc.) não chegariam ao
    cliente vindas só do ``.env``. O ambiente do sistema tem precedência
    (é o que o CI usa, via secrets); valores vazios do ``.env`` contam como
    ausentes.
    """
    from_file = {k: v for k, v in dotenv_values(env_file).items() if v}
    return {**from_file, **os.environ}


def load_models_config(path: Path) -> ModelsConfig:
    """Carregar e validar ``models.yaml`` em :class:`ModelsConfig`.

    :raises pydantic.ValidationError: arquivo ausente, YAML inválido ou
        campos faltando/errados (a mensagem indica o campo e o arquivo).
    """
    with path.open(encoding="utf-8") as fh:
        return ModelsConfig.model_validate(yaml.safe_load(fh))
