# -----------------------------------------------------------------------------
# File:     src/lab/llm/client.py
# Purpose:  Single OpenAI-compatible client: retries, timeout, cost tracking, cache.
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Cliente único para todos os provedores OpenAI-compatible.

Toda chamada a um modelo no laboratório passa por :class:LLMClient,
configurado por *papéis* de modelo (generator, judge, embedding,
synthesizer) definidos em config/models.yaml (spec, seção 4.1). Trocar
de provedor exige apenas mudar a configuração, nunca o código.

Responsabilidades do módulo:

- Retentativas com backoff exponencial para erros 429 e 5xx; erros 4xx
  (exceto 429) são fatais e não são repetidos.
- Timeout configurável por papel (Role.timeout_s).
- Registro de custo: cada resposta carrega Usage com tokens e custo
  estimado em US$; o acumulador total_cost_usd dispara
  :class:BudgetExceededError ao ultrapassar o limite do run
  (LAB_MAX_USD_PER_RUN).
- Cache em disco (SQLite, :class:lab.llm.cache.SQLiteCache) chaveado por
  hash de (modelo, mensagens, parâmetros). Ligado por padrão no CLI e no
  runner de avaliação, o que torna reruns gratuitos e reproduzíveis.
- Saída estruturada via response_format com JSON Schema
  (:meth:LLMClient.chat_json), com fallback para parsing validado por
  Pydantic quando o provedor não suporta response_format.

Segredos (chaves de API, URLs) vêm do ambiente, nunca da configuração;
erros de configuração nomeiam a *variável* ausente, sem expor valores.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    DefaultHttpxClient,
    OpenAI,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from lab.llm.cache import SQLiteCache, make_key
from lab.llm.costs import Usage, estimate_cost
from lab.settings import ModelsConfig, Role

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class LLMConfigError(RuntimeError):
    """Missing or invalid provider/role configuration. Never includes secret values."""


class BudgetExceededError(RuntimeError):
    """Estimated spend of this run went past the configured limit."""


class ChatResult(BaseModel):
    text: str
    usage: Usage
    cached: bool = False
    latency_ms: int = 0


@dataclass(frozen=True)
class Parsed[T: BaseModel]:
    """Resposta estruturada: ``value`` é ``None`` se não validou (``error`` diz por quê)."""

    value: T | None
    result: ChatResult
    error: str | None = None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):  # includes timeouts
        return True
    return isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)


class LLMClient:
    """Cliente OpenAI-compatible com papéis, retentativas, custo e cache.

    Um cliente por run. Instancie com o :class:ModelsConfig carregado de
    config/models.yaml; as chaves de API são lidas do ambiente
    (env) sob as variáveis nomeadas por cada provedor.

    :param config: papéis e provedores, de config/models.yaml.
    :param cache: cache em disco opcional; None desativa (modo sem cache).
    :param max_usd: limite de custo acumulado do run, em US$.
    :param max_retries: tentativas extras para erros 429/5xx (0 = nenhuma).
    :param backoff_base_s: base do backoff exponencial em segundos
        (tentativa *n* aguarda backoff_base_s * 2n).
    :param env: mapeamento de variáveis de ambiente; padrão os.environ.
        Injetável para testes sem tocar no ambiente real.
    :param http_client: client httpx injetável para testes (MockTransport).
    :param sleep: função de espera injetável; testes passam um no-op.
    """

    def __init__(
        self,
        config: ModelsConfig,
        *,
        cache: SQLiteCache | None = None,
        max_usd: float = 2.00,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        env: Mapping[str, str] | None = None,
        http_client: DefaultHttpxClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._cache = cache
        self._max_usd = max_usd
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._env = env if env is not None else os.environ
        self._http_client = http_client
        self._sleep = sleep
        self._sdk: dict[str, OpenAI] = {}
        self.total_cost_usd = 0.0

    # -- internals ---------------------------------------------------------
    def _role(self, name: str) -> Role:
        try:
            return self._config.roles[name]
        except KeyError:
            raise LLMConfigError(f"unknown role: {name!r}") from None

    def _sdk_for(self, role: Role) -> OpenAI:
        if role.provider not in self._sdk:
            provider = self._config.providers.get(role.provider)
            if provider is None:
                raise LLMConfigError(f"unknown provider: {role.provider!r}")
            api_key = self._env.get(provider.api_key_env)
            if not api_key:
                raise LLMConfigError(f"environment variable {provider.api_key_env} is not set")
            base_url = provider.base_url
            if provider.base_url_env:
                base_url = self._env.get(provider.base_url_env) or base_url
            if not base_url:
                raise LLMConfigError(f"environment variable {provider.base_url_env} is not set")
            self._sdk[role.provider] = OpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,  # retries are handled below, with explicit backoff
                http_client=self._http_client,
            )
        return self._sdk[role.provider]

    def _with_retry(self, fn: Callable[[], T]) -> T:
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt == self._max_retries or not _is_retryable(exc):
                    raise
                self._sleep(self._backoff_base_s * 2**attempt)
        raise AssertionError("unreachable")

    def _register_cost(self, usage: Usage) -> None:
        self.total_cost_usd += usage.cost_usd
        if self.total_cost_usd > self._max_usd:
            raise BudgetExceededError(
                f"estimated cost ${self.total_cost_usd:.4f} exceeds limit ${self._max_usd:.2f}"
            )

    # -- public API --------------------------------------------------------
    def chat(
        self,
        role_name: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatResult:
        """Enviar uma conversa a um papel de modelo e devolver texto, tokens e custo.

        :param role_name: nome do papel em models.yaml (ex.: "generator").
        :param messages: mensagens no formato de chat da API OpenAI.
        :param response_format: formato de resposta opcional (ex.: JSON Schema);
            quando informado, entra na chave do cache.
        :param params: parâmetros extras repassados à API
            (temperature, max_tokens, ...). Entram na chave do cache.
        :return: :class:ChatResult com texto, :class:Usage e latência.
            cached=True indica resposta vinda do cache em disco.
        :raises LLMConfigError: papel ou provedor desconhecido, ou variável de
            ambiente ausente (o erro nomeia a variável, não o valor).
        :raises BudgetExceededError: custo acumulado do run passou do limite.
        """
        role = self._role(role_name)
        key = make_key(role.model, messages, {**params, "response_format": response_format})

        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            return ChatResult(
                text=hit["text"], usage=Usage.model_validate(hit["usage"]), cached=True
            )

        sdk = self._sdk_for(role).with_options(timeout=role.timeout_s)
        kwargs: dict[str, Any] = {"model": role.model, "messages": messages, **params}
        if response_format is not None:
            kwargs["response_format"] = response_format

        started = time.perf_counter()

        def call() -> Any:
            return sdk.chat.completions.create(**kwargs)  # pyright: ignore[reportUnknownVariableType]  (SDK overload typing)

        response = cast(ChatCompletion, self._with_retry(call))
        latency_ms = int((time.perf_counter() - started) * 1000)

        text = response.choices[0].message.content or ""
        raw = response.usage
        usage = estimate_cost(
            role,
            raw.prompt_tokens if raw else 0,
            raw.completion_tokens if raw else 0,
        )
        if self._cache is not None:
            self._cache.set(key, {"text": text, "usage": usage.model_dump()})
        self._register_cost(usage)
        return ChatResult(text=text, usage=usage, latency_ms=latency_ms)

    def chat_parsed(
        self, role_name: str, messages: list[dict[str, Any]], schema: type[M], **params: Any
    ) -> Parsed[M]:
        """Como :meth:`chat_json`, mas sem levantar quando a resposta não valida.

        Devolve sempre o :class:`ChatResult` (texto, tokens, custo, latência),
        para quem mede o experimento não perder o custo de uma resposta ruim.

        :param role_name: nome do papel em models.yaml.
        :param messages: mensagens no formato de chat da API OpenAI.
        :param schema: classe Pydantic que define o schema JSON esperado.
        :param params: parâmetros extras repassados a :meth:`chat`.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        try:
            result = self.chat(role_name, messages, response_format=response_format, **params)
        except BadRequestError:  # provider does not support response_format
            result = self.chat(role_name, messages, **params)
        text = result.text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        try:
            return Parsed(value=schema.model_validate_json(text), result=result)
        except ValidationError as exc:
            return Parsed(value=None, result=result, error=str(exc.errors()[0]["msg"]))

    def chat_json(
        self, role_name: str, messages: list[dict[str, Any]], schema: type[M], **params: Any
    ) -> M:
        """Saída estruturada via JSON Schema, com fallback para parsing validado.

        :param role_name: nome do papel em models.yaml.
        :param messages: mensagens no formato de chat da API OpenAI.
        :param schema: classe Pydantic que define o schema JSON esperado.
        :param params: parâmetros extras repassados a :meth:`chat`.
        :return: instância validada do ``schema``.
        :raises ValueError: a resposta não valida contra o schema.
        """
        parsed = self.chat_parsed(role_name, messages, schema, **params)
        if parsed.value is None:
            raise ValueError(f"resposta fora do schema {schema.__name__}: {parsed.error}")
        return parsed.value
