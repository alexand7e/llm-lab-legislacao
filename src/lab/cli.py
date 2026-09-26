# -----------------------------------------------------------------------------
# File:     src/lab/cli.py
# Purpose:  Command line entry point (lab).
# Author:   Alexandre
# Created:  2026-09-24
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Linha de comando do laboratório (``lab``).

Ponto de entrada único: ``lab <comando>``. Os comandos atuais falam com
providores OpenAI-compatible usando o papel de modelo configurado em
``config/models.yaml``. Erros de configuração (papel/provedor desconhecido,
variável de ambiente ausente) e estouro de orçamento saem no stderr com
código de saída 1, sem imprimir valores de segredos.
"""

from __future__ import annotations

from pathlib import Path

import typer

from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.cache import SQLiteCache
from lab.settings import Settings, load_env, load_models_config

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "llm-lab-legislacao — laboratório de técnicas com LLMs aplicado à "
        "legislação brasileira. Veja `lab chat --help` para o comando de teste manual."
    ),
)


@app.callback()
def main() -> None:
    """llm-lab-legislacao command line."""


@app.command()
def chat(
    prompt: str = typer.Argument(
        ...,
        help=(
            "Mensagem enviada ao modelo (a pergunta em si). "
            "Ex.: lab chat 'Qual o prazo de resposta do controlador na LGPD?'"
        ),
    ),
    role: str = typer.Option(
        "generator",
        "--role",
        "-r",
        help=(
            "Papel de modelo em models.yaml: generator, judge, embedding ou "
            "synthesizer. Cada papel usa o modelo e o provedor configurados para ele."
        ),
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help=(
            "Desliga o cache em disco para esta chamada. Por padrão, repetições da "
            "mesma pergunta com o mesmo modelo/parâmetros saem do cache (gratuitas)."
        ),
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Caminho alternativo para models.yaml. Padrão: LAB_MODELS_CONFIG ou config/models.yaml."
        ),
    ),
) -> None:
    """Testar um papel de modelo: imprime a resposta, tokens e custo.

    Exemplos:

        lab chat "olá"

        lab chat --role judge "esta resposta está correta?"

        lab chat --no-cache "olá"   # força nova chamada, ignora o cache

    A saída da resposta vai para o stdout; tokens, custo e latência vão
    para o stderr (linha cinza no final), para não poluir o texto.
    """
    settings = Settings()
    cache = None if no_cache else SQLiteCache(settings.lab_cache_dir)
    try:
        models = load_models_config(config or settings.lab_models_config)
        client = LLMClient(
            models, cache=cache, max_usd=settings.lab_max_usd_per_run, env=load_env()
        )
        result = client.chat(role, [{"role": "user", "content": prompt}])
    except (LLMConfigError, BudgetExceededError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if cache is not None:
            cache.close()

    typer.echo(result.text)
    u = result.usage
    origin = "cache" if result.cached else f"{result.latency_ms} ms"
    typer.secho(
        f"[{role}] in={u.prompt_tokens} out={u.completion_tokens} "
        f"cost=${u.cost_usd:.6f} ({origin})",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )
