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

import httpx2
import typer

from lab.ingest.export import CorpusError
from lab.ingest.pipeline import run_ingest
from lab.ingest.sources import load_corpus
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


@app.command()
def ingest(
    fetch: bool = typer.Option(
        False,
        "--fetch",
        help=(
            "Baixa de novo todas as normas do Planalto. Sem esta opção, só baixa as "
            "que ainda não estão em --raw-dir."
        ),
    ),
    raw_dir: Path = typer.Option(
        Path("data/raw"), "--raw-dir", help="Onde ficam o HTML bruto e o meta.json (fora do git)."
    ),
    output: Path = typer.Option(
        Path("data/processed/articles.jsonl"), "--out", help="Arquivo articles.jsonl a gravar."
    ),
    corpus_config: Path | None = typer.Option(
        None,
        "--corpus",
        help=(
            "Caminho alternativo para corpus.yaml. Padrão: LAB_CORPUS_CONFIG ou config/corpus.yaml."
        ),
    ),
) -> None:
    """Coletar, parsear e exportar o corpus para articles.jsonl.

    Lê as normas de config/corpus.yaml, confere o hash do HTML coletado e
    grava um único articles.jsonl com todas elas. Se qualquer norma falhar,
    nada é gravado. O resumo por norma vai para o stdout; alertas
    (remissões para artigos que não existem no corpus) vão para o stderr.

    Exemplos:

        lab ingest             # usa data/raw; baixa só o que falta

        lab ingest --fetch     # baixa tudo de novo
    """
    settings = Settings()
    try:
        corpus = load_corpus(corpus_config or settings.lab_corpus_config)
        report = run_ingest(corpus, raw_dir, output, fetch=fetch)
    except FileNotFoundError as exc:
        typer.secho(
            f"error: arquivo não encontrado: {exc.filename or exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from exc
    except (CorpusError, ValueError, httpx2.HTTPError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for s in report.laws:
        status = ", ".join(f"{n} {name}" for name, n in sorted(s.by_status.items()))
        typer.echo(f"{s.law}: {s.articles} artigos ({status}), {s.units} dispositivos")
    typer.echo(f"{report.total} artigos gravados em {report.output}")
    if report.dangling:
        typer.secho(
            f"aviso: {len(report.dangling)} remissões apontam para artigos ausentes do corpus "
            f"(ex.: {report.dangling[0][0]} -> {report.dangling[0][1]})",
            fg=typer.colors.YELLOW,
            err=True,
        )
