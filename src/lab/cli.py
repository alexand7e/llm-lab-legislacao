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

from datetime import datetime
from pathlib import Path

import httpx2
import typer

from lab.eval.dataset import DatasetError, check_against_corpus, distribution_gaps, load_questions
from lab.eval.registry import (
    RunConfig,
    RunMeta,
    RunMetrics,
    git_state,
    sha256_file,
    write_run,
)
from lab.eval.runner import QuestionResult, run_eval
from lab.ingest.export import CorpusError, read_jsonl
from lab.ingest.pipeline import run_ingest
from lab.ingest.sources import load_corpus
from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.cache import SQLiteCache
from lab.settings import Settings, load_env, load_models_config
from lab.strategies import BaselineStrategy

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


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta sobre a legislação do corpus."),
    prompt: str = typer.Option(
        "baseline_v1", "--prompt", help="Prompt em prompts/ (a versão faz parte do nome)."
    ),
    role: str = typer.Option("generator", "--role", "-r", help="Papel de modelo em models.yaml."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignora o cache em disco."),
    config: Path | None = typer.Option(
        None, "--config", help="Caminho alternativo de models.yaml."
    ),
) -> None:
    """Fazer uma pergunta à estratégia baseline (o modelo responde de memória).

    Sem recuperação: os artigos citados vêm do que o modelo lembra e podem
    estar errados. É de propósito: esta é a linha de base que RAG e GraphRAG
    precisam superar. A resposta vai para o stdout; artigos citados, confiança,
    tokens e custo vão para o stderr.

    Exemplo:

        lab ask "Qual o prazo de resposta do controlador na LGPD?"
    """
    settings = Settings()
    cache = None if no_cache else SQLiteCache(settings.lab_cache_dir)
    try:
        models = load_models_config(config or settings.lab_models_config)
        corpus = load_corpus(settings.lab_corpus_config)
        client = LLMClient(
            models, cache=cache, max_usd=settings.lab_max_usd_per_run, env=load_env()
        )
        strategy = BaselineStrategy(client, corpus.laws, prompt=prompt, role=role)
        answer = strategy.answer(question)
    except FileNotFoundError as exc:
        typer.secho(f"error: arquivo não encontrado: {exc.filename or exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    except (LLMConfigError, BudgetExceededError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if cache is not None:
            cache.close()

    typer.echo(answer.text)
    cited = ", ".join(answer.cited_articles) or "nenhum"
    confidence = "" if answer.confidence is None else f" confiança={answer.confidence:.2f}"
    origin = f"{answer.latency_ms} ms" if answer.latency_ms else "cache"
    u = answer.usage
    typer.secho(
        f"[{strategy.name}] artigos citados: {cited}{confidence}\n"
        f"in={u.prompt_tokens} out={u.completion_tokens} cost=${u.cost_usd:.6f} ({origin})",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )
    if answer.parse_error:
        typer.secho(
            f"aviso: resposta fora do schema ({answer.parse_error})",
            fg=typer.colors.YELLOW,
            err=True,
        )


@app.command(name="eval")
def eval_command(
    questions_file: Path = typer.Option(
        Path("data/eval/questions.jsonl"),
        "--questions",
        help=(
            "Conjunto de perguntas (JSONL). Para testar o pipeline: data/eval/questions.dev.jsonl."
        ),
    ),
    articles: Path = typer.Option(
        Path("data/processed/articles.jsonl"),
        "--articles",
        help="Corpus, para conferir o gabarito das perguntas antes de gastar.",
    ),
    prompt: str = typer.Option("baseline_v1", "--prompt", help="Prompt em prompts/."),
    role: str = typer.Option("generator", "--role", "-r", help="Papel de modelo em models.yaml."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Só as N primeiras perguntas."),
    workers: int = typer.Option(4, "--workers", "-w", min=1, help="Perguntas em paralelo."),
    out: Path = typer.Option(Path("results"), "--out", help="Pasta onde gravar a run."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignora o cache em disco."),
    config: Path | None = typer.Option(
        None, "--config", help="Caminho alternativo de models.yaml."
    ),
) -> None:
    """Avaliar a estratégia baseline no conjunto de perguntas e registrar a run.

    Valida as perguntas contra o corpus, responde todas (em paralelo, com
    cache e limite de custo LAB_MAX_USD_PER_RUN), calcula as métricas
    determinísticas e grava results/<AAAAMMDD-HHMM>-baseline/ com config,
    metadados, respostas e métricas. Uma run só é oficial (meta.json) se
    veio de commit limpo, sem erros e sem perguntas de rascunho.

    Exemplo:

        lab eval --questions data/eval/questions.dev.jsonl
    """
    settings = Settings()
    cache = None if no_cache else SQLiteCache(settings.lab_cache_dir)
    try:
        questions = load_questions(questions_file)
        if articles.exists():
            check_against_corpus(questions, read_jsonl(articles))
        else:
            typer.secho(f"aviso: {articles} não existe; gabarito não conferido", err=True)
        if limit is not None:
            questions = questions[:limit]
        models = load_models_config(config or settings.lab_models_config)
        corpus = load_corpus(settings.lab_corpus_config)
        client = LLMClient(
            models, cache=cache, max_usd=settings.lab_max_usd_per_run, env=load_env()
        )
        strategy = BaselineStrategy(client, corpus.laws, prompt=prompt, role=role)

        done = 0

        def progress(result: QuestionResult) -> None:
            nonlocal done
            done += 1
            mark = "ERRO" if result.error else "ok"
            typer.secho(f"[{done}/{len(questions)}] {result.question_id} {mark}", err=True)

        started = datetime.now()
        results = run_eval(strategy, questions, workers=workers, on_result=progress)
        finished = datetime.now()
        run_config = RunConfig(
            strategy=strategy.name,
            prompt=prompt,
            roles={role: models.roles[role].model},
            questions_file=str(questions_file),
            questions_sha256=sha256_file(questions_file),
            questions_count=len(questions),
            workers=workers,
            max_usd=settings.lab_max_usd_per_run,
            use_cache=cache is not None,
        )
        run_dir = write_run(
            out,
            config=run_config,
            questions=questions,
            results=results,
            started=started,
            finished=finished,
            git=git_state(),
        )
    except FileNotFoundError as exc:
        typer.secho(f"error: arquivo não encontrado: {exc.filename or exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    except (DatasetError, CorpusError, LLMConfigError, BudgetExceededError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if cache is not None:
            cache.close()

    metrics = RunMetrics.model_validate_json((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = RunMeta.model_validate_json((run_dir / "meta.json").read_text(encoding="utf-8"))
    typer.echo(f"{'categoria':<14}{'n':>3}  {'cita gab.':>9}  {'abst. ok':>8}  {'recusa ind.':>11}")
    for name, agg in {"TOTAL": metrics.overall, **metrics.by_category}.items():
        cells = [
            "-" if agg.means.get(k) is None else f"{agg.means[k]:.2f}"
            for k in ("citation_hit", "abstention_correct", "wrongful_refusal")
        ]
        typer.echo(f"{name:<14}{agg.n:>3}  {cells[0]:>9}  {cells[1]:>8}  {cells[2]:>11}")
    typer.echo(f"custo estimado ${meta.total_cost_usd:.6f}; erros: {meta.errors}; run: {run_dir}")
    if meta.dev_questions:
        typer.secho(
            "aviso: perguntas de rascunho (dev_draft); run não oficial", fg="yellow", err=True
        )
    elif meta.git_dirty:
        typer.secho(
            "aviso: árvore com alterações não commitadas; run não oficial", fg="yellow", err=True
        )
    gaps = {k: v for k, v in distribution_gaps(questions).items() if v > 0}
    if gaps and not meta.dev_questions:
        typer.secho(f"aviso: conjunto fora da distribuição oficial (faltam: {gaps})", err=True)
