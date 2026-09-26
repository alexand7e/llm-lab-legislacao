# -----------------------------------------------------------------------------
# File:     src/lab/eval/registry.py
# Purpose:  Record each evaluation run under results/<YYYYMMDD-HHMM>-<strategy>/.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Registro de runs (SPEC 7.4).

Cada execução vira uma pasta ``results/<AAAAMMDD-HHMM>-<estrategia>/`` com:

- ``config.yaml``: estratégia, prompt, modelos de cada papel, conjunto de
  perguntas (caminho e sha256) e parâmetros do runner;
- ``meta.json``: SHA do commit, árvore suja ou não, custo, duração, erros;
- ``answers.jsonl``: uma linha por pergunta (resposta, métricas ou erro);
- ``metrics.json``: agregados gerais e por categoria.

**Um resultado só vale para o relatório se veio de um commit limpo**, com o
conjunto oficial e sem erros. ``meta.json`` traz ``official`` calculado com essa
regra, para o relatório filtrar sem reinterpretar.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from lab.eval.dataset import Question
from lab.eval.metrics import Aggregate, aggregate, aggregate_by_category
from lab.eval.runner import QuestionResult


class GitState(BaseModel):
    """Estado do repositório no momento da run."""

    commit: str | None
    dirty: bool


def git_state(root: Path = Path(".")) -> GitState:
    """SHA do commit atual e se há alterações não commitadas.

    Fora de um repositório (ou sem ``git``), ``commit`` é ``None`` e a árvore
    conta como suja: sem commit rastreável a run não é oficial.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit=None, dirty=True)
    return GitState(commit=sha, dirty=bool(status.strip()))


class RunConfig(BaseModel):
    """O que define uma run, para reproduzi-la."""

    strategy: str
    prompt: str
    roles: dict[str, str]  # papel -> modelo
    questions_file: str
    questions_sha256: str
    questions_count: int
    workers: int
    max_usd: float
    use_cache: bool


class RunMeta(BaseModel):
    """Metadados da execução (``meta.json``)."""

    commit: str | None
    git_dirty: bool
    started_at: datetime
    finished_at: datetime
    duration_s: float
    total_cost_usd: float
    questions: int
    errors: int
    dev_questions: bool  # há perguntas dev_draft: fora do conjunto congelado
    official: bool


class RunMetrics(BaseModel):
    """Agregados da run (``metrics.json``)."""

    overall: Aggregate
    by_category: dict[str, Aggregate]


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir_name(started: datetime, strategy: str) -> str:
    """``AAAAMMDD-HHMM-<estrategia>``."""
    return f"{started:%Y%m%d-%H%M}-{strategy}"


def compute_metrics(questions: list[Question], results: list[QuestionResult]) -> RunMetrics:
    """Agregar só as respostas bem-sucedidas (erros não entram nas médias)."""
    by_id = {q.id: q for q in questions}
    rows = [
        (by_id[r.question_id], r.answer, r.scores)
        for r in results
        if r.answer is not None and r.scores is not None
    ]
    return RunMetrics(overall=aggregate(rows), by_category=aggregate_by_category(rows))


def write_run(
    out_root: Path,
    *,
    config: RunConfig,
    questions: list[Question],
    results: list[QuestionResult],
    started: datetime,
    finished: datetime,
    git: GitState,
) -> Path:
    """Gravar a pasta da run e devolver o caminho.

    Se a pasta já existe (duas runs no mesmo minuto), acrescenta ``-2``, ``-3``...
    em vez de sobrescrever.
    """
    base = out_root / run_dir_name(started, config.strategy)
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}-{n}")
    run_dir.mkdir(parents=True)

    metrics = compute_metrics(questions, results)
    errors = sum(1 for r in results if r.error)
    dev = any(q.origin == "dev_draft" for q in questions)
    meta = RunMeta(
        commit=git.commit,
        git_dirty=git.dirty,
        started_at=started,
        finished_at=finished,
        duration_s=round((finished - started).total_seconds(), 3),
        total_cost_usd=metrics.overall.cost_usd,
        questions=len(questions),
        errors=errors,
        dev_questions=dev,
        official=not git.dirty and git.commit is not None and not dev and errors == 0,
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "answers.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for result in results:
            fh.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return run_dir


__all__ = [
    "GitState",
    "RunConfig",
    "RunMeta",
    "RunMetrics",
    "compute_metrics",
    "git_state",
    "run_dir_name",
    "sha256_file",
    "write_run",
]
