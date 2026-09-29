# -----------------------------------------------------------------------------
# File:     src/lab/eval/compare.py
# Purpose:  Side-by-side comparison of evaluation runs (results/<run>/).
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Comparação de runs (``lab compare``).

Lê ``config.yaml``, ``meta.json`` e ``metrics.json`` de cada run e monta uma
tabela por categoria, com a diferença de cada run contra a primeira (a
referência). A tabela sai em texto para o terminal ou em Markdown para o corpo
dos PRs de experimento (SPEC 8.4).

Comparar só faz sentido com a mesma régua. :func:`compatibility_warnings` avisa
quando as runs usam conjuntos de perguntas diferentes, juízes (modelo ou
rubrica) diferentes, ou quando alguma não é oficial. Os avisos não impedem a
comparação: servem para ninguém tirar conclusão de uma tabela enganosa.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from lab.eval.metrics import Aggregate
from lab.eval.registry import RunConfig, RunMeta, RunMetrics

# (chave em Aggregate.means, rótulo, maior é melhor?)
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("correctness", "correção", True),
    ("citation_hit", "cita gabarito", True),
    ("citation_precision", "precisão citação", True),
    ("recall_at_5", "recall@5", True),
    ("recall_at_10", "recall@10", True),
    ("mrr", "MRR", True),
    ("abstention_correct", "abstenção ok", True),
    ("wrongful_refusal", "recusa indevida", False),
)


class CompareError(ValueError):
    """Run ilegível ou incompleta."""


@dataclass(frozen=True)
class Run:
    """Uma run lida do disco."""

    name: str
    config: RunConfig
    meta: RunMeta
    metrics: RunMetrics

    @property
    def label(self) -> str:
        """Rótulo curto: estratégia, prompt e modelo."""
        models = ",".join(self.config.roles.values())
        label = f"{self.config.strategy}/{self.config.prompt}/{models}"
        return f"{label}/k={self.config.k}" if self.config.k else label


def load_run(path: Path) -> Run:
    """Ler uma run de ``results/<run>/``.

    :raises CompareError: pasta sem os arquivos da run ou com conteúdo inválido.
    """
    try:
        config = RunConfig.model_validate(
            yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        )
        meta = RunMeta.model_validate_json((path / "meta.json").read_text(encoding="utf-8"))
        metrics = RunMetrics.model_validate_json(
            (path / "metrics.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise CompareError(f"{path}: não é uma run ({Path(exc.filename).name} ausente)") from exc
    except ValueError as exc:
        raise CompareError(f"{path}: arquivos da run inválidos ({exc})") from exc
    return Run(name=path.name, config=config, meta=meta, metrics=metrics)


def compatibility_warnings(runs: Sequence[Run]) -> list[str]:
    """Motivos pelos quais a comparação pode enganar (lista vazia = régua igual)."""
    warnings: list[str] = []
    if len({r.config.questions_sha256 for r in runs}) > 1:
        warnings.append("as runs usam conjuntos de perguntas diferentes (questions_sha256)")
    judges = {(r.config.judge_model, r.config.judge_prompt) for r in runs}
    if len(judges) > 1:
        warnings.append("juiz diferente entre as runs (modelo ou rubrica): correção não comparável")
    unofficial = [r.name for r in runs if not r.meta.official]
    if unofficial:
        warnings.append("runs não oficiais: " + ", ".join(unofficial))
    return warnings


def _groups(runs: Sequence[Run]) -> list[str]:
    categories = sorted({c for r in runs for c in r.metrics.by_category})
    return ["TOTAL", *categories]


def _aggregate(run: Run, group: str) -> Aggregate | None:
    return run.metrics.overall if group == "TOTAL" else run.metrics.by_category.get(group)


def _cell(value: float | None, base: float | None, *, first: bool) -> str:
    if value is None:
        return "-"
    if first or base is None:
        return f"{value:.2f}"
    return f"{value:.2f} ({value - base:+.2f})"


def build_rows(runs: Sequence[Run]) -> list[list[str]]:
    """Linhas da tabela: grupo, métrica e um valor por run (com delta contra a 1ª)."""
    rows: list[list[str]] = []
    for group in _groups(runs):
        aggregates = [_aggregate(r, group) for r in runs]
        n = next((a.n for a in aggregates if a is not None), 0)
        for key, label, _ in METRICS:
            values = [a.means.get(key) if a is not None else None for a in aggregates]
            if all(v is None for v in values):
                continue  # métrica que não se aplica a nenhuma run neste grupo
            base = values[0]
            cells = [_cell(v, base, first=i == 0) for i, v in enumerate(values)]
            rows.append([f"{group} (n={n})", label, *cells])
    totals = [r.metrics.overall for r in runs]
    rows.append(["TOTAL", "custo (US$)", *(f"{r.meta.total_cost_usd:.4f}" for r in runs)])
    rows.append(["TOTAL", "latência p50 (s)", *(_seconds(a.latency_p50_ms) for a in totals)])
    rows.append(["TOTAL", "latência p95 (s)", *(_seconds(a.latency_p95_ms) for a in totals)])
    return rows


def _seconds(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def render_text(runs: Sequence[Run], rows: list[list[str]]) -> str:
    """Tabela alinhada para o terminal."""
    header = ["grupo", "métrica", *(f"[{i + 1}]" for i in range(len(runs)))]
    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in table]
    legend = [f"[{i + 1}] {r.name}  ({r.label})" for i, r in enumerate(runs)]
    return "\n".join([*legend, "", *lines])


def render_markdown(runs: Sequence[Run], rows: list[list[str]]) -> str:
    """Tabela Markdown (para o corpo do PR)."""
    header = ["grupo", "métrica", *(f"`{r.name}`" for r in runs)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    legend = [f"- `{r.name}`: {r.label}" for r in runs]
    return "\n".join([*lines, "", *legend])


__all__ = [
    "METRICS",
    "CompareError",
    "Run",
    "build_rows",
    "compatibility_warnings",
    "load_run",
    "render_markdown",
    "render_text",
]
