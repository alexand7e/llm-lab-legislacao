# -----------------------------------------------------------------------------
# File:     tests/test_eval_compare.py
# Purpose:  Tests for run comparison (`lab compare`) with synthetic runs on disk.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lab.cli import app
from lab.eval.compare import (
    CompareError,
    build_rows,
    compatibility_warnings,
    load_run,
    render_markdown,
    render_text,
)
from lab.eval.metrics import Aggregate
from lab.eval.registry import RunConfig, RunMeta, RunMetrics

runner = CliRunner()


def aggregate(n: int, **means: float) -> Aggregate:
    return Aggregate(
        n=n,
        means=means,
        cost_usd=0.1,
        prompt_tokens=10,
        completion_tokens=5,
        latency_p50_ms=1500,
        latency_p95_ms=4000,
    )


def make_run(
    root: Path,
    name: str,
    *,
    strategy: str = "baseline",
    correctness: float = 0.5,
    sha: str = "sha256:a",
    judge: str | None = "juiz",
    official: bool = True,
    cost: float = 0.2,
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    config = RunConfig(
        strategy=strategy,
        prompt=f"{strategy}_v1",
        roles={"generator": "modelo-x"},
        questions_file="q.jsonl",
        questions_sha256=sha,
        questions_count=4,
        workers=2,
        max_usd=2.0,
        use_cache=True,
        judge_model=judge,
        judge_prompt="judge_v1" if judge else None,
    )
    moment = datetime(2026, 9, 27, 12, 0)
    meta = RunMeta(
        commit="abc",
        git_dirty=not official,
        started_at=moment,
        finished_at=moment,
        duration_s=1.0,
        total_cost_usd=cost,
        questions=4,
        errors=0,
        dev_questions=False,
        official=official,
    )
    metrics = RunMetrics(
        overall=aggregate(4, correctness=correctness, citation_hit=0.5, abstention_correct=1.0),
        by_category={
            "factual": aggregate(3, correctness=correctness, citation_hit=0.5),
            "sem_resposta": aggregate(1, abstention_correct=1.0),
        },
    )
    (path / "config.yaml").write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
    (path / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    (path / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")
    return path


def test_load_run_and_label(tmp_path: Path):
    run = load_run(make_run(tmp_path, "r1"))
    assert run.name == "r1" and run.label == "baseline/baseline_v1/modelo-x"


def test_load_rejects_folders_that_are_not_runs(tmp_path: Path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(CompareError, match="config.yaml ausente"):
        load_run(tmp_path / "vazia")
    broken = make_run(tmp_path, "quebrada")
    (broken / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompareError, match="inválidos"):
        load_run(broken)


def test_rows_show_values_and_delta_against_the_first_run(tmp_path: Path):
    runs = [
        load_run(make_run(tmp_path, "base", correctness=0.5)),
        load_run(make_run(tmp_path, "nova", correctness=0.75, strategy="fewshot")),
    ]
    rows = build_rows(runs)
    total_correct = next(r for r in rows if r[0] == "TOTAL (n=4)" and r[1] == "correção")
    assert total_correct[2:] == ["0.50", "0.75 (+0.25)"]
    factual = next(r for r in rows if r[0] == "factual (n=3)" and r[1] == "correção")
    assert factual[2:] == ["0.50", "0.75 (+0.25)"]


def test_metrics_that_apply_to_no_run_are_hidden(tmp_path: Path):
    rows = build_rows([load_run(make_run(tmp_path, "r1"))])
    labels = {(r[0], r[1]) for r in rows}
    assert ("TOTAL (n=4)", "MRR") not in labels  # nenhuma run recupera
    assert ("sem_resposta (n=1)", "correção") not in labels
    assert ("sem_resposta (n=1)", "abstenção ok") in labels


def test_missing_category_in_one_run_shows_a_dash(tmp_path: Path):
    a = load_run(make_run(tmp_path, "a"))
    b_path = make_run(tmp_path, "b")
    metrics = RunMetrics.model_validate_json((b_path / "metrics.json").read_text("utf-8"))
    metrics.by_category.pop("factual")
    (b_path / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")
    row = next(
        r for r in build_rows([a, load_run(b_path)]) if r[:2] == ["factual (n=3)", "correção"]
    )
    assert row[2:] == ["0.50", "-"]


def test_cost_and_latency_rows(tmp_path: Path):
    rows = build_rows([load_run(make_run(tmp_path, "r1", cost=0.25))])
    assert ["TOTAL", "custo (US$)", "0.2500"] in rows
    assert ["TOTAL", "latência p50 (s)", "1.5"] in rows
    assert ["TOTAL", "latência p95 (s)", "4.0"] in rows


def test_warnings_when_the_ruler_differs(tmp_path: Path):
    same = [load_run(make_run(tmp_path, "a")), load_run(make_run(tmp_path, "b"))]
    assert compatibility_warnings(same) == []
    other = [
        load_run(make_run(tmp_path, "c")),
        load_run(make_run(tmp_path, "d", sha="sha256:b", judge=None, official=False)),
    ]
    warnings = compatibility_warnings(other)
    assert any("conjuntos de perguntas diferentes" in w for w in warnings)
    assert any("juiz diferente" in w for w in warnings)
    assert any("não oficiais: d" in w for w in warnings)


def test_text_and_markdown_rendering(tmp_path: Path):
    runs = [load_run(make_run(tmp_path, "a")), load_run(make_run(tmp_path, "b"))]
    rows = build_rows(runs)
    text = render_text(runs, rows)
    assert text.startswith("[1] a  (baseline/baseline_v1/modelo-x)")
    markdown = render_markdown(runs, rows)
    assert markdown.splitlines()[0] == "| grupo | métrica | `a` | `b` |"
    assert markdown.splitlines()[1] == "|---|---|---|---|"
    assert "- `b`: baseline/baseline_v1/modelo-x" in markdown


def test_compare_command(tmp_path: Path):
    a, b = make_run(tmp_path, "a"), make_run(tmp_path, "b", correctness=0.25, official=False)
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "0.25 (-0.25)" in result.stdout
    assert "runs não oficiais: b" in result.stderr
    markdown = runner.invoke(app, ["compare", str(a), str(b), "--markdown"])
    assert markdown.stdout.startswith("| grupo | métrica |")


def test_compare_command_with_a_bad_folder(tmp_path: Path):
    result = runner.invoke(app, ["compare", str(tmp_path)])
    assert result.exit_code == 1 and "não é uma run" in result.stderr
