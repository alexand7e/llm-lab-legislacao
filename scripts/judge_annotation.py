# -----------------------------------------------------------------------------
# File:     scripts/judge_annotation.py
# Purpose:  Export answers for blind human annotation and measure judge agreement.
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Validação do juiz contra 20 anotações humanas (issue #24, aceite do M2).

1. Exportar a amostra (a nota do juiz NÃO vai no arquivo: anotação cega):

    uv run python scripts/judge_annotation.py export results/<run> \\
        --questions data/eval/questions.jsonl

2. Abrir ``data/eval/judge_annotations.jsonl`` e preencher ``human_score`` (0, 1
   ou 2) em cada linha, seguindo a rubrica de ``prompts/judge_v1.md``. O campo
   ``note`` é livre, para justificar casos difíceis.

3. Medir a concordância com o juiz daquela run:

    uv run python scripts/judge_annotation.py agreement results/<run>

Aceite: concordância exata ≥ 80% com as 20 anotadas. Sai com código 1 se não
passar (ou se ainda houver linhas sem nota).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab.eval.annotation import (
    ACCEPTANCE,
    AnnotationError,
    agreement,
    load_judge_scores,
    read_items,
    sample_for_annotation,
    write_items,
)
from lab.eval.dataset import load_questions
from lab.eval.runner import QuestionResult

DEFAULT_FILE = Path("data/eval/judge_annotations.jsonl")


def export(args: argparse.Namespace) -> int:
    if args.file.exists() and not args.force:
        print(f"erro: {args.file} já existe (use --force para sobrescrever)", file=sys.stderr)
        return 1
    questions = load_questions(args.questions)
    with (args.run / "answers.jsonl").open(encoding="utf-8") as fh:
        results = [QuestionResult.model_validate_json(line) for line in fh]
    items = sample_for_annotation(questions, results, n=args.n, seed=args.seed)
    write_items(args.file, items)
    print(f"{len(items)} respostas em {args.file}; preencha human_score (0, 1 ou 2).")
    return 0


def measure(args: argparse.Namespace) -> int:
    report = agreement(read_items(args.file), load_judge_scores(args.run))
    if report.compared == 0:
        print("nenhuma linha anotada ainda", file=sys.stderr)
        return 1
    print(f"comparadas: {report.compared}; sem nota humana: {report.pending}")
    print(f"concordância exata: {report.exact:.0%} (aceite ≥ {ACCEPTANCE:.0%})")
    print(f"concordância ±1:    {report.within_one:.0%}")
    print("matriz (linhas = humano, colunas = juiz):")
    print("        juiz 0  juiz 1  juiz 2")
    for human, row in enumerate(report.confusion):
        print(f"hum {human}  " + "  ".join(f"{v:>6}" for v in row))
    if report.disagreements:
        print("divergências: " + ", ".join(report.disagreements))
    print("ACEITO" if report.accepted else "NÃO ACEITO")
    return 0 if report.accepted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validação do juiz contra anotações humanas.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="sortear respostas para anotação cega")
    p_export.add_argument("run", type=Path, help="pasta da run (results/<...>)")
    p_export.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    p_export.add_argument("--file", type=Path, default=DEFAULT_FILE)
    p_export.add_argument("--n", type=int, default=20)
    p_export.add_argument("--seed", type=int, default=1)
    p_export.add_argument("--force", action="store_true")
    p_export.set_defaults(func=export)

    p_agree = sub.add_parser("agreement", help="concordância humano × juiz")
    p_agree.add_argument("run", type=Path, help="a mesma run usada no export")
    p_agree.add_argument("--file", type=Path, default=DEFAULT_FILE)
    p_agree.set_defaults(func=measure)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (AnnotationError, FileNotFoundError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
