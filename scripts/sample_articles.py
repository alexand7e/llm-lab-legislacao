# -----------------------------------------------------------------------------
# File:     scripts/sample_articles.py
# Purpose:  Draw a reproducible sample of articles for the manual M1 check.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Sorteia artigos de ``articles.jsonl`` para a conferência manual do M1.

O aceite do M1 pede 20 artigos conferidos à mão contra o texto oficial, sem
divergência. Este script imprime, em Markdown, a amostra (semente fixa, então
o mesmo sorteio sai sempre): para cada artigo, o texto extraído e a URL da
norma. Abra a URL, procure o artigo e confira número, texto, status e caminho.

Uso:

    uv run python scripts/sample_articles.py [--n 20] [--seed 1] [--file PATH]

A amostra é proporcional ao tamanho de cada norma, com ao menos um artigo por
norma e priorizando casos difíceis: artigos acrescidos, revogados e vetados
entram sempre que existirem, antes do sorteio ao acaso.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from lab.ingest.export import ArticleRecord, read_jsonl
from lab.ingest.sources import load_corpus

MAX_CHARS = 500


def sample(records: list[ArticleRecord], n: int, seed: int) -> list[ArticleRecord]:
    """Amostra de ``n`` artigos: casos difíceis primeiro, depois ao acaso."""
    rng = random.Random(seed)
    hard = [r for r in records if "-" in r.article or r.status != "vigente"]
    rng.shuffle(hard)
    chosen: list[ArticleRecord] = []
    for law in sorted({r.law for r in records}):  # ao menos um por norma
        pool = [r for r in hard if r.law == law] or [r for r in records if r.law == law]
        chosen.append(rng.choice(pool))
    rest = [r for r in records if r not in chosen]
    rng.shuffle(rest)
    hard_rest = [r for r in hard if r not in chosen][: max(0, (n - len(chosen)) // 3)]
    chosen += hard_rest
    chosen += [r for r in rest if r not in chosen][: max(0, n - len(chosen))]
    return sorted(chosen[:n], key=lambda r: (r.law, r.id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--file", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("config/corpus.yaml"))
    args = parser.parse_args()

    urls = {law.id: law.url for law in load_corpus(args.corpus).laws}
    picked = sample(read_jsonl(args.file), args.n, args.seed)
    print(f"# Amostra de {len(picked)} artigos (seed={args.seed})\n")
    for i, r in enumerate(picked, start=1):
        text = r.text if len(r.text) <= MAX_CHARS else r.text[:MAX_CHARS] + " [...]"
        print(f"## {i}. {r.id} — {r.status}")
        print(f"- Norma: {urls.get(r.law, r.law)}")
        print(f"- Caminho: {' > '.join(r.path) or '(sem título)'}")
        print(f"- Dispositivos: {len(r.units)}; remissões: {len(r.references)}")
        print("- [ ] confere com o texto oficial\n\n> " + text.replace("\n", "\n> ") + "\n")


if __name__ == "__main__":
    main()
