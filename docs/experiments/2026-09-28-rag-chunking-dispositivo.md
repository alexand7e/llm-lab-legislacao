# Experimento: chunking por dispositivo × por artigo (#36)

Data: 2026-09-28. Estratégia `rag` (`prompts/rag_v1.md`), `BAAI/bge-m3`, gerador `soberano-alpha`, juiz `Qwen/Qwen3.6-35B-A3B` (`judge_v1`). Conjunto: `data/eval/questions.dev.jsonl` (16 perguntas `dev_draft`, **não oficial**).

Granularidades:

- `article`: um chunk por artigo vigente (223 chunks).
- `unit`: um chunk para o caput (com incisos e alíneas) e um por parágrafo (com os seus); artigo sem parágrafo fica inteiro (509 chunks). Trecho de parágrafo começa com o número do artigo ("Art. 19, § 1º ...").

Reproduzir:

```bash
uv run lab index --granularity unit
uv run lab eval --strategy rag -g unit --k 5  --questions data/eval/questions.dev.jsonl
uv run lab eval --strategy rag -g unit --k 10 --questions data/eval/questions.dev.jsonl
uv run lab compare results/<article k=5> results/<unit k=5> results/<unit k=10> --markdown
```

## Resultado (TOTAL)

| grupo | métrica | `20260928-2055-rag` | `20260928-2103-rag` | `20260928-2104-rag` |
|---|---|---|---|---|
| TOTAL (n=16) | correção | 0.91 | 0.81 (-0.09) | 0.97 (+0.06) |
| TOTAL (n=16) | cita gabarito | 1.00 | 1.00 (+0.00) | 1.00 (+0.00) |
| TOTAL (n=16) | precisão citação | 0.91 | 0.91 (+0.00) | 0.92 (+0.01) |
| TOTAL (n=16) | recall@5 | 0.92 | 0.92 (+0.00) | 0.92 (+0.00) |
| TOTAL (n=16) | recall@10 | 0.92 | 0.92 (+0.00) | 1.00 (+0.08) |
| TOTAL (n=16) | MRR | 0.85 | 0.87 (+0.02) | 0.87 (+0.02) |
| TOTAL (n=16) | abstenção ok | 1.00 | 1.00 (+0.00) | 1.00 (+0.00) |
| TOTAL (n=16) | recusa indevida | 0.00 | 0.00 (+0.00) | 0.00 (+0.00) |
| TOTAL | custo (US$) | 0.0000 | 0.0000 | 0.0000 |
| TOTAL | latência p50 (s) | 7.7 | 7.4 | 7.7 |
| TOTAL | latência p95 (s) | 13.1 | 11.0 | 11.0 |
- `20260928-2055-rag`: rag/rag_v1/soberano-alpha/k=5/article
- `20260928-2103-rag`: rag/rag_v1/soberano-alpha/k=5/unit
- `20260928-2104-rag`: rag/rag_v1/soberano-alpha/k=10/unit

| configuração | tokens de entrada / pergunta |
|---|---|
| article, k = 5 | 2.158 |
| unit, k = 5 | 1.089 |
| unit, k = 10 | 1.788 |

## Leitura

- **unit com k = 10** teve a maior correção até aqui (0,97) e recall@10 de 1,00 (achou o segundo artigo das perguntas de duas partes, o que article não fazia com nenhum k), gastando **menos** tokens que article com k = 5.
- **unit com k = 5** piora (0,81): trechos menores, então 5 trechos cobrem menos artigos. O k precisa acompanhar a granularidade.
- O ganho vem de caber mais artigos distintos no mesmo orçamento de tokens: parágrafos que não interessam ficam de fora.

## Decisão

Ver ADR 0002 (#39). Reavaliar com as 80 perguntas oficiais (#20).
