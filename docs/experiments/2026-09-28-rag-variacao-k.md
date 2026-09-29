# Experimento: variação de k no RAG vetorial (#38)

Data: 2026-09-28. Estratégia `rag` (`prompts/rag_v1.md`), chunking por artigo, `BAAI/bge-m3`, gerador `soberano-alpha`, juiz `Qwen/Qwen3.6-35B-A3B` (`judge_v1`). Conjunto: `data/eval/questions.dev.jsonl` (16 perguntas `dev_draft`, **não oficial**).

Reproduzir:

```bash
uv run lab index
for k in 3 5 10; do uv run lab eval --strategy rag --k $k --questions data/eval/questions.dev.jsonl; done
uv run lab compare results/<k=3> results/<k=5> results/<k=10> --markdown
```

## Resultado (TOTAL)

| grupo | métrica | `20260928-2058-rag` | `20260928-2055-rag` | `20260928-2058-rag-2` |
|---|---|---|---|---|
| TOTAL (n=16) | correção | 0.84 | 0.91 (+0.06) | 0.88 (+0.03) |
| TOTAL (n=16) | cita gabarito | 0.92 | 1.00 (+0.08) | 0.92 (+0.00) |
| TOTAL (n=16) | precisão citação | 0.92 | 0.91 (-0.01) | 0.96 (+0.04) |
| TOTAL (n=16) | recall@5 | 0.85 | 0.92 (+0.08) | 0.92 (+0.08) |
| TOTAL (n=16) | recall@10 | 0.85 | 0.92 (+0.08) | 0.96 (+0.12) |
| TOTAL (n=16) | MRR | 0.83 | 0.85 (+0.02) | 0.85 (+0.02) |
| TOTAL (n=16) | abstenção ok | 1.00 | 1.00 (+0.00) | 1.00 (+0.00) |
| TOTAL (n=16) | recusa indevida | 0.08 | 0.00 (-0.08) | 0.00 (-0.08) |
| TOTAL | custo (US$) | 0.0000 | 0.0000 | 0.0000 |
| TOTAL | latência p50 (s) | 7.2 | 7.7 | 8.7 |
| TOTAL | latência p95 (s) | 11.0 | 13.1 | 24.7 |
- `20260928-2058-rag`: rag/rag_v1/soberano-alpha/k=3
- `20260928-2055-rag`: rag/rag_v1/soberano-alpha/k=5
- `20260928-2058-rag-2`: rag/rag_v1/soberano-alpha/k=10

| k | tokens de entrada / pergunta |
|---|---|
| 3 | 1.447 |
| 5 | 2.158 |
| 10 | 3.562 |

## Leitura

- **k = 5 é o melhor ponto** nessas perguntas: maior correção (0,91) e cita o gabarito em todas.
- **k = 3** perde trechos: recall@5 cai para 0,85, a correção para 0,84 e reaparece uma recusa indevida (o modelo diz que os trechos não respondem, porque o artigo certo não veio).
- **k = 10** recupera mais (recall@10 = 0,96), mas a correção não sobe (0,88): os trechos extras são ruído, a precisão de citação melhora pouco e a latência p95 quase dobra (13 s → 25 s), com 65% mais tokens que k = 5.
- O limite não é o k: nas perguntas de duas partes, o segundo artigo não aparece nem com k = 10 (multi_artigo recall 0,50 em todos os k). Aumentar k não resolve; o caminho é mudar a consulta (reescrita/decomposição, #42) ou a busca (híbrida BM25 + densa, #40), no M5.

## Decisão

Padrão `--k 5` mantido. Reavaliar com o conjunto oficial (80 perguntas, #20): com 16 perguntas, uma resposta muda a média em ~0,06.
