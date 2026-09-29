# Plano de execução

Última avaliação: 2026-09-28. Requisitos: [SPEC.md](../SPEC.md) seções 8.5 (definição de pronto) e 10 (milestones).

## Situação por milestone

| Milestone | Código | Pendências |
|---|---|---|
| M0 — Fundação | Concluído | Tag `v0.1.0` |
| M1 — Corpus | Concluído (#11–#18, #75, #83) | Conferência manual de 20 artigos (`scripts/sample_articles.py`); tag `v0.2.0` |
| M2 — Avaliação | Concluído no que é código (#19, #21–#23, #25, #26; ferramentas da #24) | **#20** as 80 perguntas; **#24** anotar 20 respostas; #69 Langfuse |
| M3 — Baseline de prompting | #27–#31 e #92 concluídos | #32 comparação preliminar feita (juiz Qwen, por decisão); fecha com o conjunto oficial |
| M4 — RAG vetorial | #33–#38 concluídos | **#39** aceitar o ADR 0002 (proposto); Qdrant Cloud opcional (roda local sem `QDRANT_URL`) |

Pendências em **negrito** são humanas ou dependem de decisão do mantenedor.

## M2 — o que falta para fechar

1. **#20 — 80 perguntas oficiais** (manual, SPEC 7.2): 25 factual, 15 multi_artigo, 15 multi_lei, 10 numerica, 15 sem_resposta. Validar com `lab eval --questions data/eval/questions.jsonl` (o validador aponta gabarito inexistente ou revogado e a distribuição que falta). Congelar com a tag `eval-v1`.
2. **#24 — validar o juiz**: rodar `lab eval` no conjunto oficial, `scripts/judge_annotation.py export results/<run>`, preencher `human_score` (anotação cega) e `agreement`. Aceite: concordância exata ≥ 80%; abaixo disso, nova rubrica `judge_v2` antes de comparar estratégias.
3. #69 (Langfuse) é opcional para o aceite; fica para depois do M3.

## Resultados preliminares (16 perguntas de desenvolvimento, não oficiais)

| estratégia | correção | cita gabarito | recall@10 | tokens de entrada / pergunta |
|---|---|---|---|---|
| baseline (soberano-alpha) | 0,44 | 0,46 | — | 360 |
| baseline (Qwen3.8-27B) | 0,59 | 0,69 | — | 328 |
| fewshot | 0,56 | 0,46 | — | 758 |
| rag, article, k = 5 | 0,91 | 1,00 | 0,92 | 2.158 |
| **rag, unit, k = 10** | **0,97** | **1,00** | **1,00** | **1.788** |
| long_context (lei inteira) | 1,00 | 1,00 | — | 41.735 |

Leitura: de memória o modelo erra números e artigos; com a lei inteira acerta tudo a ~23× o custo em tokens do melhor RAG. O RAG por dispositivo chega perto do teto com uma fração dos tokens. Experimentos em `docs/experiments/`.

## Próximos passos (M5 — RAG avançado)

Partem de `rag -g unit --k 10`: busca híbrida BM25 + densa (#40; o provedor tem `Qdrant/bm25`), reranking (#41; o provedor tem `colbert-ir/colbertv2.0`), reescrita/decomposição de consulta (#42, alvo das perguntas de duas partes), expansão por remissões (#43), filtro por norma (#44).

## Interface local

Em `wip/interface-web` (rebaseada sobre a main; só adições: `src/lab/web/`, `lab serve`, fastapi/uvicorn). Vai à main em fatias quando o mantenedor validar a interface.

## Definição de pronto de cada issue

CI verde, testes novos, cobertura de `src/` ≥ 80%, nenhuma chamada real de API em teste unitário, README/CHANGELOG atualizados quando o comportamento visível mudar.
