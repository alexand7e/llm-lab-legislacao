# Plano de execução

Última avaliação: 2026-09-27. Requisitos: [SPEC.md](../SPEC.md) seções 8.5 (definição de pronto) e 10 (milestones).

## Situação por milestone

| Milestone | Código | Pendências |
|---|---|---|
| M0 — Fundação | Concluído | Tag `v0.1.0` |
| M1 — Corpus | Concluído (#11–#18, #75, #83) | Conferência manual de 20 artigos (`scripts/sample_articles.py`); tag `v0.2.0` |
| M2 — Avaliação | Concluído no que é código (#19, #21–#23, #25, #26; ferramentas da #24) | **#20** as 80 perguntas; **#24** anotar 20 respostas; #69 Langfuse |
| M3 — Baseline de prompting | #27–#31 e #92 concluídos | **#32** comparar modelos (decisão sobre o juiz, abaixo) |

Pendências em **negrito** são humanas ou dependem de decisão do mantenedor.

## M2 — o que falta para fechar

1. **#20 — 80 perguntas oficiais** (manual, SPEC 7.2): 25 factual, 15 multi_artigo, 15 multi_lei, 10 numerica, 15 sem_resposta. Validar com `lab eval --questions data/eval/questions.jsonl` (o validador aponta gabarito inexistente ou revogado e a distribuição que falta). Congelar com a tag `eval-v1`.
2. **#24 — validar o juiz**: rodar `lab eval` no conjunto oficial, `scripts/judge_annotation.py export results/<run>`, preencher `human_score` (anotação cega) e `agreement`. Aceite: concordância exata ≥ 80%; abaixo disso, nova rubrica `judge_v2` antes de comparar estratégias.
3. #69 (Langfuse) é opcional para o aceite; fica para depois do M3.

## M3 — resultados preliminares (16 perguntas de desenvolvimento, não oficiais)

| estratégia | correção | cita gabarito | tokens de entrada / pergunta |
|---|---|---|---|
| baseline | 0.44 | 0.46 | 360 |
| fewshot | 0.56 | 0.46 | 758 |
| long_context | 1.00 | 0.77 | 41.735 |

Leitura: de memória o modelo erra números e artigos; com a lei no prompt acerta tudo, a um custo de ~116× em tokens de entrada. É a referência do M4: o RAG precisa se aproximar dessa correção com uma fração dos tokens.

### #32 — comparar modelos (bloqueado por decisão)

Modelos disponíveis no provedor: `soberano-alpha` (gerador atual), `Qwen/Qwen3.6-35B-A3B` (juiz atual), `Qwen3.8-27B`. A SPEC 4.2 exige juiz de família diferente do gerador. Um gerador Qwen avaliado por juiz Qwen fere essa regra; trocar o juiz só para essas runs torna a correção não comparável (`lab compare` avisa). E a linhagem do `soberano-alpha` ainda não foi confirmada. Precisa de uma decisão sobre o juiz antes de rodar.

## Interface local

Em `wip/interface-web` (rebaseada sobre a main; só adições: `src/lab/web/`, `lab serve`, fastapi/uvicorn). Vai à main em fatias quando o mantenedor validar a interface.

## Definição de pronto de cada issue

CI verde, testes novos, cobertura de `src/` ≥ 80%, nenhuma chamada real de API em teste unitário, README/CHANGELOG atualizados quando o comportamento visível mudar.
