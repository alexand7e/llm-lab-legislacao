# llm-lab-legislacao

> Laboratório incremental de técnicas com LLMs (prompting, RAG, GraphRAG e
> fine-tuning) aplicado à legislação brasileira de direito digital e do
> consumidor, usando provedores em nuvem via SDK compatível com a API da
> OpenAI.

Projeto experimental e educacional. As respostas geradas não constituem
aconselhamento jurídico.

O produto não é o chatbot em si, e sim a comparação documentada entre as
abordagens — qualidade, custo e latência — medida sempre com o mesmo conjunto
de avaliação, milestone a milestone (M0 a M8 em SPEC.md).

## Visão geral

Cada milestone adiciona uma técnica e mede o ganho dela contra as anteriores:

| Milestone | Técnica | Versão |
|---|---|---|
| M0 | Fundação (cliente LLM, cache, custo) | v0.1.0 |
| M1 | Corpus (LGPD, Marco Civil, CDC) | v0.2.0 |
| M2 | Avaliação (80 perguntas, juiz LLM) | v0.3.0 |
| M3 | Baseline de prompting | v0.4.0 |
| M4 | RAG vetorial (Qdrant) | v0.5.0 |
| M5 | RAG avançado (híbrido, rerank, reescrita) | v0.6.0 |
| M6 | GraphRAG | v0.7.0 |
| M7 | Fine-tuning (LoRA) | v0.8.0 |
| M8 | Relatório comparativo final | v1.0.0 |

O código fica em src/lab/ e é organizado em módulos espelhando a arquitetura
da spec: llm/ (cliente, cache, custos), ingest/, index/, graph/,
strategies/, eval/, finetune/.

## Instalação

Requisitos: Python 3.12 e uv.

    git clone https://github.com/alexand7e/llm-lab-legislacao.git
    cd llm-lab-legislacao
    uv sync                # instala dependências + grupo dev (ruff, pyright, pytest)

## Configuração

Copie o template de variáveis e preencha o que for usar:

    cp .env.example .env

| Variável | Uso |
|---|---|
| PRIMARY_BASE_URL, PRIMARY_API_KEY | provedor principal (papéis generator/judge/synthesizer) |
| OPENROUTER_API_KEY | provedor OpenRouter |
| DEEPINFRA_API_KEY | provedor DeepInfra (papéis embedding) |
| TOGETHER_API_KEY | provedor Together AI |
| QDRANT_URL, QDRANT_API_KEY | índice vetorial (M4+) |
| LAB_MAX_USD_PER_RUN | limite de gasto por run, em US$ (padrão 2.00) |
| LAB_CACHE_DIR | diretório do cache de respostas (padrão .cache/llm) |

Os papéis de modelo e os preços ficam em config/models.yaml — preencha model
e price_in / price_out dos papéis que for usar. URL e chave de API nunca
vivem nesse arquivo: ficam no .env (que é gitignored).

## Uso

### Testar um papel de modelo

    uv run lab chat "olá"
    uv run lab chat --role judge "esta resposta está correta?"
    uv run lab chat --no-cache "olá"   # ignora o cache em disco

A resposta vai para o stdout; tokens, custo e latência vão para o stderr:

    $ uv run lab chat "olá"
    Olá! Como posso ajudar?
    [generator] in=12 out=8 cost=$0.000035 (412 ms)

Repetições idênticas (mesmo modelo, mesma mensagem, mesmos parâmetros) saem do
cache em disco e são gratuitas — o indicador (cache) aparece no stderr.

## Estrutura do repositório

    llm-lab-legislacao/
    ├── SPEC.md                  # spec completa do projeto
    ├── README.md
    ├── LICENSE                  # Apache-2.0
    ├── CHANGELOG.md
    ├── .env.example
    ├── pyproject.toml
    ├── config/
    │   ├── models.yaml          # papéis de modelo e provedores
    │   └── experiments/         # um YAML por experimento (M2+)
    ├── prompts/                 # prompts versionados (M3+)
    ├── data/
    │   ├── raw/                 # HTML coletado (gitignored)
    │   ├── processed/           # articles.jsonl (versionado, M1)
    │   ├── eval/                # conjunto de avaliação (versionado, M2)
    │   └── train/               # dados de fine-tuning (M7)
    ├── src/lab/
    │   ├── cli.py
    │   ├── settings.py
    │   ├── llm/                 # cliente, cache, custos (M0)
    │   ├── ingest/              # coleta e parser (M1)
    │   ├── index/               # embeddings e Qdrant (M4)
    │   ├── graph/               # grafo (M6)
    │   ├── strategies/          # baseline, rag, graphrag, finetuned (M3+)
    │   ├── eval/                # runner, métricas, juiz (M2)
    │   └── finetune/            # dataset e jobs (M7)
    ├── results/                 # uma pasta por run de avaliação (M2+)
    ├── reports/                 # relatório comparativo (M8)
    ├── docs/
    │   ├── adr/                 # decisões de arquitetura
    │   ├── commits.md           # padrão de commits
    │   └── TODO.md              # pendências de documentação
    ├── tests/
    └── scripts/                 # bootstrap_github.sh

## Desenvolvimento

    uv run pre-commit install        # uma vez por clone: hooks de ruff e pyright no commit
    uv run ruff check src tests      # lint
    uv run pyright                   # type-check (strict em src/)
    uv run pytest                    # testes (unitários; -m api para os que chamam API real)
    uv run pytest --cov              # testes + gate de cobertura (≥ 80%, como no CI)
    uv run pre-commit run --all-files

Padrão de commits em docs/commits.md (Conventional Commits, em português,
com issue). Fluxo de desenvolvimento: issue → branch → PR, em SPEC.md seção 8.

## Status

M0 (fundação) em andamento: cliente LLM, cache SQLite, lab chat, CI,
pre-commit, templates e ADR 0001 prontos; proteção da main pendente
(issue #5).
