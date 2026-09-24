#!/usr/bin/env bash
# Cria labels, milestones e issues definidos no SPEC.md (seção 10).
# Uso: rodar UMA vez na raiz do repositório, já autenticado com `gh auth login`.
set -euo pipefail

# ---------- Labels (tipos da seção 8.3) ----------
label() { gh label create "$1" --color "$2" --description "$3" --force >/dev/null; }
label feat     "1D76DB" "Nova funcionalidade"
label fix      "D73A4A" "Correção"
label exp      "8E44AD" "Experimento com resultados"
label data     "0E8A16" "Dados versionados"
label docs     "0075CA" "Documentação"
label chore    "BFD4F2" "Build, CI, dependências"
label adr      "FBCA04" "Decisão de arquitetura"

# ---------- Helpers ----------
MS=""  # milestone atual

milestone() {
  MS="$1"
  gh api "repos/{owner}/{repo}/milestones" -f title="$1" -f description="$2" >/dev/null
  echo "Milestone criado: $1"
}

issue() {  # issue "<label>" "<título>"
  gh issue create --title "$2" --label "$1" --milestone "$MS" \
    --body "Ver SPEC.md, seção 10 (${MS%% —*}). Critérios de aceite do milestone valem para o fechamento." >/dev/null
  echo "  #  $2"
}

# ---------- M0 ----------
milestone "M0 — Fundação" "Infraestrutura para desenvolver e chamar modelos (v0.1.0)"
issue chore "Adicionar licença, README inicial, .gitignore e .env.example"
issue chore "Configurar projeto uv com ruff, pyright, pytest e pre-commit"
issue chore "Criar workflow de CI no GitHub Actions"
issue chore "Criar templates de issue e de PR"
issue chore "Configurar proteção da branch main"
issue feat  "Implementar settings e config/models.yaml"
issue feat  "Implementar cliente OpenAI-compatible com retentativas e custo"
issue feat  "Implementar cache de respostas em SQLite"
issue feat  "Criar comando lab chat"
issue adr   "ADR 0001: escolha de provedores"

# ---------- M1 ----------
milestone "M1 — Corpus" "Textos legais estruturados e confiáveis (v0.2.0)"
issue feat "Implementar coletor das leis com hash e data de coleta"
issue feat "Implementar parser da hierarquia até alínea"
issue feat "Detectar dispositivos revogados e vetados"
issue feat "Suportar artigos acrescidos (ex.: 55-A)"
issue feat "Extrair remissões internas e externas"
issue data "Exportar articles.jsonl com validação de schema"
issue feat "Criar testes do parser com fixtures de HTML"
issue feat "Criar comando lab ingest"

# ---------- M2 ----------
milestone "M2 — Avaliação" "Régua comum para todas as fases (v0.3.0)"
issue feat "Criar schema e validador do conjunto de avaliação"
issue data "Escrever as 80 perguntas de avaliação"
issue feat "Implementar runner de avaliação com cache e limite de custo"
issue feat "Implementar métricas determinísticas"
issue feat "Implementar juiz LLM com rubrica versionada"
issue exp  "Validar juiz contra 20 anotações manuais"
issue feat "Registrar runs em results/"
issue feat "Criar comando lab compare"

# ---------- M3 ----------
milestone "M3 — Baseline de prompting" "Desempenho sem recuperação (v0.4.0)"
issue feat "Criar interface Strategy e estratégia baseline"
issue exp  "System prompt v1 zero-shot"
issue exp  "Variante few-shot"
issue feat "Saída estruturada com artigos citados e confiança"
issue exp  "Estratégia long_context com lei inteira no prompt"
issue exp  "Comparar modelos de tamanhos diferentes"

# ---------- M4 ----------
milestone "M4 — RAG vetorial" "Recuperação densa básica (v0.5.0)"
issue feat "Gerar embeddings via API em lote"
issue feat "Criar coleção no Qdrant Cloud com metadados"
issue feat "Chunking por artigo"
issue exp  "Chunking por parágrafo ou inciso"
issue feat "Estratégia rag com citações obrigatórias"
issue exp  "Variação de k (3, 5, 10)"
issue adr  "ADR: estratégia de chunking"

# ---------- M5 ----------
milestone "M5 — RAG avançado" "Técnicas contra as falhas do M4 (v0.6.0)"
issue exp "Busca híbrida BM25 + densa com RRF"
issue exp "Reranking via API"
issue exp "Reescrita de consulta"
issue exp "Expansão por remissões"
issue feat "Filtro por lei via metadados"
issue docs "Análise de erros das perguntas que ainda falham"

# ---------- M6 ----------
milestone "M6 — GraphRAG" "Relações entre dispositivos e leis (v0.7.0)"
issue feat "Grafo estrutural determinístico"
issue feat "Extração de entidades e relações via LLM"
issue exp  "Avaliar LightRAG versus implementação própria"
issue feat "Estratégia graphrag com modos local e global"
issue feat "Visualização de subgrafos para depuração"
issue adr  "ADR: biblioteca de GraphRAG"

# ---------- M7 ----------
milestone "M7 — Fine-tuning" "Ajustar modelo versus recuperar contexto (v0.8.0)"
issue data "Gerar dataset sintético de treino"
issue feat "Checar vazamento contra o conjunto de avaliação"
issue data "Split treino/validação em chat JSONL"
issue exp  "Job de fine-tuning LoRA via API"
issue exp  "Estratégias finetuned e finetuned_rag"
issue docs "Análise de custo de treino versus inferência"
issue adr  "ADR: provedor de fine-tuning"

# ---------- M8 ----------
milestone "M8 — Relatório e portfólio" "Consolidar e comunicar resultados (v1.0.0)"
issue feat "Relatório comparativo com gráficos gerado por script"
issue docs "Seção de limitações e ameaças à validade"
issue docs "README final com diagrama e reprodução"
issue docs "Artigo de divulgação"

echo "Concluído."
