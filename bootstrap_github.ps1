# Cria labels, milestones e issues definidos no SPEC.md (seção 10).
# Equivalente em PowerShell a bootstrap_github.sh; rodar UMA vez autenticado com gh.
$ErrorActionPreference = "Stop"

function Add-Label([string]$name, [string]$color, [string]$desc) {
  gh label create $name --color $color --description $desc --force *> $null
}

Add-Label "feat"  "1D76DB" "Nova funcionalidade"
Add-Label "fix"   "D73A4A" "Correção"
Add-Label "exp"   "8E44AD" "Experimento com resultados"
Add-Label "data"  "0E8A16" "Dados versionados"
Add-Label "docs"  "0075CA" "Documentação"
Add-Label "chore" "BFD4F2" "Build, CI, dependências"
Add-Label "adr"   "FBCA04" "Decisão de arquitetura"

$MS = ""

function New-Milestone([string]$title, [string]$desc) {
  $script:MS = $title
  gh api "repos/{owner}/{repo}/milestones" -f "title=$title" -f "description=$desc" *> $null
  Write-Host "Milestone criado: $title"
}

function New-Issue([string]$label, [string]$title) {
  $msPart = ($script:MS -split " —")[0]
  $body = "Ver SPEC.md, seção 10 ($msPart). Critérios de aceite do milestone valem para o fechamento."
  gh issue create --title $title --label $label --milestone $script:MS --body $body *> $null
  Write-Host "  #  $title"
}

New-Milestone "M0 — Fundação" "Infraestrutura para desenvolver e chamar modelos (v0.1.0)"
New-Issue "chore" "Adicionar licença, README inicial, .gitignore e .env.example"
New-Issue "chore" "Configurar projeto uv com ruff, pyright, pytest e pre-commit"
New-Issue "chore" "Criar workflow de CI no GitHub Actions"
New-Issue "chore" "Criar templates de issue e de PR"
New-Issue "chore" "Configurar proteção da branch main"
New-Issue "feat"  "Implementar settings e config/models.yaml"
New-Issue "feat"  "Implementar cliente OpenAI-compatible com retentativas e custo"
New-Issue "feat"  "Implementar cache de respostas em SQLite"
New-Issue "feat"  "Criar comando lab chat"
New-Issue "adr"   "ADR 0001: escolha de provedores"

New-Milestone "M1 — Corpus" "Textos legais estruturados e confiáveis (v0.2.0)"
New-Issue "feat" "Implementar coletor das leis com hash e data de coleta"
New-Issue "feat" "Implementar parser da hierarquia até alínea"
New-Issue "feat" "Detectar dispositivos revogados e vetados"
New-Issue "feat" "Suportar artigos acrescidos (ex.: 55-A)"
New-Issue "feat" "Extrair remissões internas e externas"
New-Issue "data" "Exportar articles.jsonl com validação de schema"
New-Issue "feat" "Criar testes do parser com fixtures de HTML"
New-Issue "feat" "Criar comando lab ingest"

New-Milestone "M2 — Avaliação" "Régua comum para todas as fases (v0.3.0)"
New-Issue "feat" "Criar schema e validador do conjunto de avaliação"
New-Issue "data" "Escrever as 80 perguntas de avaliação"
New-Issue "feat" "Implementar runner de avaliação com cache e limite de custo"
New-Issue "feat" "Implementar métricas determinísticas"
New-Issue "feat" "Implementar juiz LLM com rubrica versionada"
New-Issue "exp"  "Validar juiz contra 20 anotações manuais"
New-Issue "feat" "Registrar runs em results/"
New-Issue "feat" "Criar comando lab compare"

New-Milestone "M3 — Baseline de prompting" "Desempenho sem recuperação (v0.4.0)"
New-Issue "feat" "Criar interface Strategy e estratégia baseline"
New-Issue "exp"  "System prompt v1 zero-shot"
New-Issue "exp"  "Variante few-shot"
New-Issue "feat" "Saída estruturada com artigos citados e confiança"
New-Issue "exp"  "Estratégia long_context com lei inteira no prompt"
New-Issue "exp"  "Comparar modelos de tamanhos diferentes"

New-Milestone "M4 — RAG vetorial" "Recuperação densa básica (v0.5.0)"
New-Issue "feat" "Gerar embeddings via API em lote"
New-Issue "feat" "Criar coleção no Qdrant Cloud com metadados"
New-Issue "feat" "Chunking por artigo"
New-Issue "exp"  "Chunking por parágrafo ou inciso"
New-Issue "feat" "Estratégia rag com citações obrigatórias"
New-Issue "exp"  "Variação de k (3, 5, 10)"
New-Issue "adr"  "ADR: estratégia de chunking"

New-Milestone "M5 — RAG avançado" "Técnicas contra as falhas do M4 (v0.6.0)"
New-Issue "exp" "Busca híbrida BM25 + densa com RRF"
New-Issue "exp" "Reranking via API"
New-Issue "exp" "Reescrita de consulta"
New-Issue "exp" "Expansão por remissões"
New-Issue "feat" "Filtro por lei via metadados"
New-Issue "docs" "Análise de erros das perguntas que ainda falham"

New-Milestone "M6 — GraphRAG" "Relações entre dispositivos e leis (v0.7.0)"
New-Issue "feat" "Grafo estrutural determinístico"
New-Issue "feat" "Extração de entidades e relações via LLM"
New-Issue "exp"  "Avaliar LightRAG versus implementação própria"
New-Issue "feat" "Estratégia graphrag com modos local e global"
New-Issue "feat" "Visualização de subgrafos para depuração"
New-Issue "adr"  "ADR: biblioteca de GraphRAG"

New-Milestone "M7 — Fine-tuning" "Ajustar modelo versus recuperar contexto (v0.8.0)"
New-Issue "data" "Gerar dataset sintético de treino"
New-Issue "feat" "Checar vazamento contra o conjunto de avaliação"
New-Issue "data" "Split treino/validação em chat JSONL"
New-Issue "exp"  "Job de fine-tuning LoRA via API"
New-Issue "exp"  "Estratégias finetuned e finetuned_rag"
New-Issue "docs" "Análise de custo de treino versus inferência"
New-Issue "adr"  "ADR: provedor de fine-tuning"

New-Milestone "M8 — Relatório e portfólio" "Consolidar e comunicar resultados (v1.0.0)"
New-Issue "feat" "Relatório comparativo com gráficos gerado por script"
New-Issue "docs" "Seção de limitações e ameaças à validade"
New-Issue "docs" "README final com diagrama e reprodução"
New-Issue "docs" "Artigo de divulgação"

# ---------- Proteção da branch main ----------
# SPEC 8.6: merge só via PR (0 aprovações), CI obrigatório, vale também para
# admins; force push e deleção bloqueados.
# O contexto "ci" precisa existir (workflow do GitHub Actions) para o merge ser permitido.
$protection = '{"required_status_checks":{"strict":true,"contexts":["ci"]},"enforce_admins":true,"required_pull_request_reviews":{"required_approving_review_count":0},"restrictions":null,"allow_force_pushes":false,"allow_deletions":false}'
$protection | gh api -X PUT repos/{owner}/{repo}/branches/main/protection --input - *> $null
# SPEC 8.1: só merge commit; branch apagada após o merge.
gh api -X PATCH repos/{owner}/{repo} -F allow_squash_merge=false -F allow_rebase_merge=false -F allow_merge_commit=true -F delete_branch_on_merge=true *> $null
Write-Host "Proteção da branch main configurada (merge só via PR, CI obrigatório)."

Write-Host "Concluído."
