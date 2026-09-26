# Changelog

Todas as mudanças notáveis neste projeto ficam registradas aqui, na seção
Unreleased primeiro e depois movidas para a versão correspondente ao fechar
um milestone (uma versão menor por milestone, conforme SPEC.md seção 9).

Formato baseado em Keep a Changelog.

## Unreleased

### Adicionado (M2 — Avaliação)

- src/lab/eval/dataset.py: schema `Question` (SPEC 7.1) e `load_questions`,
  que recusa id repetido, gabarito incoerente com a categoria e linha inválida
  (a mensagem traz arquivo, linha e campo); `check_against_corpus` confere que
  o gabarito existe e está vigente; `distribution_gaps` mostra o que falta
  para as 80 perguntas oficiais.
- src/lab/eval/metrics.py: métricas determinísticas (citação, recall@k, MRR,
  abstenção) e agregação com custo e latência p50/p95. Métrica que não se
  aplica é `None`, nunca zero (o baseline não recupera, então não tem MRR).
- data/eval/questions.dev.jsonl: 16 perguntas de desenvolvimento (`dev_draft`),
  fora do conjunto congelado; servem para testar o pipeline. As 80 oficiais
  são escritas à mão (issue #20).

### Adicionado (M3 — Baseline de prompting)

- src/lab/strategies: interface `Strategy` e `Answer` (texto, artigos citados,
  ids recuperados, confiança, tokens/custo e latência), `normalize_citations`
  e `load_prompt`.
- Estratégia `baseline` (zero-shot, sem recuperação) com o prompt versionado
  `prompts/baseline_v1.md` e saída estruturada (`answer`, `cited_articles`,
  `confidence`). Resposta fora do schema vira texto bruto com `parse_error`,
  sem perder o custo da chamada.
- `LLMClient.chat_parsed`: saída estruturada que não levanta quando a resposta
  não valida e devolve sempre o `ChatResult`.
- `lab ask "<pergunta>"`: pergunta ao baseline; mostra artigos citados,
  confiança, tokens e custo.

### Corrigido (M0 — Fundação)

- As chaves e URLs dos provedores no `.env` não chegavam ao cliente (só o
  ambiente do sistema era lido); `lab.settings.load_env` junta o `.env` com
  o ambiente, que tem precedência.
- `config/models.yaml` com os modelos reais do provedor principal
  (generator, judge, embedding e synthesizer); aceite do M0 verificado:
  `lab chat --role generator "olá"` responde e mostra tokens e custo.

### Adicionado (M1 — Corpus)

- config/corpus.yaml + src/lab/ingest/sources.py: normas do corpus como
  configuração (loader tipado, caminho via LAB_CORPUS_CONFIG); nenhum
  código depende de uma norma específica.
- src/lab/ingest/collect.py: coletor do HTML bruto em data/raw/, com
  URL, sha256 dos bytes e data de coleta em `<lei>.meta.json`.
- Dependência httpx2 (já transitiva via openai) declarada diretamente.
- tests/fixtures/: recortes de HTML real para os testes do parser, gerados
  por scripts/make_fixtures.py a partir de trechos em selections.yaml.
- Dependências beautifulsoup4 e lxml.
- src/lab/ingest/parser.py: parser da hierarquia (Livro → ... → Subseção
  → Artigo → Parágrafo → Inciso → Alínea), genérico para normas no
  formato do Planalto; descarta redações riscadas e anotações de alteração.
- Status `vigente | revogado | vetado` por artigo e por dispositivo; o
  texto do artigo inclui só os dispositivos vigentes.
- Artigos acrescidos (ex.: `55-A`) como artigos próprios, com ordenação
  natural (`article_key`: 55 < 55-A < 55-B < 56).
- src/lab/ingest/references.py: remissões internas ("art. 7º", intervalos
  como "arts. 7º a 11") e externas ("art. 5º da Lei nº 8.078"), como
  `{target, raw}`; normas do corpus resolvem para o id do corpus
  (`cdc:art:5`), as demais para uma chave estável (`lei:7347:art:1`).
  Ignora texto entre aspas, anotações de alteração e dispositivos revogados.
- src/lab/ingest/export.py: registros `ArticleRecord` (schema da SPEC 2.3,
  com `key`, `parent` e `status` por unidade), validados ao criar e ao ler;
  `write_jsonl` grava `articles.jsonl` de forma determinística (normas na
  ordem do corpus, artigos em ordem natural) e recusa ids repetidos;
  `read_jsonl` aponta a linha inválida; `dangling_references` alerta sobre
  remissões para artigos do corpus que não existem.
- `lab ingest` (src/lab/ingest/pipeline.py): coleta o que falta em
  `data/raw/` (ou tudo, com `--fetch`), confere o hash, parseia as normas
  de `config/corpus.yaml` e grava `data/processed/articles.jsonl`; se uma
  norma falhar nada é gravado. Imprime o resumo por norma e avisa sobre
  remissões para artigos ausentes.
- data/processed/articles.jsonl versionado (LGPD, Marco Civil, CDC) e
  tests/test_corpus.py, que trava a contagem oficial dos artigos originais
  (65, 32 e 119, sem lacunas) e invariantes do schema.
- scripts/sample_articles.py: sorteia 20 artigos (semente fixa, priorizando
  acrescidos, revogados e vetados) para a conferência manual do aceite do M1.

### Adicionado (M0 — Fundação)

- src/lab/llm/: cliente OpenAI-compatible com papéis de modelo,
  retentativas com backoff exponencial (429/5xx), timeout por papel e
  registro de custo por chamada.
- src/lab/llm/cache.py: cache de respostas em SQLite, chaveado por hash de
  (modelo, mensagens, parâmetros); reruns gratuitos e reproduzíveis.
- src/lab/llm/costs.py: estimativa de custo em US$ a partir dos preços por
  1M de tokens em config/models.yaml.
- src/lab/settings.py: settings via pydantic-settings e loader tipado de
  config/models.yaml (papéis + provedores, com suporte a base_url_env).
- src/lab/cli.py: comando lab chat para teste manual de um papel de
  modelo (resposta no stdout; tokens, custo e latência no stderr).
- config/models.yaml: papéis generator, judge, embedding e
  synthesizer + provedores primary, openrouter, deepinfra, together.
- .env.example: template de variáveis de ambiente.
- tests/test_llm_client.py: testes unitários do cliente, cache e custos
  (HTTP mockado, sem chamada real de API).
- tests/test_settings.py: testes de Settings (.env e ambiente) e do
  loader de config/models.yaml, incluindo o arquivo versionado.
- tests/test_cli.py: testes do comando lab chat com cliente falso.
- .pre-commit-config.yaml: hooks de higiene, ruff e pyright (via uv run,
  mesmas versões do uv.lock).
- Gate de cobertura de 80% em src/ (pyproject + `pytest --cov` no CI).
- .github/: template de PR com o checklist da definição de pronto e
  formulários de issue (funcionalidade, bug, experimento, tarefa).
- Proteção da main conforme SPEC 8.6 (merge só via PR, check ci
  obrigatório, vale para admins) e só merge commit; bootstrap_github.sh e
  .ps1 aplicam a mesma configuração.
- bootstrap_github.ps1: versão PowerShell do bootstrap (labels,
  milestones, issues e proteção da main).
- Docstrings em português em lab.llm.client, lab.llm.cache,
  lab.settings, lab.llm.costs e lab.cli.
- README.md: visão geral, instalação, configuração, uso de lab chat,
  aviso de que não constitui aconselhamento jurídico.
- CHANGELOG.md: este arquivo.
- LICENSE (Apache-2.0).
- docs/adr/0001-escolha-de-provedores.md: ADR da estratégia de provedores
  (principal configurável + especialistas; judge de família diferente).
- docs/commits.md e .gitmessage: padrão de commits em português
  (Conventional Commits).
