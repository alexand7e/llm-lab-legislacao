# Changelog

Todas as mudanças notáveis neste projeto ficam registradas aqui, na seção
`Unreleased` primeiro e depois movidas para a versão correspondente ao fechar
um milestone (uma versão menor por milestone, conforme `SPEC.md` seção 9).

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## Unreleased

### Adicionado (M0 — Fundação)

- `src/lab/llm/`: cliente OpenAI-compatible com papéis de modelo,
  retentativas com backoff exponencial (429/5xx), timeout por papel e
  registro de custo por chamada.
- `src/lab/llm/cache.py`: cache de respostas em SQLite, chaveado por hash de
  (modelo, mensagens, parâmetros); reruns gratuitos e reproduzíveis.
- `src/lab/llm/costs.py`: estimativa de custo em US$ a partir dos preços por
  1M de tokens em `config/models.yaml`.
- `src/lab/settings.py`: settings via `pydantic-settings` e loader tipado de
  `config/models.yaml` (papéis + provedores, com suporte a `base_url_env`).
- `src/lab/cli.py`: comando `lab chat` para teste manual de um papel de
  modelo (resposta no stdout; tokens, custo e latência no stderr).
- `config/models.yaml`: papéis `generator`, `judge`, `embedding` e
  `synthesizer` + provedores `primary`, `openrouter`, `deepinfra`, `together`.
- `.env.example`: template de variáveis de ambiente.
- `tests/test_llm_client.py`: testes unitários do cliente, cache e custos
  (HTTP mockado, sem chamada real de API).
- `bootstrap_github.ps1`: versão PowerShell do bootstrap (labels,
  milestones, issues e proteção da `main`).
- Docstrings em português em `lab.llm.client`, `lab.llm.cache`,
  `lab.settings`, `lab.llm.costs` e `lab.cli`.
- `README.md`: visão geral, instalação, configuração, uso de `lab chat`,
  aviso de que não constitui aconselhamento jurídico.
- `CHANGELOG.md`: este arquivo.
- `LICENSE` (Apache-2.0).
- `docs/adr/0001-escolha-de-provedores.md`: ADR da estratégia de provedores
  (principal configurável + especialistas; `judge` de família diferente).
- `docs/commits.md` e `.gitmessage`: padrão de commits em português
  (Conventional Commits).

### Pendente (M0)

- Workflow de CI no GitHub Actions — issue #3.
- Templates de issue e de PR — issue #4.
