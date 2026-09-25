<!-- Título no padrão de commits: <tipo>(<escopo>): <descrição> (#<issue>) — ver docs/commits.md -->

Closes #

## O quê e por quê

<!-- O que muda e qual problema resolve. O diff mostra o como. -->

## Métricas (obrigatório em PRs `exp`)

<!-- Tabela de métricas do run (results/<run>/metrics.json), comparada com a referência anterior. Remover a seção se não for `exp`. -->

| Estratégia | Correção | Citação | Recall@5 | Custo (US$) | p50 (ms) |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Checklist (SPEC 8.5)

- [ ] CI verde (ruff, pyright, pytest)
- [ ] Código novo com testes; cobertura de `src/` ≥ 80%
- [ ] Nenhuma chamada real de API em testes unitários (`@pytest.mark.api` para as reais)
- [ ] README/docs atualizados se o comportamento visível mudou
- [ ] Entrada em `CHANGELOG.md` (Unreleased)
- [ ] Diff com até ~400 linhas (sem contar dados e resultados)
- [ ] Nenhum segredo no diff, em logs ou em resultados
- [ ] Auto-revisão do diff completo no GitHub
