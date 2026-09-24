# Padrão de commits

Toda mensagem de commit segue Conventional Commits, **em português**, no formato:

```
<tipo>(<escopo>): <descrição no imperativo> (#<issue>)
```

## Regras

1. **Idioma:** a descrição é sempre em português (imperativo, sem ponto final).
2. **Tipo obrigatório.** Escopo e issue são opcionais, mas usados sempre que se aplica:
   - o escopo corresponde a um módulo de `src/lab/` (`llm`, `ingest`, `index`, `graph`, `strategies`, `eval`, `finetune`, `cli`) ou a uma área de projeto (`config`, `ci`, `tests`, `repo`);
   - a issue é o número da issue do GitHub que motivou a mudança (fluxo: issue → branch → PR, seção 8.1 da spec).
3. **Um commit por mudança lógica.** Se a mudança toca mais de um assunto, dividir.
4. **Assunto com no máximo 72 caracteres.** O corpo (opcional, separado por linha em branco) explica o *porquê*, nunca o *quê* — o diff mostra o quê.
5. **Nenhum segredo** em mensagem ou diff.

## Tipos

| Tipo | Uso |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção |
| `refactor` | Mudança sem alterar comportamento |
| `test` | Testes |
| `docs` | Documentação (README, ADRs, CHANGELOG, comentários de docstring) |
| `exp` | Experimento e seus resultados |
| `data` | Alteração em dados versionados |
| `chore` | Build, CI, dependências, infraestrutura de repositório |

## Exemplos

```
feat(llm): adicionar cliente OpenAI-compatible com retentativas e custo (#7)

fix(cli): não travar quando a variável de base_url não existe (#5)

test(llm): cobrir cache de respostas com httpx2 mockado (#8)

docs: adicionar ADR 0001 de escolha de provedores (#10)

chore(ci): adicionar workflow de GitHub Actions (#3)

data: adicionar articles.jsonl da LGPD (v1, coletado em 2026-09-24) (#55)
```

## Configuração local

O repositório carrega o padrão via `.gitmessage`:

```bash
git config commit.template .gitmessage
```

O template é apenas um esqueleto comentado — as regras acima valem.
