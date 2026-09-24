# ADR 0001: escolha de provedores

- **Status:** Aceito (provisório)
- **Data:** 2026-09-24
- **Decisor:** mantenedor do laboratório
- **Relacionado:** issue #10; `SPEC.md` seção 4 (Provedores em nuvem)

## Contexto

Todo acesso a modelos passa por um único cliente
(`src/lab/llm/client.py`) baseado no SDK `openai`, configurado por
`base_url` e `api_key` (spec 4.1). O laboratório precisa de quatro papéis de
modelo (spec 4.2):

- `generator` — gera as respostas das estratégias;
- `judge` — LLM-as-judge na avaliação; **deve ser de família diferente do
  gerador**, para reduzir viés de autoavaliação (spec 4.2);
- `embedding` — vetores para o RAG (M4+);
- `synthesizer` — gera o dataset sintético de treino (M7).

A escolha de provedor/modelo impacta diretamente a métrica central do
projeto — **custo** — e a reprodutibilidade (preços e catálogo mudam com
frequência).

## Alternativas

| Opção | Pro | Contra |
|---|---|---|
| OpenAI (direto) | API canônica, catálogo amplo | Não há papel `judge` de família diferente na mesma conta; preços altos; não é "compatível" — é a referência |
| OpenRouter (agregador) | Um endpoint para centenas de modelos (famílias Anthropic, Google, Meta, Mistral, DeepSeek, Qwen...); facilita o requisito de famílias diferentes para `judge` | Agregador: preços são markup do provedor de origem; disponibilidade varia por modelo |
| Together AI | Modelos open-source com preço competitivo; bom para `synthesizer` | Catálogo menor que o de um agregador |
| DeepInfra | Embeddings baratos (`BAAI/bge-m3`); open-source | Foco em open-source; não cobre todos os papéis |
| Fireworks / Groq | Latência baixa | Groq: catálogo limitado; Fireworks: preços por modelo variam |
| Cloud local (self-hosted) | Custo zero marginal | Fora de escopo (spec 1.2): toda inferência é via API |

## Decisão

**Estratégia: provedor principal configurável + provedores fixos por especialidade.**

1. **Provedor principal (`primary`)** — usado pelos papéis `generator`,
   `judge` e `synthesizer`. A URL e a chave ficam em variáveis de ambiente
   (`PRIMARY_BASE_URL`, `PRIMARY_API_KEY`) no `.env`, e os **modelos e preços
   ficam em `config/models.yaml`**, preenchidos na mão. Isso torna a troca de
   provedor uma mudança de configuração, sem código (princípio da spec 4.1).
   - Escolha inicial (a confirmar antes do primeiro run pago):
     - `generator`: um modelo de fronteira acessível via OpenRouter
       (ex.: uma família Anthropic ou Google) — verificar preço e cota na
       página do provedor.
     - `judge`: **de família diferente** do gerador (ex.: se gerador for
       Anthropic, juiz Qwen ou DeepSeek, ou vice-versa) — verificado no
       `models.yaml`.
     - `synthesizer`: um open-source barato (Together AI ou DeepInfra), pois
       o M7 gera volume alto de texto.
2. **Provedores fixos** (URL pública, chave no `.env`):
   - `deepinfra` para `embedding` (`BAAI/bge-m3`, 1024 dims) — embeddings
     multilíngues, barato.
   - `openrouter` e `together` disponíveis para trocar papéis sem código.

**Critérios de escolha dos modelos (documentados, a preencher no momento da
escolha final):**

- custo por 1M de tokens (entrada/saída) na página oficial do provedor;
- `judge` de família diferente de `generator` (obrigatório);
- janela de contexto ≥ 128K para a estratégia `long_context` (M3);
- disponibilidade de `response_format` com JSON Schema (ou fallback
  funcional, já implementado em `chat_json`).

## Consequências

- **Positivas:** troca de provedor sem código; `judge` de família
  independente garantido por configuração; custo por papel auditável no
  `models.yaml`.
- **Negativas:** o `primary` exige duas variáveis no `.env`; o
  `models.yaml` precisa ser mantido à mão quando preços mudam (mitigado
  porque os preços entram no custo estimado e a comparação é relativa).
- **Pendências:** a escolha concreta dos modelos (ids e preços) é
  registrada no próprio `config/models.yaml` no momento em que as chaves de
  API forem provisionadas; este ADR não fixa ids de modelo porque eles
  mudam com frequência.
