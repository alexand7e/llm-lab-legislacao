# ADR 0002: estratégia de chunking

- Status: Proposto (aguarda o mantenedor e a confirmação com o conjunto oficial)
- Data: 2026-09-28
- Decisor: mantenedor do laboratório
- Relacionado: issues #35, #36, #38, #39; SPEC.md seção 8.7 (ADRs obrigatórios)

## Contexto

O RAG vetorial (M4) recupera trechos do corpus por similaridade e responde só
com eles. A unidade do trecho (chunk) define o que a busca consegue achar e
quanto texto vai ao gerador. A avaliação mede recuperação por **artigo**
(recall@k e MRR contra `gold_articles`), e a resposta cita artigos; então todo
chunk precisa saber de que artigo veio, qualquer que seja a granularidade.

O texto legal tem estrutura própria: artigo → caput / parágrafos → incisos →
alíneas. Artigos variam de uma linha a ~6 mil caracteres. Perguntas reais
costumam depender de um dispositivo específico ("em formato simplificado" está
no inciso I do caput do art. 19 da LGPD) ou de dois artigos de leis diferentes.

## Alternativas

| Opção | Pró | Contra |
|---|---|---|
| Janela fixa de tokens (ex.: 512 com sobreposição) | Genérica, independe do parser | Corta dispositivos no meio; o trecho perde o número do artigo; ignora a estrutura que o parser já extraiu |
| Por artigo (`article`, #35) | Unidade natural da citação; 1 chunk = 1 artigo; simples | Artigos longos diluem o embedding; poucos artigos distintos cabem no orçamento de tokens |
| Por dispositivo: caput e cada parágrafo, com seus incisos e alíneas (`unit`, #36) | Trecho focado; cabe mais artigos distintos no mesmo orçamento; incisos ficam com o dispositivo a que pertencem | Mais chunks (509 × 223); exige k maior; parágrafo isolado precisa do número do artigo no texto |
| Por inciso/alínea | Máxima precisão | Fragmentos curtos demais, sem o caput que dá sentido ("I - imediatamente;" sozinho não diz nada) |

Em todas as opções baseadas na estrutura, o texto de embedding leva um
cabeçalho com a norma e a hierarquia (capítulo, seção), e o texto que vai ao
gerador é o oficial vigente (dispositivos revogados e vetados ficam de fora).

## Evidência (16 perguntas de desenvolvimento, não oficial)

Detalhes em `docs/experiments/2026-09-28-rag-variacao-k.md` e
`docs/experiments/2026-09-28-rag-chunking-dispositivo.md`.

| configuração | correção | recall@10 | tokens de entrada / pergunta |
|---|---|---|---|
| article, k = 3 | 0,84 | 0,85 | 1.447 |
| article, k = 5 | 0,91 | 0,92 | 2.158 |
| article, k = 10 | 0,88 | 0,96 | 3.562 |
| unit, k = 5 | 0,81 | 0,92 | 1.089 |
| **unit, k = 10** | **0,97** | **1,00** | **1.788** |
| long_context (sem busca, referência) | 1,00 | — | 41.735 |

## Decisão (proposta)

Chunking **por dispositivo** (`unit`: caput e parágrafos, com incisos e alíneas)
com **k = 10** como configuração padrão do RAG a partir do M4. O chunking por
artigo continua disponível (`--granularity article`) como linha de base.

Motivos: maior correção entre as configurações testadas, única a recuperar o
segundo artigo das perguntas de duas partes, e menos tokens que a melhor
configuração por artigo.

## Consequências

- `lab index -g unit` passa a ser o índice do M4; `lab eval --strategy rag`
  deve usar `-g unit --k 10` (hoje os padrões da CLI ainda são `article` e 5:
  trocar os padrões é uma mudança separada, depois de aceito este ADR).
- O k agora depende da granularidade: runs com granularidades diferentes só
  são comparáveis com k ajustado; `lab compare` mostra ambos no rótulo.
- Reavaliar com as 80 perguntas oficiais (#20). Com 16 perguntas, uma resposta
  muda a média em ~0,06: a diferença entre `article k=5` e `unit k=10` é
  sugestiva, não conclusiva. Se o resultado oficial inverter a ordem, este ADR é
  revisto.
- Técnicas do M5 (busca híbrida #40, reranking #41, reescrita de consulta #42)
  partem desta configuração.
