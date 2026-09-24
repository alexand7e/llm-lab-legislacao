# chat-reference

Referência de como as mensagens de um chat com IA são renderizadas (Markdown, citações `[^n]`,
cards de ferramenta com passos, painel de fontes, listas selecionáveis).

- `chat-renderer.js`: funções puras que recebem dados e devolvem nós DOM. Sem rede, sem auth, sem estado global.
- `chat-messages.css`: estilos correspondentes.

Dependências: `marked`, `dompurify`, Phosphor Icons (CSS).

Modelo de mensagem: `{ role, text?, md?, meta?, tool?, sources? }`.
Eventos de streaming são aplicados com `applyEvent(reply, event, data)`.
