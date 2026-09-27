// Interface local do laboratório: liga o chat-reference à API (/api/chat, SSE).
// O renderer (chat-renderer.js) é cópia fiel do chat-reference; aqui só há
// estado, transporte e os acréscimos da interface: fonte ausente, estatísticas,
// logs recolhidos (passos, raciocínio e saída bruta) e o contexto da conversa.

// O renderer importa './chat-messages.css', o que o navegador não sabe fazer; o
// CSS já vem por <link>. Carrega o arquivo sem essa linha, sem alterá-lo em disco.
const source = (await (await fetch('/chat-renderer.js')).text())
  .replace(/^import '\.\/chat-messages\.css'\s*$/m, '')
const R = await import(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })))

const $ = (id) => document.getElementById(id)
const els = {
  scroll: $('scroll'), messages: $('messages'), empty: $('empty'), examples: $('examples'),
  form: $('composer'), input: $('input'), send: $('send'), subtitle: $('app-subtitle'), reset: $('new-chat'),
}

// Ícone neutro do assistente (balança); o reference aceita avatarUrl.
const AVATAR = 'data:image/svg+xml;utf8,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="#155FAC">' +
  '<path d="M239.5 148.6l-32-72a8 8 0 00-1.6-2.4A8 8 0 00200 72h-64V49.4a24 24 0 10-16 0V72H56a8 8 0 00-5.9 2.2 8 8 0 00-1.6 2.4l-32 72A8 8 0 0016 152c0 22.1 20.9 32 40 32s40-9.9 40-32a8 8 0 00-.7-3.2L68.6 88H120v96H96a8 8 0 000 16h64a8 8 0 000-16h-24V88h51.4l-27.7 60.8A8 8 0 00160 152c0 22.1 20.9 32 40 32s40-9.9 40-32a8 8 0 00-.5-3.4zM56 168c-9.3 0-19.6-3.4-22.9-10.4L56 106.6l22.9 51c-3.3 7-13.6 10.4-22.9 10.4zm144 0c-9.3 0-19.6-3.4-22.9-10.4l22.9-51 22.9 51c-3.3 7-13.6 10.4-22.9 10.4z"/></svg>')

const EXAMPLES = [
  ['Qual o prazo para o controlador confirmar o tratamento em formato simplificado?', 'LGPD · o baseline costuma errar o artigo'],
  ['Por quanto tempo o provedor de aplicações deve guardar os registros de acesso?', 'Marco Civil da Internet'],
  ['Em quanto tempo o consumidor pode desistir de uma compra feita por telefone?', 'Código de Defesa do Consumidor'],
  ['Qual a alíquota do imposto de renda sobre o salário?', 'Fora do corpus · deve se abster'],
]

// ?logs=1 abre os logs de cada resposta por padrão (depuração).
const LOGS_OPEN = new URLSearchParams(location.search).get('logs') === '1'
const HISTORY_LIMIT = 20 // mensagens anteriores enviadas; o servidor usa só as mais recentes
const state = { messages: [], busy: false, current: null }

const newReply = () => ({
  role: 'assistant',
  t0: performance.now(),
  streaming: true,
  plain: '', // só o texto da resposta (sem chips de citação): vira contexto das próximas perguntas
  logs: { reasoning: '', raw: '' },
  ui: { toolOpen: LOGS_OPEN, reasoningOpen: false, rawOpen: false },
})

/* ─── render ─────────────────────────────────────────────── */
let scheduled = false
let forceScroll = false

// Vários eventos por quadro (streaming) viram um único render.
function scheduleRender() {
  if (scheduled) return
  scheduled = true
  requestAnimationFrame(() => { scheduled = false; renderNow() })
}

function renderNow() {
  const assistant = state.messages.filter((m) => m.role === 'assistant')
  const box = els.scroll
  const nearBottom = forceScroll || box.scrollHeight - box.scrollTop - box.clientHeight < 90
  forceScroll = false

  for (const m of assistant) {
    if (!m.tool) continue
    m.tool.collapsed = !(m.ui.toolOpen || m.tool.error) // logs recolhidos, salvo erro
    m.tool.summary = summaryOf(m)
  }
  els.empty.hidden = state.messages.length > 0
  els.messages.replaceChildren(R.renderMessages(state.messages, {
    userInitials: 'VC', assistantName: 'Assistente', avatarUrl: AVATAR,
    typing: state.busy && !state.messages.includes(state.current),
  }))
  decorate(assistant)
  if (nearBottom) box.scrollTop = box.scrollHeight
}

function summaryOf(m) {
  const steps = m.tool.steps || []
  if (m.streaming) {
    const running = steps.filter((s) => s.state === 'running').pop()
    const thinking = m.logs.reasoning.length
    if (running && running.label === 'Consultando o modelo' && thinking) {
      return `Raciocinando… ${thinking.toLocaleString('pt-BR')} caracteres`
    }
    return `${running ? running.label : 'Respondendo'}…`
  }
  const secs = m.t1 ? ` · ${((m.t1 - m.t0) / 1000).toFixed(1)} s` : ''
  return `Logs · ${steps.length} etapas${secs}`
}

// Acréscimos que o reference não tem: fonte ausente/revogada, logs, cursor e estatísticas.
function decorate(assistant) {
  const nodes = els.messages.querySelectorAll('.chat-msg-ai')
  assistant.forEach((m, i) => {
    const node = nodes[i]
    if (!node) return
    for (const s of m.sources || []) {
      const card = node.querySelector(`.chat-source-card[data-source-n="${s.n}"]`)
      if (!card) continue
      if (!s.exists) card.classList.add('is-missing')
      else if (s.status !== 'vigente') card.classList.add('is-status')
    }
    const collapsed = node.querySelector('.chat-tc-collapsed i:first-child')
    if (collapsed && m.streaming) collapsed.className = 'ph ph-circle-notch chat-spin'
    const card = node.querySelector('.chat-tool-card')
    if (card) appendLogs(card, m)
    if (m.streaming) node.querySelector('.chat-ai-text')?.appendChild(caret())
    if (m.stats) node.querySelector('.chat-ai-body')?.appendChild(statsLine(m.stats))
  })
}

function caret() {
  const span = document.createElement('span')
  span.className = 'app-caret'
  return span
}

// Logs do modelo dentro do card expandido: consultáveis, mas fechados por padrão.
function appendLogs(card, m) {
  const sections = [
    ['reasoning', 'reasoningOpen', 'Raciocínio do modelo'],
    ['raw', 'rawOpen', 'Saída bruta do modelo (JSON)'],
  ]
  for (const [channel, flag, title] of sections) {
    const text = m.logs[channel]
    if (!text) continue
    const details = document.createElement('details')
    details.className = 'app-log'
    details.open = m.ui[flag]
    details.addEventListener('toggle', () => { m.ui[flag] = details.open })
    const summary = document.createElement('summary')
    summary.textContent = `${title} · ${text.length.toLocaleString('pt-BR')} caracteres`
    const pre = document.createElement('pre')
    pre.textContent = text
    details.append(summary, pre)
    card.appendChild(details)
    pre.scrollTop = pre.scrollHeight // acompanha o fim enquanto chega
  }
}

function statsLine(s) {
  const line = document.createElement('div')
  line.className = 'app-stats'
  const add = (text, warn) => {
    const span = document.createElement('span')
    span.textContent = text
    if (warn) span.className = 'is-warn'
    line.appendChild(span)
  }
  add(`${s.prompt_tokens} → ${s.completion_tokens} tokens`)
  add(`$${s.cost_usd.toFixed(6)}`)
  add(s.cached ? 'cache' : `${(s.latency_ms / 1000).toFixed(1)} s`)
  if (s.confidence != null) add(`confiança ${Math.round(s.confidence * 100)}%`)
  if (s.missing) add(`${s.missing} de ${s.cited} artigos citados não existem no corpus`, true)
  if (s.parse_error) add('resposta fora do formato esperado', true)
  return line
}

// O reference expande/recolhe o card só no DOM, e cada render o recria. Guardamos
// o estado e refazemos o render: capturamos o clique antes do handler do reference.
els.messages.addEventListener('click', (event) => {
  const open = event.target.closest('.chat-tc-collapsed')
  const close = event.target.closest('.chat-tc-collapse')
  const button = open || close
  if (!button) return
  const nodes = [...els.messages.querySelectorAll('.chat-msg-ai')]
  const m = state.messages.filter((x) => x.role === 'assistant')[nodes.indexOf(button.closest('.chat-msg-ai'))]
  if (!m) return
  event.stopPropagation()
  event.preventDefault()
  m.ui.toolOpen = Boolean(open)
  scheduleRender()
}, true)

/* ─── transporte (SSE sobre fetch) ────────────────────────── */
async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      let event = 'message', data = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (data) onEvent(event, JSON.parse(data))
    }
  }
}

// Conversa até aqui, para o modelo entender perguntas como "quem fez essa lei?".
function historyOf() {
  const history = []
  for (const m of state.messages) {
    if (m.role === 'user') history.push({ role: 'user', content: m.text })
    else if (m.plain.trim()) history.push({ role: 'assistant', content: m.plain.trim() })
  }
  return history.slice(-HISTORY_LIMIT)
}

function handleEvent(reply, event, data) {
  if (!state.messages.includes(reply)) state.messages.push(reply) // 1º evento tira o "digitando"
  if (event === 'done') {
    reply.stats = data.stats
    reply.streaming = false
    reply.t1 = performance.now()
  } else if (event === 'log') {
    reply.logs[data.channel] = (reply.logs[data.channel] || '') + data.delta
  } else {
    if (event === 'text_chunk') { reply.plain += data.delta; reply.streamed = true }
    else if (event === 'markdown' && !reply.streamed) reply.plain = data.md.replace(/\n*Artigos citados:[^\n]*$/, '')
    else if (event === 'sources') reply.sources = data.sources
    R.applyEvent(reply, event, data)
  }
  scheduleRender()
}

function setBusy(busy) {
  state.busy = busy
  els.send.disabled = busy || !els.input.value.trim()
  els.input.disabled = busy
}

async function ask(question) {
  const text = question.trim()
  if (!text || state.busy) return
  const history = historyOf() // antes de acrescentar a pergunta atual
  state.messages.push({ role: 'user', text })
  const reply = state.current = newReply()
  setBusy(true)
  forceScroll = true
  renderNow()
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, history }),
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('; ') : body.detail
      throw new Error(detail || `HTTP ${response.status}`)
    }
    await consumeSSE(response, (event, data) => handleEvent(reply, event, data))
  } catch (error) {
    if (!state.messages.includes(reply)) state.messages.push(reply)
    R.applyEvent(reply, 'tool_error', { title: 'Erro de conexão', detail: String(error.message || error) })
  } finally {
    reply.streaming = false
    setBusy(false)
    renderNow()
    els.input.focus()
  }
}

/* ─── composer e ações ───────────────────────────────────── */
function autosize() {
  els.input.style.height = 'auto'
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + 'px'
  els.send.disabled = state.busy || !els.input.value.trim()
}

els.input.addEventListener('input', autosize)
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit() }
})
els.form.addEventListener('submit', (e) => {
  e.preventDefault()
  const text = els.input.value
  els.input.value = ''
  autosize()
  ask(text)
})
els.reset.addEventListener('click', () => {
  if (state.busy) return
  state.messages = []
  state.current = null
  renderNow()
  els.input.focus()
})

for (const [question, hint] of EXAMPLES) {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'app-example'
  button.append(question)
  const small = document.createElement('small')
  small.textContent = hint
  button.appendChild(small)
  button.addEventListener('click', () => ask(question))
  els.examples.appendChild(button)
}

fetch('/api/health')
  .then((r) => r.json())
  .then((h) => { els.subtitle.textContent = `${h.strategy} · ${h.model} · ${h.articles} artigos no corpus` })
  .catch(() => { els.subtitle.textContent = 'servidor indisponível' })

renderNow()
els.input.focus()

// ?ask=<pergunta> envia a pergunta ao abrir (link compartilhável).
const preset = new URLSearchParams(location.search).get('ask')
if (preset) ask(preset)
