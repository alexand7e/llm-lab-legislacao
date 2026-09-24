// Referencia de como as mensagens de um chat sao renderizadas no DOM.
// Sem rede, sem autenticacao, sem estado global: recebe dados, devolve nos DOM.
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import './chat-messages.css'

marked.setOptions({ gfm: true, breaks: true })

/* ─── helpers ─────────────────────────────────────────────── */
export function h(tag, props = {}, children = []) {
    const node = document.createElement(tag)
    for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue
        if (k === 'class') node.className = v
        else if (k === 'text') node.textContent = v
        else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v)
        else if (k === 'dataset') Object.assign(node.dataset, v)
        else node.setAttribute(k, v)
    }
    for (const c of Array.isArray(children) ? children : [children]) {
        if (c == null || c === false) continue
        node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)
    }
    return node
}

const icon = (name) => h('i', { class: `ph ph-${name}` })
const iconBold = (name) => h('i', { class: `ph-bold ph-${name}` })
const spinIcon = () => h('i', { class: 'ph ph-circle-notch chat-spin' })

const formatBRL = (v) =>
    v == null || Number.isNaN(Number(v))
        ? 'Não informado'
        : Number(v).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' })

/* ─── markdown + citacoes [^n] ────────────────────────────── */
export function renderMarkdown(md, { onCiteHover, onCiteJump } = {}) {
    if (!md) return []
    const withCites = md.replace(/\[\^(\d+)\]/g, (_, n) => `<chat-cite n="${n}"></chat-cite>`)
    const safe = DOMPurify.sanitize(marked.parse(withCites), {
        ADD_TAGS: ['chat-cite'],
        ADD_ATTR: ['n', 'target', 'rel'],
    })
    const tpl = document.createElement('template')
    tpl.innerHTML = safe

    tpl.content.querySelectorAll('chat-cite').forEach((el) => {
        const n = parseInt(el.getAttribute('n'), 10)
        el.replaceWith(h('button', {
            class: 'chat-cite-chip',
            dataset: { citeN: String(n) },
            text: String(n),
            onmouseenter: () => onCiteHover?.(n),
            onmouseleave: () => onCiteHover?.(null),
            onclick: (e) => { e.preventDefault(); onCiteJump?.(n) },
        }))
    })
    tpl.content.querySelectorAll('a[href]').forEach((a) => {
        a.classList.add('chat-md-link')
        a.setAttribute('target', '_blank')
        a.setAttribute('rel', 'noreferrer')
    })
    tpl.content.querySelectorAll('table').forEach((t) => {
        const wrap = h('div', { class: 'chat-md-table-wrap' })
        t.parentNode.insertBefore(wrap, t)
        wrap.appendChild(t)
    })
    return Array.from(tpl.content.childNodes)
}

// Modelos com raciocinio emitem "<think>...</think>" antes da resposta: descarta.
export function stripThink(md) {
    if (!md) return md
    const i = md.indexOf('</think>')
    return i === -1 ? md : md.slice(i + '</think>'.length).replace(/^\s+/, '')
}

/* ─── resultados de ferramenta (registry por `kind`) ───────── */
function renderFileResult(file) {
    const ext = (file.filename || '').split('.').pop().toUpperCase()
    const ctaLabel = ext && ext.length <= 5 ? `Baixar ${ext}` : 'Baixar arquivo'
    return h('div', { class: 'chat-file-result' }, [
        h('div', { class: 'chat-fr-thumb' }, [icon('file-text')]),
        h('div', { class: 'chat-fr-body' }, [
            h('div', { class: 'chat-fr-title', text: file.title }),
            h('div', { class: 'chat-fr-sub', text: file.subtitle }),
            file.meta && h('div', { class: 'chat-fr-meta' }, file.meta.map((m) => h('span', { text: m }))),
        ]),
        h('div', { class: 'chat-fr-actions' }, [
            h('a', { class: 'chat-fr-cta', href: file.download_url || '#', target: '_blank', rel: 'noreferrer' },
                [icon('download-simple'), h('span', { text: ctaLabel })]),
            h('a', { class: 'chat-fr-ghost', href: file.view_url || '#', target: '_blank', rel: 'noreferrer' },
                [icon('arrow-square-out'), h('span', { text: 'Visualizar' })]),
        ]),
    ])
}

function renderItemCompare(r) {
    const s = r.summary || {}
    const items = r.items || []
    const stat = (label, value) => h('div', { class: 'chat-ic-stat' }, [
        h('span', { class: 'chat-ic-stat-label', text: label }),
        h('strong', { text: value }),
    ])
    return h('div', { class: 'chat-ic-result' }, [
        h('div', { class: 'chat-ic-head' }, [icon('chart-bar'), h('span', { text: `${s.items_total || items.length} itens encontrados` })]),
        h('div', { class: 'chat-ic-stats' }, [
            stat('Grupos', String(s.distinct_groups ?? '0')),
            stat('Com valor unitário', String(s.unit_values_count ?? '0')),
            stat('Mediana unitária', formatBRL(s.unit_median)),
        ]),
        (r.warnings || []).length
            ? h('ul', { class: 'chat-ic-warnings' }, r.warnings.map((w) => h('li', { text: w })))
            : null,
        h('ul', { class: 'chat-ic-list' }, items.map((item) => h('li', { class: 'chat-ic-row' }, [
            h('div', { class: 'chat-ic-main' }, [
                h('strong', { text: item.title || '(sem título)' }),
                h('span', { text: [item.group, item.code].filter(Boolean).join(' · ') }),
            ]),
            h('div', { class: 'chat-ic-values' }, [
                h('span', { text: `Unitário: ${formatBRL(item.unit_value)}` }),
                h('span', { text: `Total: ${formatBRL(item.total_value)}` }),
            ]),
        ]))),
    ])
}

// Lista com checkbox; o botao dispara `onAction(actionId, payload, label)`.
function renderSelectableList(r, { onAction }) {
    const selected = new Set()
    const countEl = h('span', { class: 'chat-ps-btn-count', text: '' })
    const btn = h('button', {
        class: 'chat-ps-btn chat-ps-btn-primary chat-ps-btn-lg',
        type: 'button',
        disabled: true,
        onclick: () => selected.size && onAction?.(
            r.action_id, { item_ids: [...selected] }, `${r.action_label} (${selected.size})`),
    }, [icon('chart-bar'), h('span', { text: `${r.action_label} ` }), countEl])

    const rows = (r.items || []).map((item) => {
        const cb = h('input', { type: 'checkbox', class: 'chat-dl-check' })
        cb.addEventListener('change', () => {
            cb.checked ? selected.add(item.id) : selected.delete(item.id)
            btn.disabled = selected.size === 0
            countEl.textContent = selected.size ? `(${selected.size})` : ''
        })
        return h('li', { class: 'chat-ic-row' }, [h('label', { class: 'chat-dl-row' }, [
            cb,
            h('div', { class: 'chat-ic-main' }, [
                h('strong', { text: item.title || '(sem título)' }),
                h('span', { text: item.subtitle || '' }),
            ]),
        ])])
    })
    return h('div', { class: 'chat-ic-result' }, [
        h('div', { class: 'chat-ic-head' }, [icon('list-checks'), h('span', { text: r.title })]),
        h('ul', { class: 'chat-ic-list' }, rows),
        h('div', { class: 'chat-ps-actions chat-ps-actions-center' }, [btn]),
    ])
}

const RESULT_RENDERERS = {
    file: renderFileResult,
    'item-compare': renderItemCompare,
    'selectable-list': renderSelectableList,
}

/* ─── card de ferramenta: loading -> done | error, recolhivel ─ */
export function renderToolCard(tool, ctx = {}) {
    if (tool.collapsed) {
        const card = h('button', { class: 'chat-tc-collapsed', type: 'button' }, [
            iconBold('check'), h('span', { text: tool.summary || tool.title }), icon('caret-down'),
        ])
        card.addEventListener('click', () => card.replaceWith(renderToolCard({ ...tool, collapsed: false }, ctx)))
        return card
    }

    const isError = !!tool.error
    const isDone = !!tool.result && !isError
    const stateClass = isError ? 'chat-tc-error' : isDone ? 'chat-tc-done' : 'chat-tc-loading'
    const headIcon = isError ? icon('warning') : isDone ? iconBold('check') : spinIcon()

    let card
    const head = h('div', { class: 'chat-tc-head' }, [
        h('div', { class: 'chat-tc-ico' }, [headIcon]),
        h('div', { class: 'chat-tc-title', text: tool.title }),
        (tool.result || tool.summary) && h('button', {
            class: 'chat-tc-collapse',
            'aria-label': 'Recolher',
            onclick: () => card.replaceWith(renderToolCard({ ...tool, collapsed: true }, ctx)),
        }, [icon('x')]),
    ])

    const steps = tool.steps && h('ol', { class: 'chat-tc-steps' }, tool.steps.map((s) =>
        h('li', { class: `chat-tc-step is-${s.state}` }, [
            h('span', { class: 'chat-tc-dot' }, [
                s.state === 'done' && iconBold('check'),
                s.state === 'running' && spinIcon(),
                s.state === 'error' && icon('x'),
            ]),
            h('span', { class: 'chat-tc-step-label', text: s.label }),
        ])))

    const render = tool.result && RESULT_RENDERERS[tool.result.kind]
    const result = render ? render(tool.result, ctx) : null

    const errorBody = isError && h('div', { class: 'chat-tc-error-body' }, [
        h('div', { class: 'chat-tc-error-title', text: tool.error.title }),
        h('div', { class: 'chat-tc-error-detail', text: tool.error.detail }),
    ])

    card = h('div', { class: `chat-tool-card ${stateClass}` }, [head, steps, result, errorBody])
    return card
}

/* ─── painel de fontes, ligado aos chips [^n] ─────────────── */
export function renderSourcesPanel(sources, scope, limit = 3) {
    if (!sources?.length) return null
    const setHot = (n) => {
        scope.querySelectorAll('[data-cite-n]').forEach((el) => el.classList.toggle('is-hot', n != null && Number(el.dataset.citeN) === n))
        scope.querySelectorAll('[data-source-n]').forEach((el) => el.classList.toggle('is-hot', n != null && Number(el.dataset.sourceN) === n))
    }
    const cards = sources.map((s, i) => h(s.url ? 'a' : 'div', {
        class: 'chat-source-card' + (s.url ? ' is-link' : ''),
        dataset: { sourceN: String(s.n) },
        onmouseenter: () => setHot(s.n),
        onmouseleave: () => setHot(null),
        ...(s.url ? { href: s.url, target: '_blank', rel: 'noreferrer' } : {}),
        ...(i >= limit ? { style: 'display:none' } : {}),
    }, [
        h('div', { class: 'chat-source-n', text: String(s.n) }),
        h('div', { class: 'chat-source-body' }, [
            h('div', { class: 'chat-source-title', text: s.title }),
            h('div', { class: 'chat-source-ref', text: s.ref }),
            h('div', { class: 'chat-source-snippet', text: `"${s.snippet}"` }),
        ]),
    ]))

    let toggle = null
    if (sources.length > limit) {
        let expanded = false
        const label = h('span', { text: `Ver mais ${sources.length - limit} fontes` })
        toggle = h('button', { class: 'chat-sources-toggle', type: 'button' }, [label, icon('caret-down')])
        toggle.addEventListener('click', () => {
            expanded = !expanded
            cards.forEach((c, i) => { if (i >= limit) c.style.display = expanded ? '' : 'none' })
            toggle.classList.toggle('is-expanded', expanded)
            label.textContent = expanded ? 'Ver menos' : `Ver mais ${sources.length - limit} fontes`
        })
    }
    const panel = h('div', { class: 'chat-sources' }, [
        h('div', { class: 'chat-sources-head' }, [
            h('span', { text: 'Fontes consultadas' }),
            h('span', { class: 'chat-sources-count', text: String(sources.length) }),
        ]),
        h('div', { class: 'chat-sources-grid' }, cards),
        toggle,
    ])
    panel._setHot = setHot
    return panel
}

/* ─── lista de mensagens ──────────────────────────────────── */
export function renderMessages(messages, { userInitials = 'U', assistantName = 'Assistente', avatarUrl, typing, onAction } = {}) {
    const wrap = h('div', { class: 'chat-messages' })
    const avatar = () => h('div', { class: 'chat-ai-avatar' }, avatarUrl ? [h('img', { src: avatarUrl, alt: '' })] : [])

    for (const m of messages) {
        if (m.role === 'user') {
            wrap.appendChild(h('div', { class: 'chat-msg-user' }, [
                h('div', { class: 'chat-bubble', text: m.text }),
                h('div', { class: 'chat-avatar-sm', text: userInitials }),
            ]))
            continue
        }
        const body = h('div', { class: 'chat-ai-body' }, [
            h('div', { class: 'chat-ai-name' }, [
                `${assistantName} `,
                m.meta && h('span', { class: 'chat-ai-tag', text: m.meta.model || '' }),
            ]),
        ])
        const text = h('div', { class: 'chat-ai-text' })
        if (m.text) text.appendChild(h('p', { class: 'chat-md-p', text: m.text }))
        if (m.tool) text.appendChild(renderToolCard(m.tool, { onAction }))

        const sourcesPanel = m.sources?.length ? renderSourcesPanel(m.sources, body) : null
        if (m.md) {
            const nodes = renderMarkdown(stripThink(m.md), {
                onCiteHover: (n) => sourcesPanel?._setHot(n),
                onCiteJump: (n) => {
                    sourcesPanel?._setHot(n)
                    sourcesPanel?.querySelector(`[data-source-n="${n}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
                },
            })
            nodes.forEach((n) => text.appendChild(n))
        }
        body.appendChild(text)
        if (sourcesPanel) body.appendChild(sourcesPanel)
        wrap.appendChild(h('div', { class: 'chat-msg-ai' }, [avatar(), body]))
    }

    if (typing) {
        wrap.appendChild(h('div', { class: 'chat-msg-ai' }, [avatar(), h('div', { class: 'chat-ai-body' }, [
            h('div', { class: 'chat-typing' }, [h('span'), h('span'), h('span')]),
        ])]))
    }
    return wrap
}

/* ─── eventos de streaming -> modelo de mensagem ──────────────
   Cada evento (SSE ou outro transporte) atualiza o objeto `reply`;
   depois basta chamar renderMessages() de novo.                  */
export function applyEvent(reply, event, data) {
    switch (event) {
        case 'meta': reply.meta = data; break
        case 'tool_start': reply.tool = { title: data.title, steps: [] }; break
        case 'tool_step': {
            reply.tool ??= { title: 'Processando', steps: [] }
            const s = reply.tool.steps.find((x) => x.label === data.label)
            if (s) s.state = data.state
            else reply.tool.steps.push({ label: data.label, state: data.state })
            break
        }
        case 'tool_done':
            reply.tool ??= { title: 'Pronto', steps: [] }
            reply.tool.result = data.result
            reply.tool.steps.forEach((x) => { if (x.state === 'running') x.state = 'done' })
            break
        case 'tool_error':
            reply.tool ??= { title: 'Erro', steps: [] }
            reply.tool.error = data
            reply.tool.steps.forEach((x) => { if (x.state === 'running') x.state = 'error' })
            break
        case 'text': reply.text = (reply.text ? reply.text + ' ' : '') + data.text; break
        case 'markdown': reply.md = (reply.md ? reply.md + '\n\n' : '') + data.md; break
        case 'text_chunk': reply.md = (reply.md || '') + data.delta; break
        case 'sources': reply.sources = data.sources; break
    }
}
