// Markdown 渲染：markdown-it + KaTeX
// 支持四种 LaTeX 分隔符（先用占位符保护，渲染后再还原为 KaTeX HTML）：
//   块级：\[...\] 与 $$...$$
//   行内：\(...\) 与 $...$
import MarkdownIt from 'markdown-it'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const md = new MarkdownIt({
  html: false, // 防 XSS：不渲染原始 HTML
  linkify: true,
  breaks: true, // 单换行转 <br>
})

// 占位符：用 \u0001（SOH 控制字符），markdown-it 会完整保留（\u0000 会被替换成 �）
const PH = '\u0001'

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

// 流式期间轻量渲染：仅转义 + 换行，不跑完整 markdown（长文本性能关键）
export function renderStreamingText(text) {
  if (!text) return ''
  return escapeHtml(text).replace(/\n/g, '<br>')
}

// 完整 markdown 渲染（含 LaTeX / 代码块 / 表格）
export function renderMarkdown(src) {
  if (!src) return ''

  const latex = []
  let html = String(src)

  const push = (tex, display) => {
    latex.push({ tex, display })
    return `${PH}LTX${latex.length - 1}${PH}`
  }

  // 0. 保护 LaTeX（顺序：先块级，再行内；$$ 必须先于 $ 处理，避免被拆成两个 $）
  // 块级 \[...\]
  html = html.replace(/\\\[([\s\S]*?)\\\]/g, (m, tex) => push(tex, true))
  // 块级 $$...$$
  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (m, tex) => push(tex, true))
  // 行内 \(...\)
  html = html.replace(/\\\(([\s\S]*?)\\\)/g, (m, tex) => push(tex, false))
  // 行内 $...$（不跨行；前面不能是 $、后面不能是 $，排除 $$ 残留）
  html = html.replace(/(^|[^$])\$([^$\n]+?)\$(?!\$)/g, (m, pre, tex) => pre + push(tex, false))

  // 1. markdown-it 渲染（html:false 会自动转义 HTML，防 XSS）
  html = md.render(html)

  // 2. 还原 LaTeX 占位符
  html = html.replace(new RegExp(`${PH}LTX(\\d+)${PH}`, 'g'), (m, idx) => {
    const item = latex[+idx]
    if (!item) return ''
    try {
      return katex.renderToString(item.tex, {
        displayMode: item.display,
        throwOnError: false,
      })
    } catch (e) {
      // 渲染失败降级为原文
      return `<code>${escapeHtml(item.tex)}</code>`
    }
  })

  return html
}
