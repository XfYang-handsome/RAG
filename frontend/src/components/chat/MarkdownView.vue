<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { renderMarkdown, renderStreamingText } from '../../utils/markdown'

const props = defineProps({
  content: { type: String, default: '' },
  streaming: { type: Boolean, default: false },
})

const rootEl = ref(null)

// 流式期间用轻量纯文本渲染（性能关键），完成后用完整 markdown + KaTeX。
const html = computed(() =>
  props.streaming ? renderStreamingText(props.content) : renderMarkdown(props.content),
)

// 给代码块包裹容器并注入「复制」按钮（v-html 内容不受 scoped 影响，需原生 DOM 操作）
function enhanceCodeBlocks() {
  const root = rootEl.value
  if (!root) return
  root.querySelectorAll('pre').forEach((pre) => {
    if (pre.closest('.code-block')) return
    const wrapper = document.createElement('div')
    wrapper.className = 'code-block'
    pre.parentNode.insertBefore(wrapper, pre)
    wrapper.appendChild(pre)

    const lang = detectLang(pre)
    if (lang) {
      const tag = document.createElement('span')
      tag.className = 'code-lang'
      tag.textContent = lang
      wrapper.appendChild(tag)
    }

    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'code-copy-btn'
    btn.textContent = '复制'
    btn.addEventListener('click', async () => {
      const text = pre.innerText || ''
      try {
        await navigator.clipboard.writeText(text)
        btn.textContent = '已复制'
        btn.classList.add('done')
        setTimeout(() => {
          btn.textContent = '复制'
          btn.classList.remove('done')
        }, 1500)
      } catch (e) {
        btn.textContent = '复制失败'
        setTimeout(() => (btn.textContent = '复制'), 1500)
      }
    })
    wrapper.appendChild(btn)
  })
}

// 从 ```language 信息行推断语言标签
function detectLang(pre) {
  const code = pre.querySelector('code')
  const cls = code ? (code.className || '') : ''
  const m = cls.match(/language-([\w-]+)/)
  return m ? m[1] : ''
}

// 流式结束后（pending false）切换到完整 markdown，此时才有 pre 块
watch(
  () => props.streaming,
  (s) => {
    if (!s) nextTick(enhanceCodeBlocks)
  },
)

onMounted(() => nextTick(enhanceCodeBlocks))
</script>

<template>
  <div ref="rootEl" class="md" v-html="html"></div>
</template>
