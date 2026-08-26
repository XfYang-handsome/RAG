<script setup>
import { computed } from 'vue'
import { renderMarkdown, renderStreamingText } from '../../utils/markdown'

const props = defineProps({
  content: { type: String, default: '' },
  streaming: { type: Boolean, default: false },
})

// 流式期间用轻量纯文本渲染（性能关键），完成后用完整 markdown + KaTeX。
// computed 同时依赖 streaming 和 content，任一变化都会重新计算，
// 确保 pending 从 true→false 时切到完整渲染（LaTeX 此时才会被 KaTeX 处理）。
const html = computed(() =>
  props.streaming ? renderStreamingText(props.content) : renderMarkdown(props.content),
)
</script>

<template>
  <div class="md" v-html="html"></div>
</template>
