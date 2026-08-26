<script setup>
import { ref, watch } from 'vue'
import MarkdownView from './MarkdownView.vue'
import ReasoningBox from './ReasoningBox.vue'
import AgentWb from './AgentWb.vue'
import Citations from './Citations.vue'

const props = defineProps({
  msg: { type: Object, required: true },
  index: { type: Number, default: 0 },
})

// 思考块：正文开始（第一个 token）后折叠
const reasoningCollapsed = ref(false)
watch(
  () => props.msg.content,
  (v) => {
    if (v && props.msg.reasoning) reasoningCollapsed.value = true
  },
)
</script>

<template>
  <!-- 系统消息 -->
  <div v-if="msg.role === 'system'" class="msg msg-system" :data-msg-index="index">{{ msg.content }}</div>

  <!-- 用户消息 -->
  <div v-else-if="msg.role === 'user'" class="msg msg-user" :data-msg-index="index">{{ msg.content }}</div>

  <!-- AI 消息 -->
  <div v-else class="msg msg-ai" :data-msg-index="index">
    <AgentWb v-if="msg.agentTrace && msg.agentTrace.length" :trace="msg.agentTrace" />
    <ReasoningBox
      v-if="msg.reasoning"
      :reasoning="msg.reasoning"
      :collapsed="reasoningCollapsed"
      :thinking="msg.pending && !msg.content"
      @toggle="reasoningCollapsed = !reasoningCollapsed"
    />
    <div v-if="msg.status" class="ai-status">{{ msg.status }}</div>
    <div v-if="msg.warning" class="ai-warning">⚠ {{ msg.warning }}</div>
    <MarkdownView :content="msg.content" :streaming="msg.pending" />
    <Citations v-if="msg.citations && msg.citations.length" :citations="msg.citations" />
    <div v-if="msg.errorText" class="ai-error">[错误: {{ msg.errorText }}]</div>
    <div v-if="msg.pending && !msg.content && !msg.reasoning" class="ai-thinking">正在思考…</div>
  </div>
</template>

<style scoped>
.ai-status {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.ai-warning {
  font-size: 12px;
  color: var(--warn-text);
  background: var(--warn-bg);
  border-radius: 6px;
  padding: 4px 10px;
  margin-bottom: 6px;
}
.ai-error {
  font-size: 12px;
  color: var(--danger);
  margin-top: 6px;
}
.ai-thinking {
  font-size: 13px;
  color: var(--text-faint);
  font-style: italic;
}
</style>
