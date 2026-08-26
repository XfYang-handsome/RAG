<script setup>
import { ref, watch } from 'vue'
import { Bot, Copy, Check } from '@lucide/vue'
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

// 复制消息正文
const copied = ref(false)
async function copyMessage() {
  const text = props.msg.content || ''
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    setTimeout(() => (copied.value = false), 1600)
  } catch (e) {
    /* 忽略剪贴板权限错误 */
  }
}
</script>

<template>
  <!-- 系统消息 -->
  <div v-if="msg.role === 'system'" class="msg msg-system" :data-msg-index="index">{{ msg.content }}</div>

  <!-- 用户消息 -->
  <div v-else-if="msg.role === 'user'" class="msg msg-user" :data-msg-index="index">
    {{ msg.content }}
    <button class="msg-copy" :title="copied ? '已复制' : '复制'" @click="copyMessage">
      <Check v-if="copied" :size="14" />
      <Copy v-else :size="14" />
    </button>
  </div>

  <!-- AI 消息 -->
  <div v-else class="ai-row" :data-msg-index="index">
    <div class="ai-avatar" :class="{ active: msg.pending }">
      <Bot :size="18" />
    </div>
    <div class="msg msg-ai">
      <button class="msg-copy ai" :title="copied ? '已复制' : '复制'" @click="copyMessage">
        <Check v-if="copied" :size="14" />
        <Copy v-else :size="14" />
      </button>
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
      <div v-if="msg.pending && !msg.content && !msg.reasoning" class="typing">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ===== 消息气泡（迁移自 ChatView：.msg-ai 现为 .ai-row 子元素，
   父组件 scoped 无法穿透，故必须在本组件作用域内定义背景等样式） ===== */
.msg {
  max-width: 85%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
  animation: msgIn 0.3s cubic-bezier(0.21, 1.02, 0.73, 1) both;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px) scale(0.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.msg-user {
  align-self: flex-end;
  background: var(--grad-primary-strong);
  color: #fff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 3px 12px rgba(99, 102, 241, 0.35);
}
.msg-ai {
  flex: 1;
  min-width: 0;
  background: var(--msg-ai-bg);
  color: var(--text);
  border-bottom-left-radius: 4px;
}
.msg-system {
  align-self: center;
  background: var(--warn-bg);
  color: var(--warn-text);
  font-size: 12px;
  padding: 6px 12px;
  border-radius: 8px;
  max-width: 95%;
}

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

/* AI 消息行：头像 + 气泡 */
.ai-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  max-width: 88%;
  align-self: flex-start;
}
.ai-avatar {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  border-radius: 11px;
  background: var(--grad-primary);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.22);
  margin-top: 2px;
  transition: box-shadow 0.3s ease;
}
.ai-avatar.active {
  animation: avatarGlow 1.6s ease-in-out infinite;
}
@keyframes avatarGlow {
  0%, 100% { box-shadow: 0 4px 12px rgba(99, 102, 241, 0.38); }
  50% { box-shadow: 0 4px 20px rgba(99, 102, 241, 0.65); }
}

/* 复制按钮（hover 显现） */
.msg-user,
.msg-ai {
  position: relative;
}
.msg-copy {
  position: absolute;
  top: 6px;
  border: 1px solid var(--border-strong);
  background: var(--panel-bg);
  color: var(--text-muted);
  width: 26px;
  height: 26px;
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s ease, color 0.15s, border-color 0.15s;
  z-index: 2;
}
.msg-copy.ai { right: 8px; }
.msg-user .msg-copy { left: -34px; }
.msg-copy:hover {
  color: var(--primary);
  border-color: var(--primary);
}
.msg-user:hover .msg-copy,
.msg-ai:hover .msg-copy {
  opacity: 1;
}

/* 打字指示器：三点跳动 */
.typing {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 2px;
}
.typing-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--primary);
  opacity: 0.35;
  animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%, 100% { opacity: 0.25; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(-3px); }
}
</style>
