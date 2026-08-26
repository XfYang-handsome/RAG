<script setup>
import { ref } from 'vue'
import { useChatStore } from '../../stores/chat'

const chat = useChatStore()
const input = ref('')

const modes = [
  { value: 'rag', label: '知识库问答', title: '' },
  { value: 'agentic', label: 'Agentic 检索', title: '多步检索：需求拆解 → 循环检索 → 证据评估 → 合成' },
  { value: 'direct', label: '直接对话', title: '' },
]

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function send() {
  const q = input.value.trim()
  if (!q || chat.isStreaming) return
  input.value = ''
  chat.send(q)
}
</script>

<template>
  <div class="chat-input-wrap">
    <div class="chat-mode-row">
      <span class="chat-mode-label">对话模式</span>
      <div class="chat-mode-group">
        <button
          v-for="m in modes"
          :key="m.value"
          class="chat-mode-btn"
          :class="{ active: chat.chatMode === m.value }"
          :title="m.title"
          @click="chat.setChatMode(m.value)"
        >{{ m.label }}</button>
      </div>
    </div>
    <div class="chat-input-area">
      <textarea
        v-model="input"
        placeholder="输入问题..."
        @keydown="onKeydown"
      ></textarea>
      <button :disabled="chat.isStreaming" @click="send">发送</button>
    </div>
  </div>
</template>

<style scoped>
.chat-input-wrap {
  flex-shrink: 0;
}
.chat-mode-row {
  padding: 10px 20px 0;
  display: flex;
  align-items: center;
  gap: 10px;
}
.chat-mode-label {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}
.chat-mode-group {
  display: flex;
  gap: 6px;
}
.chat-mode-btn {
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 4px 12px;
  font-size: 12px;
  cursor: pointer;
  background: var(--panel-bg);
  color: var(--text-secondary);
  transition: all 0.18s;
}
.chat-mode-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}
.chat-mode-btn.active {
  background: var(--grad-primary);
  color: #fff;
  border-color: transparent;
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.32);
}
.chat-input-area {
  padding: 12px 20px;
  border-top: 1px solid var(--border);
  display: flex;
  gap: 8px;
}
.chat-input-area textarea {
  flex: 1;
  border: 1px solid var(--border-strong);
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 14px;
  resize: none;
  outline: none;
  font-family: inherit;
  height: 60px;
  background: var(--panel-bg);
  color: var(--text);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.chat-input-area textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14);
}
.chat-input-area button {
  background: var(--grad-primary);
  color: #fff;
  border: none;
  border-radius: 10px;
  padding: 10px 20px;
  font-size: 14px;
  cursor: pointer;
  white-space: nowrap;
  font-weight: 500;
  box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
  transition: transform 0.15s, box-shadow 0.2s;
}
.chat-input-area button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
}
.chat-input-area button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
