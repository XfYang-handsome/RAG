<script setup>
import { computed } from 'vue'
import { useChatStore } from '../../stores/chat'

const chat = useChatStore()

// 收集用户消息（保留在 messages 中的 index，用于 DOM 定位）
const userMsgs = computed(() =>
  chat.messages
    .map((m, i) => ({ role: m.role, index: i, content: m.content }))
    .filter((x) => x.role === 'user'),
)

function jump(index) {
  const el = document.querySelector(`[data-msg-index="${index}"]`)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}
</script>

<template>
  <div class="question-rail">
    <div class="rail-title" :title="userMsgs.length ? `问题追溯（共 ${userMsgs.length} 条）` : '问题追溯'">⋯</div>
    <div class="rail-list">
      <div v-if="!userMsgs.length" class="rail-empty">暂无</div>
      <div
        v-for="m in userMsgs"
        :key="m.index"
        class="rail-item"
        :title="m.content"
        @click="jump(m.index)"
      ></div>
    </div>
  </div>
</template>

<style scoped>
.question-rail {
  width: 30px;
  flex-shrink: 0;
  border-left: 1px solid var(--border);
  background: var(--panel-sub-bg);
  display: flex;
  flex-direction: column;
  align-items: center;
}
.rail-title {
  width: 100%;
  height: 40px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid var(--border);
  color: var(--text-faint);
  font-size: 15px;
  line-height: 1;
  cursor: default;
}
.rail-title:hover {
  color: var(--text-secondary);
}
.rail-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 9px;
}
.rail-empty {
  padding: 20px 4px;
  text-align: center;
  font-size: 10px;
  color: var(--text-faint);
  writing-mode: vertical-rl;
  letter-spacing: 2px;
}
.rail-item {
  position: relative;
  width: 18px;
  height: 4px;
  border-radius: 2px;
  background: var(--border-strong);
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.15s;
}
.rail-item:hover {
  background: var(--primary);
  width: 24px;
  height: 5px;
}
</style>
