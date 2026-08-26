<script setup>
import { ChevronDown } from '@lucide/vue'

defineProps({
  reasoning: { type: String, default: '' },
  collapsed: { type: Boolean, default: false },
  thinking: { type: Boolean, default: false },
})

const emit = defineEmits(['toggle'])
</script>

<template>
  <div class="reasoning-box" :class="{ collapsed }">
    <div class="reasoning-header" @click="emit('toggle')">
      <ChevronDown :size="12" class="arrow" />
      <span class="title">
        <span v-if="thinking" class="dot"></span>
        <span class="title-text">{{ thinking ? '正在思考...' : collapsed ? '已思考（点击展开）' : '思考过程' }}</span>
      </span>
    </div>
    <div class="reasoning-body">{{ reasoning }}</div>
  </div>
</template>

<style scoped>
.reasoning-box {
  margin-bottom: 8px;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  overflow: hidden;
  background: var(--panel-sub-bg);
}
.reasoning-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
}
.arrow {
  transition: transform 0.2s;
}
.reasoning-box.collapsed .arrow {
  transform: rotate(-90deg);
}
.title {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
}
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--primary);
  animation: pulse 1.2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.reasoning-body {
  padding: 8px 12px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-muted);
  white-space: pre-wrap;
  word-break: break-word;
  border-top: 1px dashed var(--border-light);
  max-height: 200px;
  overflow-y: auto;
}
.reasoning-box.collapsed .reasoning-body {
  display: none;
}
</style>
