<script setup>
import { ref } from 'vue'
import { ChevronDown } from '@lucide/vue'

const props = defineProps({
  title: { type: String, required: true },
  defaultOpen: { type: Boolean, default: true },
  persistKey: { type: String, default: '' },
})

const STORAGE_PREFIX = 'rag-panel-'

function readInitial() {
  if (!props.persistKey) return props.defaultOpen
  const v = localStorage.getItem(STORAGE_PREFIX + props.persistKey)
  return v === null ? props.defaultOpen : v === '1'
}

const open = ref(readInitial())

function toggle() {
  open.value = !open.value
  if (props.persistKey) {
    localStorage.setItem(STORAGE_PREFIX + props.persistKey, open.value ? '1' : '0')
  }
}
</script>

<template>
  <section class="cc-card" :class="{ collapsed: !open }">
    <header class="cc-head" @click="toggle">
      <span class="cc-icon"><slot name="icon" /></span>
      <span class="cc-title">{{ title }}</span>
      <div class="cc-actions" @click.stop>
        <slot name="actions" />
      </div>
      <button type="button" class="cc-chevron" :class="{ open }" aria-label="折叠/展开">
        <ChevronDown :size="16" />
      </button>
    </header>

    <!-- grid-rows 高度过渡：折叠时平滑收起，无需 JS 计算高度 -->
    <div class="cc-body" :class="{ collapsed: !open }">
      <div class="cc-body-inner">
        <slot />
      </div>
    </div>
  </section>
</template>

<style scoped>
.cc-card {
  margin: 0 16px 16px;
  background: var(--panel-sub-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.cc-card:hover {
  box-shadow: var(--shadow-sm);
}
.cc-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border);
  transition: background 0.15s;
}
.cc-head:hover {
  background: rgba(128, 128, 128, 0.06);
}
.cc-icon {
  display: flex;
  align-items: center;
  color: var(--primary);
  flex-shrink: 0;
}
.cc-title {
  flex: 1;
  font-weight: 600;
  font-size: 14px;
  color: var(--text);
}
.cc-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}
.cc-chevron {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  color: var(--text-faint);
  cursor: pointer;
  padding: 3px;
  border-radius: 6px;
  transition: transform 0.25s ease, color 0.15s;
  flex-shrink: 0;
}
.cc-chevron:hover {
  color: var(--primary);
}
.cc-chevron.open {
  transform: rotate(180deg);
}
.cc-body {
  display: grid;
  grid-template-rows: 1fr;
  transition: grid-template-rows 0.28s cubic-bezier(0.4, 0, 0.2, 1);
}
.cc-body.collapsed {
  grid-template-rows: 0fr;
}
.cc-body-inner {
  overflow: hidden;
  min-height: 0;
}
.cc-card.collapsed .cc-head {
  border-bottom: none;
}
</style>
