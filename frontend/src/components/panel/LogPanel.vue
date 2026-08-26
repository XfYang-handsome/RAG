<script setup>
import { ref, computed } from 'vue'
import { ScrollText, RefreshCw, Trash2 } from '@lucide/vue'
import { useSettingsStore } from '../../stores/settings'
import CollapsibleCard from './CollapsibleCard.vue'

const settings = useSettingsStore()
const following = ref(true)
const boxEl = ref(null)

const logs = computed(() => settings.logs)

const LEVEL_COLOR = {
  INFO: 'var(--success)',
  WARN: 'var(--warn-text)',
  ERROR: 'var(--danger)',
  DEBUG: 'var(--text-faint)',
}

function fmtTs(ts) {
  if (!ts) return ''
  // 后端已格式化为 "HH:MM:SS" 字符串，直接展示；兼容时间戳
  if (typeof ts === 'string' && ts.includes(':')) return ts
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function scrollToEnd() {
  requestAnimationFrame(() => {
    if (boxEl.value && following.value) {
      boxEl.value.scrollTop = boxEl.value.scrollHeight
    }
  })
}

function refresh() {
  settings.refreshLogs().then(scrollToEnd)
}

function onScroll() {
  const el = boxEl.value
  if (!el) return
  following.value = el.scrollHeight - el.scrollTop - el.clientHeight < 30
}

function clear() {
  settings.logs = []
}
</script>

<template>
  <CollapsibleCard title="服务日志" persist-key="log">
    <template #icon><ScrollText :size="15" /></template>
    <template #actions>
      <span class="follow" :class="{ on: following }" @click="following = !following" title="点击切换自动滚动">
        {{ following ? '跟随' : '不跟随' }}
      </span>
      <button class="icon-btn" title="清空显示" @click="clear"><Trash2 :size="14" /></button>
      <button class="icon-btn" title="刷新" @click="refresh"><RefreshCw :size="14" /></button>
    </template>
    <div ref="boxEl" class="log-box" @scroll="onScroll">
      <div v-if="!logs.length" class="log-empty">暂无日志</div>
      <div v-for="(l, i) in logs" :key="i" class="log-line">
        <span class="log-ts">{{ fmtTs(l.ts) }}</span>
        <span class="log-level" :style="{ color: LEVEL_COLOR[l.level] || 'var(--text-faint)' }">{{ l.level }}</span>
        <span class="log-msg">{{ l.msg }}</span>
      </div>
    </div>
  </CollapsibleCard>
</template>

<style scoped>
.head-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.follow {
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--text-faint);
  background: var(--panel-sub-bg);
  user-select: none;
}
.follow.on {
  color: var(--primary);
}
.icon-btn {
  background: none;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  padding: 4px 6px;
  cursor: pointer;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  transition: all 0.15s;
}
.icon-btn:hover { color: var(--primary); border-color: var(--primary); }
.log-box {
  height: 140px;
  overflow-y: auto;
  padding: 10px 14px;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 11px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-all;
  background: var(--panel-bg);
  color: var(--text-secondary);
}
.log-empty {
  color: var(--text-faint);
  text-align: center;
  padding: 20px;
}
.log-line {
  display: flex;
  gap: 8px;
}
.log-ts {
  flex-shrink: 0;
  color: var(--text-faint);
}
.log-level {
  flex-shrink: 0;
  font-weight: 700;
  min-width: 44px;
}
.log-msg {
  flex: 1;
}
</style>
