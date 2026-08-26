<script setup>
import { computed } from 'vue'
import { Play, Square, ScrollText, Trash2, RefreshCw } from '@lucide/vue'
import { useMcpStore } from '../../stores/mcp'
import ToolItem from './ToolItem.vue'

const props = defineProps({
  server: { type: Object, required: true },
})

const mcp = useMcpStore()

const status = computed(() => (props.server._status || {}).status || 'stopped')
const running = computed(() => status.value === 'running')
const pid = computed(() => (props.server._status || {}).pid)

const tools = computed(() => mcp.toolsCache[props.server.name] || [])

const statusText = computed(() => {
  if (running.value) return '运行中' + (pid.value ? ` (pid=${pid.value})` : '')
  if (status.value === 'error') return '出错'
  return '已停止'
})
</script>

<template>
  <div class="server">
    <div class="s-head">
      <span class="dot" :class="status"></span>
      <span class="s-name">{{ server.name }}</span>
      <span class="s-status">{{ statusText }}</span>
      <span class="s-url">{{ server.url || '' }}</span>
      <span class="s-actions">
        <button class="btn primary sm" :disabled="running" @click="mcp.startServer(server.name)">
          <Play :size="13" />启动
        </button>
        <button class="btn sm" :disabled="!running" @click="mcp.stopServer(server.name)">
          <Square :size="13" />停止
        </button>
        <button class="btn sm" @click="mcp.showLogs(server.name)"><ScrollText :size="13" />日志</button>
        <button class="btn danger sm" @click="mcp.deleteServer(server.name)"><Trash2 :size="13" />删除</button>
      </span>
    </div>

    <div class="tools">
      <div class="tools-head">
        <span class="lbl">工具</span>
        <span class="count">{{ tools.length }} 个</span>
        <button class="btn sm" style="margin-left: auto" @click="mcp.loadServerTools(server.name, true)">
          <RefreshCw :size="13" />刷新
        </button>
      </div>
      <div class="tools-body">
        <div v-if="!tools.length" class="empty">加载工具...</div>
        <ToolItem
          v-for="t in tools"
          :key="t.name"
          :server-name="server.name"
          :tool="t"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.server {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.server:hover {
  border-color: var(--border-strong);
  box-shadow: var(--card-hover);
  transform: translateY(-2px);
}
.s-head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.s-name {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}
.s-url {
  font-size: 12px;
  color: var(--text-faint);
  flex: 1;
  min-width: 130px;
  word-break: break-all;
  font-family: ui-monospace, monospace;
}
.s-status {
  font-size: 12px;
  color: var(--text-2, #555);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot.running {
  background: var(--success);
  box-shadow: 0 0 0 3px rgba(47, 214, 113, 0.18);
  animation: dotPulse 2s ease-in-out infinite;
}
@keyframes dotPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(47, 214, 113, 0.5); }
  50% { box-shadow: 0 0 0 5px rgba(47, 214, 113, 0); }
}
.dot.stopped {
  background: var(--text-faint);
}
.dot.error {
  background: var(--danger);
  box-shadow: 0 0 0 3px rgba(240, 82, 79, 0.18);
}
.s-actions {
  display: flex;
  gap: 8px;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--border-strong);
  background: var(--panel-sub-bg);
  color: var(--text-2, #555);
  border-radius: 8px;
  padding: 6px 11px;
  cursor: pointer;
  font-size: 12.5px;
  transition: all 0.2s;
}
.btn:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--primary);
}
.btn.sm {
  padding: 5px 10px;
  font-size: 12px;
}
.btn.primary {
  background: var(--grad-primary);
  border-color: transparent;
  color: #fff;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.28);
}
.btn.primary:hover:not(:disabled) {
  background: var(--grad-primary-strong);
  color: #fff;
}
.btn.danger:hover:not(:disabled) {
  color: var(--danger);
  border-color: var(--danger);
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.tools {
  margin-top: 16px;
  border-top: 1px solid var(--border);
  padding-top: 4px;
}
.tools-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0 4px;
}
.tools-head .lbl {
  font-size: 12px;
  color: var(--text-faint);
  font-weight: 600;
  letter-spacing: 0.4px;
}
.count {
  font-size: 12px;
  color: var(--text-faint);
  background: var(--panel-sub-bg);
  border-radius: 10px;
  padding: 1px 9px;
}
.empty {
  padding: 14px;
  text-align: center;
  font-size: 12px;
  color: var(--text-faint);
}
</style>
