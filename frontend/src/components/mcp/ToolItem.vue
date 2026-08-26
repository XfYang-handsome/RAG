<script setup>
import { reactive, ref } from 'vue'
import { Bug } from '@lucide/vue'
import { useMcpStore } from '../../stores/mcp'

const props = defineProps({
  serverName: { type: String, required: true },
  tool: { type: Object, required: true },
})

const mcp = useMcpStore()

const expanded = ref(false)
const args = reactive({})
const result = ref('')
const isErr = ref(false)
const running = ref(false)

function paramsFromSchema(schema) {
  const propsMap = (schema && schema.properties) || {}
  const req = new Set((schema && schema.required) || [])
  return Object.keys(propsMap).map((n) => ({
    name: n,
    type: (propsMap[n] || {}).type || 'string',
    required: req.has(n),
    description: (propsMap[n] || {}).description || '',
    default: (propsMap[n] || {}).default,
  }))
}

function togglePanel() {
  if (expanded.value) {
    expanded.value = false
    result.value = ''
    return
  }
  // 初始化参数默认值
  const params = paramsFromSchema(props.tool.input_schema || {})
  for (const p of params) {
    if (p.default !== null && p.default !== undefined) args[p.name] = p.default
  }
  expanded.value = true
}

function parseValue(v) {
  if (/^-?\d+$/.test(v)) return parseInt(v, 10)
  if (/^-?\d+\.\d+$/.test(v)) return parseFloat(v)
  if (v === 'true') return true
  if (v === 'false') return false
  return v
}

async function run() {
  const params = paramsFromSchema(props.tool.input_schema || {})
  const payload = {}
  for (const p of params) {
    const v = String(args[p.name] ?? '').trim()
    if (v === '') continue
    payload[p.name] = parseValue(v)
  }
  running.value = true
  result.value = '调用中...'
  isErr.value = false
  try {
    const d = await mcp.callTool(props.serverName, props.tool.name, payload)
    result.value = d.success
      ? (typeof d.result === 'string' ? d.result : JSON.stringify(d.result))
      : (d.message || '调用失败')
    if (!d.success) isErr.value = true
  } catch (e) {
    result.value = '调用失败: ' + e.message
    isErr.value = true
  } finally {
    running.value = false
  }
}

async function onToggle(e) {
  try {
    await mcp.toggleTool(props.serverName, props.tool.name, e.target.checked)
  } catch (err) {
    alert('切换失败: ' + err.message)
  }
}
</script>

<template>
  <div class="tool">
    <div class="tool-main">
      <div class="tool-name">{{ tool.name }}</div>
      <div class="tool-desc">{{ (tool.description || '').split('\n')[0] }}</div>
    </div>
    <div class="tool-side">
      <span class="tool-state" :class="tool.enabled ? 'on' : 'off'">{{ tool.enabled ? '启用' : '停用' }}</span>
      <label class="switch">
        <input type="checkbox" :checked="tool.enabled" @change="onToggle" />
        <span class="slider"></span>
      </label>
      <button class="btn sm" @click="togglePanel"><Bug :size="13" />调试</button>
    </div>

    <!-- 调试面板 -->
    <div v-if="expanded" class="test-panel">
      <div class="t">测试调用</div>
      <div
        v-for="p in paramsFromSchema(tool.input_schema || {})"
        :key="p.name"
        class="p-row"
      >
        <label>{{ p.name }}<span v-if="p.required" class="req">*</span></label>
        <input v-model="args[p.name]" type="text" :placeholder="p.description || ''" />
      </div>
      <button class="btn primary sm" :disabled="running" @click="run">
        {{ running ? '调用中...' : '调用' }}
      </button>
      <div v-if="result" class="result" :class="{ err: isErr }">{{ result }}</div>
    </div>
  </div>
</template>

<style scoped>
.tool {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 14px;
  padding: 13px 12px;
  border-bottom: 1px solid var(--border);
}
.tool:last-child {
  border-bottom: none;
}
.tool-main {
  flex: 1;
  min-width: 0;
}
.tool-name {
  font-size: 13.5px;
  font-weight: 600;
  font-family: ui-monospace, Menlo, monospace;
  color: var(--text);
}
.tool-desc {
  font-size: 12px;
  color: var(--text-2, #555);
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.tool-side {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.tool-state {
  font-size: 12px;
  color: var(--text-faint);
  min-width: 30px;
  text-align: center;
}
.tool-state.on {
  color: var(--success);
}
.switch {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
  flex-shrink: 0;
}
.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}
.slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background: var(--border-strong);
  border-radius: 22px;
  transition: 0.25s;
}
.slider:before {
  content: '';
  position: absolute;
  height: 16px;
  width: 16px;
  left: 3px;
  top: 3px;
  background: #fff;
  border-radius: 50%;
  transition: 0.25s;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.switch input:checked + .slider {
  background: var(--primary);
}
.switch input:checked + .slider:before {
  transform: translateX(18px);
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--border-strong);
  background: var(--card-2, var(--panel-sub-bg));
  color: var(--text-2, #555);
  border-radius: 8px;
  padding: 6px 11px;
  cursor: pointer;
  font-size: 12.5px;
  transition: all 0.2s;
}
.btn:hover {
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
.btn.primary:hover {
  background: var(--grad-primary-strong);
  color: #fff;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.test-panel {
  width: 100%;
  padding: 14px;
  background: var(--card-2, var(--panel-sub-bg));
  border: 1px solid var(--border);
  border-radius: 10px;
}
.t {
  font-size: 12px;
  color: var(--text-faint);
  margin-bottom: 10px;
  font-weight: 600;
}
.p-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 9px;
}
.p-row label {
  font-size: 12px;
  color: var(--text-2, #555);
  min-width: 110px;
  flex-shrink: 0;
  font-family: ui-monospace, Menlo, monospace;
}
.p-row input {
  flex: 1;
  min-width: 140px;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  background: var(--panel-bg);
  color: var(--text);
  outline: none;
}
.req {
  color: var(--danger);
}
.result {
  margin-top: 10px;
  background: var(--log-bg, var(--panel-bg));
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--text-2, #555);
  max-height: 240px;
  overflow: auto;
  font-family: ui-monospace, Menlo, monospace;
}
.result.err {
  color: var(--danger);
  border-color: rgba(220, 38, 38, 0.4);
}
</style>
