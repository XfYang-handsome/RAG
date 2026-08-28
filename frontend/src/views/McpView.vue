<script setup>
import { ref, reactive, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowLeft, Plus, RefreshCw } from '@lucide/vue'
import PrismLogo from '../components/PrismLogo.vue'
import { useMcpStore } from '../stores/mcp'
import ServerCard from '../components/mcp/ServerCard.vue'

const router = useRouter()
const mcp = useMcpStore()

// 工具决策模型表单
const toolLlmFormVisible = ref(false)
const toolLlmForm = reactive({ name: '', model: '', base_url: '', api_key: '', protocol: 'openai' })

// 服务器表单
const serverFormVisible = ref(false)
const serverForm = reactive({ name: '', command: '', args: '', url: '', transport: 'streamable-http' })

const logBox = ref(null)

async function submitToolLlm() {
  if (!toolLlmForm.name || !toolLlmForm.model) return alert('请填写名称与模型名称')
  try {
    await mcp.addToolLlm({ ...toolLlmForm, name: toolLlmForm.name.trim(), type: 'online' })
    toolLlmFormVisible.value = false
  } catch (e) {
    alert('新增失败: ' + e.message)
  }
}

async function onSelectToolLlm(e) {
  try {
    await mcp.selectToolLlm(e.target.value)
  } catch (err) {
    alert('切换失败: ' + err.message)
  }
}

async function submitServer() {
  if (!serverForm.name.trim()) return alert('请填写服务器名称')
  try {
    await mcp.addServer({
      name: serverForm.name.trim(),
      command: serverForm.command.trim(),
      args: serverForm.args.trim(),
      url: serverForm.url.trim(),
      transport: serverForm.transport,
      auto_start: false,
    })
    serverFormVisible.value = false
  } catch (e) {
    alert('新增失败: ' + e.message)
  }
}

// 日志刷新：userInitiated=true 滚到底；否则仅当用户已在底部才跟随
async function refreshLogs(userInitiated = false) {
  const box = logBox.value
  const nearBottom = box ? box.scrollTop + box.clientHeight >= box.scrollHeight - 24 : true
  const prevTop = box ? box.scrollTop : 0
  await mcp.refreshSelectedLogs(userInitiated)
  await nextTick()
  if (box) {
    if (userInitiated || nearBottom) box.scrollTop = box.scrollHeight
    else box.scrollTop = prevTop
  }
}

let serversTimer = null
let logsTimer = null

// 切换服务器（点击「日志」）后滚到底
watch(
  () => mcp.selectedServer,
  async () => {
    await nextTick()
    if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
  },
)

onMounted(() => {
  mcp.loadToolLlms()
  mcp.loadServers(true)
  serversTimer = setInterval(() => mcp.loadServers(), 8000)
  logsTimer = setInterval(() => refreshLogs(false), 4000)
})

onUnmounted(() => {
  if (serversTimer) clearInterval(serversTimer)
  if (logsTimer) clearInterval(logsTimer)
})
</script>

<template>
  <div class="mcp-shell">
    <!-- 顶栏 -->
    <header class="app-header">
      <button class="header-btn" @click="router.push('/')"><ArrowLeft :size="16" />返回</button>
      <span class="brand"><span class="logo"><PrismLogo :size="16" /></span>MCP 管理</span>
      <span class="sub">工具全部注册在 MCP 服务器上 · 每个工具可独立启停</span>
    </header>

    <div class="wrap">
      <!-- 工具决策模型工具条 -->
      <div class="toolbar">
        <span class="lbl">工具决策模型</span>
        <select :value="mcp.currentToolLlm" @change="onSelectToolLlm">
          <option v-if="!mcp.toolLlms.length" value="">（未配置）</option>
          <option v-for="m in mcp.toolLlms" :key="m.name" :value="m.name">{{ m.name }}</option>
        </select>
        <button class="btn" @click="toolLlmFormVisible = !toolLlmFormVisible"><Plus :size="13" />新增</button>
        <button class="btn" @click="mcp.loadToolLlms()"><RefreshCw :size="13" />刷新</button>
      </div>

      <!-- 工具决策模型新增表单 -->
      <div v-if="toolLlmFormVisible" class="form">
        <div class="sec-head"><h2>新增工具决策模型</h2></div>
        <div class="f-row"><label>名称 *</label><input v-model="toolLlmForm.name" placeholder="deepseek-tool" /></div>
        <div class="f-row"><label>模型名称 *</label><input v-model="toolLlmForm.model" placeholder="deepseek-chat" /></div>
        <div class="f-row"><label>API 地址 *</label><input v-model="toolLlmForm.base_url" placeholder="https://api.deepseek.com/v1" /></div>
        <div class="f-row"><label>API Key *</label><input v-model="toolLlmForm.api_key" type="password" placeholder="sk-..." /></div>
        <div class="f-row">
          <label>协议</label>
          <select v-model="toolLlmForm.protocol">
            <option value="openai">OpenAI 兼容</option>
            <option value="doubao">豆包 Responses API</option>
          </select>
        </div>
        <div class="f-actions">
          <button class="btn" @click="toolLlmFormVisible = false">取消</button>
          <button class="btn primary" @click="submitToolLlm">保存</button>
        </div>
      </div>

      <!-- 服务器列表 -->
      <div class="sec-head">
        <h2>服务器</h2>
        <button class="btn primary" @click="serverFormVisible = !serverFormVisible"><Plus :size="13" />新增服务器</button>
      </div>

      <!-- 服务器新增表单 -->
      <div v-if="serverFormVisible" class="form">
        <div class="sec-head"><h2>新增 MCP 服务器</h2></div>
        <div class="f-row"><label>名称 *</label><input v-model="serverForm.name" placeholder="PrismRAG-Service" /></div>
        <div class="f-row"><label>启动命令</label><input v-model="serverForm.command" placeholder="-m（留空表示仅连接外部 URL）" /></div>
        <div class="f-row"><label>启动参数</label><input v-model="serverForm.args" placeholder="mcp_service --host 127.0.0.1 --port 8765" /></div>
        <div class="f-row"><label>连接 URL</label><input v-model="serverForm.url" placeholder="http://127.0.0.1:8765/mcp" /></div>
        <div class="f-row">
          <label>传输协议</label>
          <select v-model="serverForm.transport">
            <option value="streamable-http">streamable-http</option>
            <option value="http">http</option>
            <option value="sse">sse</option>
          </select>
        </div>
        <div class="f-actions">
          <button class="btn" @click="serverFormVisible = false">取消</button>
          <button class="btn primary" @click="submitServer">保存</button>
        </div>
      </div>

      <!-- 服务器卡片列表 -->
      <div v-if="mcp.loading && !mcp.servers.length" class="loading">加载中...</div>
      <div v-else-if="!mcp.servers.length" class="empty">暂无服务器，点击右上角「新增服务器」添加</div>
      <ServerCard v-for="s in mcp.servers" :key="s.name" :server="s" />

      <!-- 日志 -->
      <div class="sec-head"><h2>服务器日志</h2></div>
      <div class="log-panel">
        <div class="log-head">
          <span class="lbl">{{ mcp.selectedServer ? '服务器：' + mcp.selectedServer : '请点击服务器上的「日志」查看' }}</span>
          <button class="btn sm" @click="refreshLogs(true)"><RefreshCw :size="13" />刷新</button>
        </div>
        <div ref="logBox" class="log-box">
          <span v-if="!mcp.selectedServer" style="color: var(--text-faint)">未选择服务器</span>
          <span v-else-if="!mcp.selectedLogs.length" style="color: var(--text-faint)">暂无日志</span>
          <div v-for="(l, i) in mcp.selectedLogs" :key="i" class="log-line">
            <span class="ts">[{{ l.ts }}]</span>
            <span :class="'lv-' + (l.level || 'INFO')">[{{ l.level || 'INFO' }}]</span>
            {{ l.msg }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mcp-shell {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
  min-height: 0;
  overflow: hidden;
  background: var(--bg);
}
.app-header {
  background: var(--header-grad);
  color: var(--header-text);
  height: 60px;
  padding: 0 28px;
  display: flex;
  align-items: center;
  gap: 14px;
  flex-shrink: 0;
  box-shadow: 0 3px 18px rgba(0, 0, 0, 0.14);
}
.header-btn {
  background: var(--header-btn);
  color: var(--header-text);
  border: none;
  border-radius: 8px;
  padding: 7px 14px;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: background 0.2s ease;
}
.header-btn:hover {
  background: var(--header-btn-hover);
}
.brand {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.3px;
  display: flex;
  align-items: center;
  gap: 9px;
  color: #fff;
}
.logo {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--grad-primary);
  display: flex;
  align-items: center;
  justify-content: center;
}
.sub {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.55);
  margin-left: auto;
}
.wrap {
  max-width: 900px;
  margin: 0 auto;
  padding: 26px 24px 60px;
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  width: 100%;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}
.lbl {
  font-size: 13px;
  color: var(--text-2, #555);
  white-space: nowrap;
}
select,
input {
  background: var(--panel-sub-bg);
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 9px 12px;
  color: var(--text);
  font-size: 13px;
  outline: none;
  transition: all 0.2s;
  font-family: inherit;
}
select:focus,
input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}
select {
  min-width: 200px;
  cursor: pointer;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--border-strong);
  background: var(--panel-sub-bg);
  color: var(--text-2, #555);
  border-radius: 8px;
  padding: 7px 13px;
  cursor: pointer;
  font-size: 12.5px;
  transition: all 0.2s;
}
.btn:hover {
  color: var(--text);
  border-color: var(--primary);
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
.btn.sm {
  padding: 5px 10px;
  font-size: 12px;
}
.sec-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 26px 0 12px;
}
.sec-head h2 {
  font-size: 15px;
  font-weight: 600;
  margin: 0;
  color: var(--text);
}
.form {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
  background: var(--panel-sub-bg);
}
.f-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 11px;
  flex-wrap: wrap;
}
.f-row label {
  font-size: 13px;
  color: var(--text-2, #555);
  min-width: 100px;
  flex-shrink: 0;
}
.f-row input,
.f-row select {
  flex: 1;
  min-width: 180px;
}
.f-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 4px;
}
.loading {
  color: var(--text-faint);
  text-align: center;
  padding: 36px;
  font-size: 13px;
}
.empty {
  color: var(--text-faint);
  text-align: center;
  padding: 20px;
  font-size: 13px;
}
.log-panel {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}
.log-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.log-head .lbl {
  font-size: 13px;
  color: var(--text-2, #555);
}
.log-box {
  background: var(--log-bg, var(--panel-bg));
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  font-family: ui-monospace, Menlo, monospace;
  font-size: 12px;
  max-height: 280px;
  overflow: auto;
  white-space: pre-wrap;
  color: var(--text-2, #555);
  line-height: 1.7;
}
.log-line .ts {
  color: var(--text-faint);
}
.lv-INFO {
  color: var(--success);
}
.lv-WARN {
  color: var(--warn-text);
}
.lv-ERROR {
  color: var(--danger);
}
</style>
