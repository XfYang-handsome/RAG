<script setup>
import { ref, reactive, computed } from 'vue'
import { NModal, darkTheme } from 'naive-ui'
import { Settings, Plus, Pencil, Trash2, Check, Database, Search, Cpu } from '@lucide/vue'
import { useSettingsStore } from '../../stores/settings'
import { useThemeStore } from '../../stores/theme'

const settings = useSettingsStore()
const themeStore = useThemeStore()

// NModal 会被 teleport 到 body，脱离 NConfigProvider 的主题上下文，
// 必须显式传入 theme，否则暗色主题下弹窗仍是白色。
const naiveTheme = computed(() => (themeStore.isDark ? darkTheme : null))

const tab = ref('models')

const MODEL_KINDS = [
  { kind: 'llm', label: '生成模型', desc: '最终回答的 LLM' },
  { kind: 'embedding', label: '向量模型', desc: '文本向量化' },
  { kind: 'reranker', label: '重排序', desc: '检索结果重排' },
  { kind: 'tool_llm', label: '工具决策模型', desc: '决定调用哪些工具' },
  { kind: 'rewrite', label: '查询重写', desc: '改写查询（可选）' },
  { kind: 'summary', label: '摘要模型', desc: '章节摘要（可选）' },
]

const activeKind = ref('llm')

const modelsOf = computed(() => {
  const list = settings.models[activeKind.value] || []
  return list
})

// ============ 模型表单 ============
const formVisible = ref(false)
const formMode = ref('add') // add / edit
const formKind = ref('llm')
const formOldName = ref('')
const form = reactive({
  name: '',
  type: 'online',
  model: '',
  base_url: '',
  api_key: '',
  model_path: '',
  protocol: 'openai',
})

const isReranker = computed(() => formKind.value === 'reranker')
const hasProtocol = computed(() => ['llm', 'rewrite', 'tool_llm'].includes(formKind.value))
const isLocal = computed(() => form.type === 'local')

function openAdd(kind) {
  formMode.value = 'add'
  formKind.value = kind
  formOldName.value = ''
  Object.assign(form, {
    name: '', type: 'online', model: '', base_url: '', api_key: '', model_path: '', protocol: 'openai',
  })
  formVisible.value = true
}

function openEdit(kind, m) {
  formMode.value = 'edit'
  formKind.value = kind
  formOldName.value = m.name
  Object.assign(form, {
    name: m.name || '',
    type: m.type || 'online',
    model: m.model || '',
    base_url: m.base_url || '',
    api_key: m.api_key || '',
    model_path: m.model_path || '',
    protocol: m.protocol || 'openai',
  })
  formVisible.value = true
}

async function submitModel() {
  if (!form.name.trim()) return alert('请填写模型名称')
  const payload = {
    name: form.name.trim(),
    type: form.type,
    model: form.model,
    base_url: form.base_url,
    api_key: form.api_key,
    model_path: form.model_path,
    protocol: form.protocol,
  }
  try {
    if (formMode.value === 'add') {
      await settings.addModel(formKind.value, payload)
    } else {
      await settings.updateModel(formKind.value, formOldName.value, payload)
    }
    formVisible.value = false
  } catch (e) {
    alert('保存失败: ' + e.message)
  }
}

async function delModel(kind, name) {
  if (!confirm(`确定删除模型 "${name}" 吗？`)) return
  await settings.deleteModel(kind, name)
}

async function selectModel(kind, name) {
  await settings.selectModel(kind, name)
}

function isCurrent(kind, name) {
  return settings.current[kind] === name
}

// ============ 数据库表单 ============
const dbFormVisible = ref(false)
const dbForm = reactive({ name: '', type: 'local', url: '', token: '', db_name: '' })

function openAddDb() {
  Object.assign(dbForm, { name: '', type: 'local', url: '', token: '', db_name: '' })
  dbFormVisible.value = true
}

async function submitDb() {
  if (!dbForm.name.trim()) return alert('请填写数据库名称')
  try {
    await settings.addDb({ ...dbForm, name: dbForm.name.trim() })
    dbFormVisible.value = false
  } catch (e) {
    alert('保存失败: ' + e.message)
  }
}

async function delDb(name) {
  if (!confirm(`确定删除数据库 "${name}" 吗？`)) return
  await settings.deleteDb(name)
}

async function selectDb(name) {
  await settings.selectDb(name)
}

// ============ 检索 ============
const RETRIEVAL_MODES = [
  { value: 'vector', label: '向量检索' },
  { value: 'hybrid', label: '混合检索' },
  { value: 'tree', label: '树导航' },
]

function open() {
  settings.settingsOpen = true
}

const RERANKER_TEXT = {
  idle: '未加载',
  loading: '加载中...',
  loaded: '已加载',
  error: '加载失败',
  online: '在线模型',
}
const rerankerText = computed(() => RERANKER_TEXT[settings.reranker.status] || settings.reranker.status)
const rerankerOk = computed(() => settings.reranker.status === 'loaded' || settings.reranker.status === 'online')
const rerankerLoading = computed(() => settings.reranker.status === 'loading')
</script>

<template>
  <!-- 触发按钮（由父组件放置，这里也导出一个 open 供复用） -->
  <n-modal
    v-model:show="settings.settingsOpen"
    preset="card"
    :title="'设置'"
    style="width: 720px; max-width: 94vw;"
    :bordered="false"
    :theme="naiveTheme"
  >
    <div class="settings-wrap">
      <!-- Tab 切换 -->
      <div class="settings-tabs">
        <button class="tab-btn" :class="{ active: tab === 'models' }" @click="tab = 'models'">
          <Cpu :size="14" />模型
        </button>
        <button class="tab-btn" :class="{ active: tab === 'dbs' }" @click="tab = 'dbs'">
          <Database :size="14" />数据库
        </button>
        <button class="tab-btn" :class="{ active: tab === 'search' }" @click="tab = 'search'">
          <Search :size="14" />检索
        </button>
      </div>

      <!-- ============ 模型 Tab ============ -->
      <div v-if="tab === 'models'" class="tab-body">
        <div class="kind-row">
          <button
            v-for="k in MODEL_KINDS"
            :key="k.kind"
            class="kind-btn"
            :class="{ active: activeKind === k.kind }"
            :title="k.desc"
            @click="activeKind = k.kind"
          >{{ k.label }}</button>
          <button class="add-btn" @click="openAdd(activeKind)"><Plus :size="13" />添加</button>
        </div>

        <div class="model-list">
          <div v-if="!modelsOf.length" class="empty">暂无{{ activeKind }}模型</div>
          <div v-for="m in modelsOf" :key="m.name" class="model-item">
            <div class="m-info">
              <div class="m-name">
                {{ m.name }}
                <span v-if="isCurrent(activeKind, m.name)" class="m-current">当前</span>
              </div>
              <div class="m-desc">
                {{ m.type === 'local' ? (m.model_path || '本地模型') : (m.model || m.base_url || '') }}
              </div>
            </div>
            <div class="m-ops">
              <button v-if="!isCurrent(activeKind, m.name)" class="op-btn" title="设为当前" @click="selectModel(activeKind, m.name)">
                <Check :size="13" />
              </button>
              <button class="op-btn" title="编辑" @click="openEdit(activeKind, m)"><Pencil :size="13" /></button>
              <button class="op-btn danger" title="删除" @click="delModel(activeKind, m.name)"><Trash2 :size="13" /></button>
            </div>
          </div>
        </div>
      </div>

      <!-- ============ 数据库 Tab ============ -->
      <div v-else-if="tab === 'dbs'" class="tab-body">
        <div class="kind-row">
          <span class="kind-hint">数据库与向量库（Milvus）</span>
          <button class="add-btn" @click="openAddDb"><Plus :size="13" />添加</button>
        </div>
        <div class="model-list">
          <div v-if="!settings.dbs.length" class="empty">暂无数据库</div>
          <div v-for="d in settings.dbs" :key="d.name" class="model-item">
            <div class="m-info">
              <div class="m-name">
                {{ d.name }}
                <span v-if="settings.current.db === d.name" class="m-current">当前</span>
              </div>
              <div class="m-desc">{{ d.type === 'local' ? '本地' : '在线' }} · {{ d.url }}</div>
            </div>
            <div class="m-ops">
              <button v-if="settings.current.db !== d.name" class="op-btn" title="设为当前" @click="selectDb(d.name)">
                <Check :size="13" />
              </button>
              <button class="op-btn danger" title="删除" @click="delDb(d.name)"><Trash2 :size="13" /></button>
            </div>
          </div>
        </div>
      </div>

      <!-- ============ 检索 Tab ============ -->
      <div v-else class="tab-body">
        <div class="search-group">
          <div class="group-title">检索模式</div>
          <div class="radio-row">
            <label v-for="rm in RETRIEVAL_MODES" :key="rm.value" class="radio-label">
              <input
                type="radio"
                :value="rm.value"
                :checked="settings.search.retrieval_mode === rm.value"
                @change="settings.setRetrievalMode(rm.value)"
              />
              {{ rm.label }}
            </label>
          </div>
        </div>

        <div class="search-group">
          <div class="group-title">章节摘要</div>
          <label class="switch-row">
            <input type="checkbox" :checked="settings.summaryEnabled" @change="settings.setSummary($event.target.checked)" />
            <span>入库时生成章节摘要（增强结构树检索）</span>
          </label>
        </div>

        <div class="search-group">
          <div class="group-title">工具调用</div>
          <label class="switch-row">
            <input type="checkbox" :checked="settings.toolCallingEnabled" @change="settings.setToolCalling($event.target.checked)" />
            <span>可插拔工具决策（决定是否调用 MCP 工具）</span>
          </label>
        </div>

        <div class="search-group">
          <div class="group-title">Reranker 重排序模型</div>
          <div class="reranker-row">
            <span class="rr-status" :class="{ ok: rerankerOk }">{{ rerankerText }}</span>
            <button
              class="add-btn"
              :disabled="rerankerLoading"
              @click="settings.loadReranker()"
            >{{ rerankerLoading ? '加载中...' : '下载并加载' }}</button>
          </div>
          <div v-if="settings.reranker.message" class="rr-msg">{{ settings.reranker.message }}</div>
        </div>
      </div>
    </div>

    <!-- 模型表单弹窗 -->
    <n-modal
      v-model:show="formVisible"
      preset="card"
      :title="(formMode === 'add' ? '添加' : '编辑') + '模型'"
      style="width: 520px; max-width: 94vw;"
      :theme="naiveTheme"
    >
      <div class="form-body">
        <div class="form-row">
          <label>名称 <span class="req">*</span></label>
          <input v-model="form.name" placeholder="模型显示名称" />
        </div>
        <div v-if="isReranker" class="form-row">
          <label>类型</label>
          <select v-model="form.type">
            <option value="online">在线 API</option>
            <option value="local">本地模型</option>
          </select>
        </div>
        <template v-if="!isLocal">
          <div class="form-row">
            <label>模型名</label>
            <input v-model="form.model" placeholder="如 deepseek-chat / BAAI/bge-m3" />
          </div>
          <div class="form-row">
            <label>API 地址</label>
            <input v-model="form.base_url" placeholder="https://..." />
          </div>
          <div class="form-row">
            <label>API 密钥</label>
            <input v-model="form.api_key" type="password" placeholder="密钥" />
          </div>
          <div v-if="hasProtocol" class="form-row">
            <label>协议</label>
            <select v-model="form.protocol">
              <option value="openai">openai</option>
              <option value="doubao">doubao</option>
            </select>
          </div>
        </template>
        <template v-else>
          <div class="form-row">
            <label>本地路径</label>
            <input v-model="form.model_path" placeholder="如 BAAI/bge-reranker-v2-m3 或本地绝对路径" />
          </div>
        </template>
        <div class="form-actions">
          <button class="cancel-btn" @click="formVisible = false">取消</button>
          <button class="ok-btn" @click="submitModel">保存</button>
        </div>
      </div>
    </n-modal>

    <!-- 数据库表单弹窗 -->
    <n-modal
      v-model:show="dbFormVisible"
      preset="card"
      title="添加数据库"
      style="width: 520px; max-width: 94vw;"
      :theme="naiveTheme"
    >
      <div class="form-body">
        <div class="form-row">
          <label>名称 <span class="req">*</span></label>
          <input v-model="dbForm.name" placeholder="数据库显示名称" />
        </div>
        <div class="form-row">
          <label>类型</label>
          <select v-model="dbForm.type">
            <option value="local">本地</option>
            <option value="online">在线</option>
          </select>
        </div>
        <div class="form-row">
          <label>URL</label>
          <input v-model="dbForm.url" placeholder="http://localhost:19530" />
        </div>
        <div class="form-row" v-if="dbForm.type === 'online'">
          <label>Token</label>
          <input v-model="dbForm.token" placeholder="访问令牌" />
        </div>
        <div class="form-row">
          <label>库名</label>
          <input v-model="dbForm.db_name" placeholder="default" />
        </div>
        <div class="form-actions">
          <button class="cancel-btn" @click="dbFormVisible = false">取消</button>
          <button class="ok-btn" @click="submitDb">保存</button>
        </div>
      </div>
    </n-modal>
  </n-modal>
</template>

<style scoped>
.settings-wrap {
  display: flex;
  flex-direction: column;
}
.settings-tabs {
  display: flex;
  gap: 6px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin-bottom: 14px;
}
.tab-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  border: 1px solid transparent;
  background: none;
  border-radius: 8px;
  padding: 6px 14px;
  font-size: 13px;
  cursor: pointer;
  color: var(--text-secondary);
  transition: all 0.15s;
}
.tab-btn:hover {
  color: var(--primary);
}
.tab-btn.active {
  background: var(--grad-primary);
  color: #fff;
}
.tab-body {
  min-height: 300px;
  max-height: 60vh;
  overflow-y: auto;
}
.kind-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.kind-btn {
  border: 1px solid var(--border-strong);
  background: var(--panel-bg);
  border-radius: 8px;
  padding: 4px 12px;
  font-size: 12px;
  cursor: pointer;
  color: var(--text-secondary);
  transition: all 0.15s;
}
.kind-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}
.kind-btn.active {
  background: var(--grad-primary);
  color: #fff;
  border-color: transparent;
}
.kind-hint {
  flex: 1;
  font-size: 13px;
  color: var(--text-secondary);
}
.add-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  border: none;
  border-radius: 8px;
  padding: 5px 12px;
  font-size: 12px;
  cursor: pointer;
  background: var(--grad-primary);
  color: #fff;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.28);
  transition: all 0.15s;
}
.add-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
.add-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.model-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.empty {
  padding: 24px;
  text-align: center;
  font-size: 12px;
  color: var(--text-faint);
}
.model-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--panel-sub-bg);
}
.m-info {
  flex: 1;
  min-width: 0;
}
.m-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 6px;
}
.m-current {
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 8px;
  background: var(--primary-soft);
  color: var(--primary);
  font-weight: 600;
}
.m-desc {
  font-size: 11px;
  color: var(--text-faint);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.m-ops {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}
.op-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 4px;
  display: flex;
  align-items: center;
  border-radius: 5px;
  transition: all 0.15s;
}
.op-btn:hover {
  color: var(--primary);
  background: var(--primary-soft);
}
.op-btn.danger:hover {
  color: var(--danger);
  background: rgba(220, 38, 38, 0.1);
}
.search-group {
  margin-bottom: 18px;
}
.group-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 8px;
}
.radio-row {
  display: flex;
  gap: 16px;
}
.radio-label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
}
.radio-label input {
  accent-color: var(--primary);
  cursor: pointer;
}
.switch-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
}
.switch-row input {
  accent-color: var(--primary);
  cursor: pointer;
}
.reranker-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.rr-status {
  font-size: 13px;
  color: var(--text-faint);
}
.rr-status.ok {
  color: var(--success);
}
.rr-msg {
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-muted);
}
.form-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.form-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.form-row label {
  width: 90px;
  flex-shrink: 0;
  font-size: 13px;
  color: var(--text-secondary);
}
.form-row .req {
  color: var(--danger);
}
.form-row input,
.form-row select {
  flex: 1;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  background: var(--panel-bg);
  color: var(--text);
  outline: none;
  transition: border-color 0.2s;
}
.form-row input:focus,
.form-row select:focus {
  border-color: var(--primary);
}
.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 6px;
}
.cancel-btn {
  border: 1px solid var(--border-strong);
  background: var(--panel-bg);
  border-radius: 8px;
  padding: 7px 16px;
  font-size: 13px;
  cursor: pointer;
  color: var(--text-secondary);
}
.cancel-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}
.ok-btn {
  border: none;
  border-radius: 8px;
  padding: 7px 18px;
  font-size: 13px;
  cursor: pointer;
  background: var(--grad-primary);
  color: #fff;
}
</style>
