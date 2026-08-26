import { defineStore } from 'pinia'
import { api } from '../api'

export const useSettingsStore = defineStore('settings', {
  state: () => ({
    // 健康状态
    health: { db_available: false, parent_count: 0, count: 0, checked: false, online: false },
    // 数据库父块（按 source 分组）
    parentGroups: [],
    parentCache: null,
    expandedSources: {}, // source -> true
    childrenCache: {}, // parentId -> [children]
    expandedParents: {}, // parentId -> true
    treeCache: {}, // docId -> { expanded, nodes }
    clearMsg: '',
    // 日志
    logs: [],
    // 系统提示词
    systemPrompt: '',
    // 配置
    models: {},
    dbs: [],
    current: {},
    search: { rewrite: false, hybrid: true, retrieval_mode: 'hybrid' },
    summaryEnabled: true,
    toolCallingEnabled: false,
    reranker: { status: 'idle', message: '' }, // idle/loading/loaded/error/online
    settingsOpen: false,
  }),

  actions: {
    // ============ 健康检查 ============
    async checkHealth() {
      try {
        const d = await api.health()
        this.health = {
          db_available: !!d.db_available,
          parent_count: d.parent_count || 0,
          count: d.count || 0,
          checked: true,
          online: true,
        }
      } catch (e) {
        this.health = { db_available: false, parent_count: 0, count: 0, checked: true, online: false }
      }
      if (this.health.db_available) this.refreshParents(false)
    },

    // ============ 数据库父块 ============
    async refreshParents(force) {
      if (!force && this.parentCache) {
        this.parentGroups = this.parentCache
        return
      }
      if (force) {
        // 数据变更后子块/树缓存失效，下次展开重新拉取
        this.childrenCache = {}
        this.treeCache = {}
        this.expandedParents = {}
      }
      try {
        const d = await api.listParents()
        const parents = (d && d.success) ? d.parents || [] : []
        const groups = {}
        for (const p of parents) {
          const key = p.source || '未知来源'
          if (!groups[key]) groups[key] = []
          groups[key].push(p)
        }
        const arr = Object.entries(groups).map(([source, list]) => ({ source, parents: list }))
        this.parentCache = arr
        this.parentGroups = arr
      } catch (e) {
        this.parentGroups = []
      }
    },

    toggleSource(source) {
      this.expandedSources[source] = !this.expandedSources[source]
    },

    // 展开/收起父块（普通父块走 children 接口；结构树文档 is_tree_doc 走 tree 接口）
    async toggleParent(p) {
      const key = p.parent_id
      if (this.expandedParents[key]) {
        this.expandedParents[key] = false
        return
      }
      this.expandedParents[key] = true
      if (p.is_tree_doc) {
        if (this.treeCache[key]) return
        try {
          const docId = String(key).replace(/^tree:/, '')
          const d = await api.getTree(docId)
          this.treeCache[key] = (d && d.tree && d.tree.children) || []
        } catch (e) {
          this.treeCache[key] = []
        }
      } else {
        if (this.childrenCache[key]) return
        try {
          const d = await api.listChildren(key)
          this.childrenCache[key] = (d && d.children) || []
        } catch (e) {
          this.childrenCache[key] = []
        }
      }
    },

    async deleteParent(pid) {
      await api.deleteParent(pid)
      this._flashClearMsg('父块已删除')
      this.refreshParents(true)
      this.checkHealth()
    },

    async deleteSource(source) {
      await api.deleteSource(source)
      this._flashClearMsg('文件已删除')
      this.refreshParents(true)
      this.checkHealth()
    },

    async renameSource(oldSource, newSource) {
      await api.renameSource(oldSource, newSource)
      this.refreshParents(true)
    },

    async deleteChild(childId) {
      await api.deleteChild(childId)
      this._flashClearMsg('子块已删除')
      this.refreshParents(true)
      this.checkHealth()
    },

    async clearLocal() {
      await api.clearLocal()
      this.parentCache = null
      this.parentGroups = []
      this.childrenCache = {}
      this.expandedParents = {}
      this._flashClearMsg('已清空全部数据')
      this.checkHealth()
    },

    _flashClearMsg(msg) {
      this.clearMsg = msg
      setTimeout(() => {
        if (this.clearMsg === msg) this.clearMsg = ''
      }, 3000)
    },

    // ============ 日志 ============
    async refreshLogs() {
      try {
        const d = await api.logs(200)
        this.logs = (d && d.logs) || []
      } catch (e) {
        this.logs = []
      }
    },

    // ============ 配置 ============
    async loadConfig() {
      const c = await api.getConfig()
      this.models = c.models || {}
      this.dbs = c.dbs || []
      this.current = c.current || {}
      this.search = {
        rewrite: !!(c.search && c.search.rewrite),
        hybrid: !!(c.search && c.search.hybrid),
        retrieval_mode: (c.search && c.search.retrieval_mode) || 'hybrid',
      }
      this.summaryEnabled = !!(c.summary && c.summary.enabled)
      this.toolCallingEnabled = !!(c.tool_calling && c.tool_calling.enabled)
      this.systemPrompt = c.system_prompt || ''
    },

    async loadSystemPrompt() {
      try {
        const d = await api.getSystemPrompt()
        this.systemPrompt = d.system_prompt || ''
      } catch (e) {
        /* ignore */
      }
    },

    async saveSystemPrompt() {
      await api.setSystemPrompt(this.systemPrompt)
    },

    // ============ 模型 / 数据库 ============
    async addModel(kind, form) {
      await api.addModel({ kind, ...form })
      await this.loadConfig()
    },
    async updateModel(kind, name, form) {
      await api.updateModel(kind, name, form)
      await this.loadConfig()
    },
    async deleteModel(kind, name) {
      await api.deleteModel(kind, name)
      await this.loadConfig()
    },
    async addDb(form) {
      await api.addDb(form)
      await this.loadConfig()
    },
    async deleteDb(name) {
      await api.deleteDb(name)
      await this.loadConfig()
    },
    async selectModel(kind, name) {
      await api.selectConfig({ [kind]: name })
      await this.loadConfig()
    },
    async selectDb(name) {
      await api.selectConfig({ db: name })
      await this.loadConfig()
    },

    // ============ 检索 / 摘要 / 工具调用 ============
    async setRetrievalMode(mode) {
      await api.setSearch({ retrieval_mode: mode })
      this.search.retrieval_mode = mode
    },
    async setSummary(enabled) {
      await api.setSummary({ enabled })
      this.summaryEnabled = enabled
    },
    async setToolCalling(enabled) {
      await api.setToolCalling({ enabled })
      this.toolCallingEnabled = enabled
    },

    // ============ Reranker ============
    async refreshReranker() {
      try {
        const d = await api.rerankerStatus()
        this.reranker.status = d.status || 'idle'
        if (d.message) this.reranker.message = d.message
      } catch (e) {
        /* ignore */
      }
    },
    async loadReranker() {
      if (this.reranker.status === 'loading') return
      this.reranker.status = 'loading'
      try {
        await api.rerankerLoad()
        this._pollReranker()
      } catch (e) {
        this.reranker.status = 'error'
        this.reranker.message = e.message
      }
    },
    // 轮询 /reranker/status，直到结束（loaded/error/online/idle）
    async _pollReranker() {
      if (this._rrTimer) clearInterval(this._rrTimer)
      this._rrTimer = setInterval(async () => {
        try {
          const d = await api.rerankerStatus()
          this.reranker.status = d.status || 'idle'
          if (d.message) this.reranker.message = d.message
          if (d.status !== 'loading') {
            clearInterval(this._rrTimer)
            this._rrTimer = null
          }
        } catch (e) {
          clearInterval(this._rrTimer)
          this._rrTimer = null
        }
      }, 3000)
    },
  },
})
