import { defineStore } from 'pinia'
import { api } from '../api'

export const useMcpStore = defineStore('mcp', {
  state: () => ({
    servers: [],
    toolsCache: {}, // {serverName: [tool]}
    selectedServer: null,
    selectedLogs: [],
    toolLlms: [],
    currentToolLlm: '',
    loading: false,
    // 调试面板运行时状态（serverName|toolName -> {visible, args, result, err, running}）
    tests: {},
  }),

  actions: {
    // ============ 工具决策模型 ============
    async loadToolLlms() {
      try {
        const d = await api.getConfig()
        const llms = d.models.tool_llm || []
        this.toolLlms = llms
        this.currentToolLlm = d.current.tool_llm || (llms[0] && llms[0].name) || ''
      } catch (e) {
        this.toolLlms = []
      }
    },
    async selectToolLlm(name) {
      if (!name) return
      try {
        await api.selectConfig({ tool_llm: name })
        this.currentToolLlm = name
      } catch (e) {
        throw e
      }
    },
    async addToolLlm(body) {
      await api.addModel({ kind: 'tool_llm', ...body })
      await this.loadToolLlms()
    },

    // ============ 服务器 ============
    async loadServers(initial = false) {
      if (initial) this.loading = true
      try {
        const d = await api.listMcpServers()
        this.servers = d.servers || []
        if (initial) {
          // 初始加载：拉取每个服务器的工具
          this.servers.forEach((s) => this.loadServerTools(s.name))
        }
      } catch (e) {
        if (initial) this.servers = []
      } finally {
        this.loading = false
      }
    },

    async startServer(name) {
      await api.startMcpServer(name)
      setTimeout(() => {
        this.loadServers()
        this.loadServerTools(name, true)
      }, 1500)
    },
    async stopServer(name) {
      await api.stopMcpServer(name)
      setTimeout(() => this.loadServers(), 800)
    },
    async deleteServer(name) {
      if (!confirm(`确定删除服务器「${name}」吗？`)) return
      await api.deleteMcpServer(name)
      delete this.toolsCache[name]
      await this.loadServers(true)
    },
    async addServer(body) {
      const d = await api.upsertMcpServer(body)
      if (!d.success) throw new Error(d.detail || '新增失败')
      await this.loadServers(true)
    },

    // ============ 工具 ============
    async loadServerTools(name, force = false) {
      if (!force && this.toolsCache[name]) return
      try {
        const d = await api.listMcpTools(name)
        if (!d.success) {
          this.toolsCache[name] = []
          return
        }
        this.toolsCache[name] = d.tools || []
      } catch (e) {
        this.toolsCache[name] = []
      }
    },

    async toggleTool(name, toolName, enabled) {
      try {
        const d = await api.toggleMcpTool(toolName, enabled)
        if (!d.success) throw new Error(d.message || 'failed')
        const t = (this.toolsCache[name] || []).find((x) => x.name === toolName)
        if (t) t.enabled = enabled
      } catch (e) {
        // 失败回滚（重新拉取）
        await this.loadServerTools(name, true)
        throw e
      }
    },

    async callTool(name, toolName, args) {
      const d = await api.callMcpTool(name, toolName, args)
      return d
    },

    // ============ 日志 ============
    async showLogs(name) {
      this.selectedServer = name
      await this.refreshSelectedLogs(true)
    },
    async refreshSelectedLogs(userInitiated = false) {
      if (!this.selectedServer) return
      try {
        const d = await api.getMcpLogs(this.selectedServer)
        this.selectedLogs = d.logs || []
        return userInitiated
      } catch (e) {
        this.selectedLogs = []
      }
    },
  },
})
