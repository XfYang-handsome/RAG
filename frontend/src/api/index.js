// 后端接口层：与 server.py / mcp_service/manager.py 一一对应
import { http } from './http'

export const api = {
  // ============ 健康 / 日志 ============
  health: () => http.get('/health'),
  logs: (limit = 200) => http.get(`/logs?limit=${limit}`),

  // ============ 对话 ============
  listConversations: () => http.get('/conversations'),
  createConversation: (title = '') => http.post('/conversations', { title }),
  getConversation: (id) => http.get(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id) => http.del(`/conversations/${encodeURIComponent(id)}`),

  // ============ 上传 / 入库任务 ============
  upload: (formData) => http.postForm('/upload', formData),
  listUploadTasks: () => http.get('/upload/tasks'),
  getUploadStatus: (taskId) => http.get(`/upload/${taskId}/status`),
  retryUpload: (taskId) => http.post(`/upload/${taskId}/retry`),
  deleteUpload: (taskId) => http.del(`/upload/${taskId}`),

  // ============ 配置 ============
  getConfig: () => http.get('/config'),
  selectConfig: (body) => http.post('/config/select', body),
  setSearch: (body) => http.post('/config/search', body),
  setToolCalling: (body) => http.post('/config/tool_calling', body),
  setSummary: (body) => http.post('/config/summary', body),
  getSystemPrompt: () => http.get('/config/system_prompt'),
  setSystemPrompt: (sp) => http.post('/config/system_prompt', { system_prompt: sp }),
  getTheme: () => http.get('/config/theme'),
  saveTheme: (theme) => http.post('/config/theme', { theme }),

  // ============ 模型 ============
  listModels: () => http.get('/models'),
  addModel: (m) => http.post('/models', m),
  updateModel: (kind, name, m) => http.put(`/models/${kind}/${encodeURIComponent(name)}`, m),
  deleteModel: (kind, name) => http.del(`/models/${kind}/${encodeURIComponent(name)}`),

  // ============ Reranker ============
  rerankerStatus: () => http.get('/reranker/status'),
  rerankerLoad: () => http.post('/reranker/load'),

  // ============ 数据库 ============
  listDbs: () => http.get('/dbs'),
  addDb: (d) => http.post('/dbs', d),
  deleteDb: (name) => http.del(`/dbs/${encodeURIComponent(name)}`),

  // ============ 本地数据（父块/子块/文档树） ============
  listLocalDbs: () => http.get('/local/databases'),
  createLocalDb: (dbName) => http.post('/local/databases', { db_name: dbName }),
  listParents: () => http.get('/local/parents'),
  listChildren: (parentId) => http.get(`/local/parents/${encodeURIComponent(parentId)}/children`),
  getTree: (docId) => http.get(`/local/tree/${encodeURIComponent(docId)}`),
  deleteChild: (childId) => http.del(`/local/children/${childId}`),
  deleteParent: (parentId) => http.del(`/local/parents/${encodeURIComponent(parentId)}`),
  deleteSource: (source) => http.del(`/local/sources/${encodeURIComponent(source)}`),
  renameSource: (oldSource, newSource) =>
    http.put('/local/sources/rename', { old_source: oldSource, new_source: newSource }),
  clearLocal: () => http.del('/local/clear'),

  // ============ MCP 服务器 / 工具 ============
  listMcpServers: () => http.get('/mcp/servers'),
  upsertMcpServer: (s) => http.post('/mcp/servers', s),
  deleteMcpServer: (name) => http.del(`/mcp/servers/${encodeURIComponent(name)}`),
  startMcpServer: (name) => http.post(`/mcp/servers/${encodeURIComponent(name)}/start`),
  stopMcpServer: (name) => http.post(`/mcp/servers/${encodeURIComponent(name)}/stop`),
  serverStatus: (name) => http.get(`/mcp/servers/${encodeURIComponent(name)}/status`),
  listMcpTools: (name) => http.get(`/mcp/servers/${encodeURIComponent(name)}/tools`),
  callMcpTool: (name, toolName, args) =>
    http.post(`/mcp/servers/${encodeURIComponent(name)}/call`, {
      tool_name: toolName,
      arguments: args,
    }),
  toggleMcpTool: (toolName, enabled) =>
    http.post(`/mcp/tools/${encodeURIComponent(toolName)}/toggle`, { enabled }),
  getMcpLogs: (name, limit = 200) =>
    http.get(`/mcp/servers/${encodeURIComponent(name)}/logs?limit=${limit}`),
  getMcpSettings: () => http.get('/mcp/settings'),
  setMcpSettings: (body) => http.post('/mcp/settings', body),

  // ============ 目录浏览（本地模型路径选择） ============
  browse: (path = '') => http.get(`/browse?path=${encodeURIComponent(path)}`),
  pickDirectory: () => http.get('/pick-directory'),
}
