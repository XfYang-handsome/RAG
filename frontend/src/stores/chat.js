import { defineStore } from 'pinia'
import { api } from '../api'
import { streamChat } from '../api/sse'
import { useSettingsStore } from './settings'

const PERSIST_KEY = 'rag-chat-state'

// 模块级流式缓冲（非响应式，避免每 token 触发一次渲染）
let tokenBuffer = ''
let flushPending = false

let msgSeq = 0
function makeMessage(role, content = '') {
  return {
    id: ++msgSeq,
    role,
    content,
    reasoning: '',
    citations: [],
    agentTrace: [],
    status: '', // 生成中状态（done 后清空）
    warning: '', // 警告（保留）
    errorText: '', // 错误（保留）
    pending: false,
  }
}

function loadPersisted() {
  try {
    const raw = localStorage.getItem(PERSIST_KEY)
    if (raw) return JSON.parse(raw)
  } catch (e) {
    /* 损坏的缓存忽略 */
  }
  return { messages: [], currentConversationId: '' }
}

export const useChatStore = defineStore('chat', {
  state: () => {
    const p = loadPersisted()
    return {
      messages: (p.messages || []).map((m) => ({ ...makeMessage(m.role, m.content) })),
      currentConversationId: p.currentConversationId || '',
      conversations: [],
      chatMode: 'rag', // rag / agentic / direct
      isStreaming: false,
    }
  },

  getters: {
    hasMessages: (s) => s.messages.length > 0,
  },

  actions: {
    // 持久化（只存 role + content，思考/引用/轨迹为流式临时态，不落盘）
    persist() {
      try {
        const slim = this.messages.map((m) => ({ role: m.role, content: m.content }))
        localStorage.setItem(
          PERSIST_KEY,
          JSON.stringify({ messages: slim, currentConversationId: this.currentConversationId }),
        )
      } catch (e) {
        /* ignore */
      }
    },

    persistThrottled() {
      if (this._persistTimer) return
      this._persistTimer = setTimeout(() => {
        this._persistTimer = null
        this.persist()
      }, 400)
    },

    async loadConversations() {
      try {
        const d = await api.listConversations()
        this.conversations = d.conversations || []
        // 校验当前对话是否仍存在：删除所有历史后，localStorage 可能残留已删除的
        // conversation_id，若不清理，send() 会误判"已有对话"而不新建，导致后端
        // 写入不存在的 conversation、标题无法更新。
        if (
          this.currentConversationId &&
          !this.conversations.some((c) => c.conversation_id === this.currentConversationId)
        ) {
          this.currentConversationId = ''
          this.messages = []
          this.persist()
        }
      } catch (e) {
        this.conversations = []
      }
    },

    async newConversation() {
      const d = await api.createConversation('')
      if (!d.success) throw new Error('新建对话失败')
      this.currentConversationId = d.conversation.conversation_id
      this.messages = []
      this.persist()
      await this.loadConversations()
    },

    async openConversation(id) {
      const d = await api.getConversation(id)
      if (!d.success) throw new Error('加载对话失败')
      this.currentConversationId = id
      this.messages = (d.messages || []).map((m) => makeMessage(m.role, m.content))
      this.persist()
      await this.loadConversations()
    },

    async deleteConversation(id) {
      await api.deleteConversation(id)
      if (id === this.currentConversationId) {
        this.currentConversationId = ''
        this.messages = []
        this.persist()
      }
      await this.loadConversations()
    },

    setChatMode(mode) {
      this.chatMode = mode
    },

    async send(question) {
      if (!question || this.isStreaming) return

      // 无当前对话时自动新建，保证消息能持久化到后端
      if (!this.currentConversationId) {
        try {
          const d = await api.createConversation('')
          this.currentConversationId = d.conversation.conversation_id
        } catch (e) {
          /* 新建失败不阻断发送，仅不持久化到后端 */
        }
      }

      this.messages.push(makeMessage('user', question))
      const aiMsg = makeMessage('assistant')
      aiMsg.pending = true
      this.messages.push(aiMsg)
      // 关键：push 后 aiMsg 仍是「原始对象」，但 Pinia 数组里存的是「响应式代理」。
      // 直接改 aiMsg 不会触发 Vue 更新（导致流式内容不刷新、pending 不更新、LaTeX 不渲染），
      // 必须从响应式数组取回代理引用再修改。
      const ai = this.messages[this.messages.length - 1]
      this.persist()
      this.isStreaming = true

      const historyForRequest = this.messages.slice(0, -2).map((m) => ({
        role: m.role,
        content: m.content,
      }))

      try {
        // 检索模式从 settings store 读取（config 加载的真实值，而非前端硬编码）
        const settings = useSettingsStore()
        const retrievalMode = settings.search.retrieval_mode
        await streamChat(
          {
            question,
            history: historyForRequest,
            top_k: 5,
            mode: this.chatMode,
            retrieval_mode: retrievalMode,
            conversation_id: this.currentConversationId,
          },
          (ev) => this._handleEvent(ai, ev),
        )
      } catch (e) {
        if (!ai.content) ai.content = `[请求失败: ${e.message}]`
      } finally {
        // 冲刷剩余 tokenBuffer，确保不丢最后一段
        if (flushPending) {
          cancelAnimationFrame(flushPending)
          flushPending = false
          if (tokenBuffer) {
            ai.content += tokenBuffer
            tokenBuffer = ''
          }
        }
        ai.status = ''
        ai.pending = false
        this.isStreaming = false
        this.persist()
        this.loadConversations()
      }
    },

    _flushToken(aiMsg) {
      flushPending = false
      if (tokenBuffer) {
        aiMsg.content += tokenBuffer
        tokenBuffer = ''
        this.persistThrottled()
      }
    },

    _handleEvent(aiMsg, ev) {
      const { type, content } = ev
      switch (type) {
        case 'token':
          // rAF 节流：合并同一帧内的 token，避免每 token 触发一次渲染
          tokenBuffer += content
          if (!flushPending) {
            flushPending = requestAnimationFrame(() => this._flushToken(aiMsg))
          }
          break
        case 'reasoning':
          aiMsg.reasoning += content
          break
        case 'status':
          aiMsg.status = content
          break
        case 'warning':
          aiMsg.warning = content
          break
        case 'citations':
          try {
            aiMsg.citations = JSON.parse(content)
          } catch (e) {
            /* ignore */
          }
          break
        case 'agent_trace':
          try {
            aiMsg.agentTrace.push(JSON.parse(content))
          } catch (e) {
            /* ignore */
          }
          break
        case 'error':
          aiMsg.errorText = content
          break
        case 'done':
          break
        default:
          break
      }
    },
  },
})
