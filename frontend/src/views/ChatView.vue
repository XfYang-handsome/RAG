<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { Sparkles, SunMoon, Blocks, Settings2 } from '@lucide/vue'
import { useThemeStore } from '../stores/theme'
import { useChatStore } from '../stores/chat'
import { useSettingsStore } from '../stores/settings'
import { useUploadStore } from '../stores/upload'
import ConversationSidebar from '../components/chat/ConversationSidebar.vue'
import QuestionRail from '../components/chat/QuestionRail.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import MessageItem from '../components/chat/MessageItem.vue'
import DbPanel from '../components/panel/DbPanel.vue'
import UploadPanel from '../components/panel/UploadPanel.vue'
import SystemPromptPanel from '../components/panel/SystemPromptPanel.vue'
import LogPanel from '../components/panel/LogPanel.vue'
import SettingsModal from '../components/panel/SettingsModal.vue'

const router = useRouter()
const themeStore = useThemeStore()
const chat = useChatStore()
const settings = useSettingsStore()
const upload = useUploadStore()

const messagesEl = ref(null)

// 智能滚动：仅当用户接近底部时才跟随
function scrollToBottomIfNear() {
  const el = messagesEl.value
  if (!el) return
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  if (nearBottom) el.scrollTop = el.scrollHeight
}

watch(
  () => chat.messages.length,
  () => requestAnimationFrame(() => {
    const el = messagesEl.value
    if (el) el.scrollTop = el.scrollHeight
  }),
)

watch(
  () => {
    const last = chat.messages[chat.messages.length - 1]
    return last ? last.content.length : 0
  },
  () => scrollToBottomIfNear(),
)

let healthTimer = null
let logTimer = null

onMounted(() => {
  chat.loadConversations()
  settings.loadConfig()
  settings.loadSystemPrompt()
  settings.checkHealth()
  settings.refreshLogs()
  upload.loadHistory()

  healthTimer = setInterval(() => settings.checkHealth(), 30000)
  logTimer = setInterval(() => settings.refreshLogs(), 5000)
})

onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
  if (logTimer) clearInterval(logTimer)
})
</script>

<template>
  <div class="app-shell">
    <!-- 顶栏 -->
    <header class="app-header">
      <div class="header-title">
        <span class="logo"><Sparkles :size="20" color="#fff" /></span>
        <h1>RAG 知识库</h1>
      </div>
      <div class="header-actions">
        <span class="status-dot" :class="settings.health.db_available ? 'ok' : 'err'">
          {{ settings.health.db_available ? '数据库已连接' : '数据库未连接' }}
        </span>
        <button class="header-btn" @click="themeStore.cycle()" title="切换主题">
          <SunMoon :size="16" /><span>{{ themeStore.label }}</span>
        </button>
        <button class="header-btn" @click="router.push('/mcp')" title="MCP 服务器与工具管理">
          <Blocks :size="16" /><span>MCP 管理</span>
        </button>
        <button class="header-btn" @click="settings.settingsOpen = true" title="设置">
          <Settings2 :size="16" /><span>设置</span>
        </button>
      </div>
    </header>

    <!-- 主体 -->
    <div class="container">
      <!-- 左侧聊天面板 -->
      <div class="panel panel-left">
        <div class="chat-body">
          <ConversationSidebar />
          <div class="chat-main">
            <div class="chat-header"><span>对话</span></div>
            <div class="chat-content">
              <div ref="messagesEl" class="chat-messages">
                <div v-if="!chat.messages.length" class="msg msg-system">
                  欢迎使用 RAG 知识库！点击左侧「新建对话」开始。
                </div>
                <MessageItem
                  v-for="(m, i) in chat.messages"
                  :key="m.id"
                  :msg="m"
                  :index="i"
                />
              </div>
              <QuestionRail />
            </div>
            <ChatInput />
          </div>
        </div>
      </div>

      <!-- 右侧面板 -->
      <div class="panel panel-right">
        <div class="right-scroll">
          <DbPanel />
          <UploadPanel />
          <SystemPromptPanel />
          <LogPanel />
        </div>
      </div>
    </div>

    <!-- 设置弹窗 -->
    <SettingsModal />
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
.app-header {
  background: linear-gradient(135deg, var(--header-bg) 0%, #2f2b57 60%, #4338a6 140%);
  color: var(--header-text);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  box-shadow: 0 3px 18px rgba(0, 0, 0, 0.14);
}
.header-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-title h1 {
  font-size: 19px;
  font-weight: 700;
  letter-spacing: 0.2px;
  color: #fff;
}
.logo {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.45);
}
.header-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}
.header-btn {
  background: var(--header-btn);
  color: var(--header-text);
  border: none;
  border-radius: 8px;
  padding: 6px 12px;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  transition: background 0.2s ease, transform 0.15s ease;
}
.header-btn:hover {
  background: var(--header-btn-hover);
  transform: translateY(-1px);
}
.status-dot {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 13px;
  opacity: 0.85;
  color: #fff;
}
.status-dot::before {
  content: '';
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
.status-dot.ok::before {
  background: #4ade80;
}
.status-dot.err::before {
  background: #f87171;
}

/* 主体布局 */
.container {
  width: 100%;
  max-width: 1560px;
  margin: 0 auto;
  padding: 20px;
  display: flex;
  gap: 20px;
  flex: 1 1 0;
  min-height: 0;
  overflow: hidden;
}
.panel {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-left {
  flex: 1;
  min-width: 0;
  min-height: 0;
}
.panel-right {
  width: 420px;
  flex-shrink: 0;
  min-width: 0;
  overflow-y: auto;
  overflow-x: hidden;
  min-height: 0;
}
.right-scroll {
  padding-top: 16px;
}

/* 聊天区 */
.chat-body {
  flex: 1;
  min-height: 0;
  display: flex;
}
.chat-main {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat-header {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  font-size: 15px;
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.chat-content {
  flex: 1 1 0;
  min-height: 0;
  display: flex;
  overflow: hidden;
  align-self: stretch;
}
.chat-messages {
  flex: 1 1 0;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 消息气泡 */
.msg {
  max-width: 85%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
  animation: msgIn 0.3s cubic-bezier(0.21, 1.02, 0.73, 1) both;
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px) scale(0.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.msg-user {
  align-self: flex-end;
  background: var(--grad-primary-strong);
  color: #fff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 3px 12px rgba(99, 102, 241, 0.35);
}
.msg-ai {
  align-self: flex-start;
  background: var(--msg-ai-bg);
  color: var(--text);
  border-bottom-left-radius: 4px;
  position: relative;
  margin-left: 44px;
}
.msg-ai::before {
  content: '✦';
  color: #fff;
  font-size: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: absolute;
  left: -44px;
  top: 0;
  width: 32px;
  height: 32px;
  border-radius: 10px;
  background: var(--grad-primary);
  box-shadow: 0 3px 10px rgba(99, 102, 241, 0.38);
}
.msg-system {
  align-self: center;
  background: var(--warn-bg);
  color: var(--warn-text);
  font-size: 12px;
  padding: 6px 12px;
  border-radius: 8px;
  max-width: 95%;
}
</style>
