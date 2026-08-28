<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  SunMoon, Blocks, Settings2, MessageSquarePlus,
} from '@lucide/vue'
import PrismLogo from '../components/PrismLogo.vue'
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

// 空状态快捷提问
const quickPrompts = [
  '介绍一下这个知识库',
  '帮我检索最近的知识点',
  '总结上传文档的核心内容',
]

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
        <span class="logo"><PrismLogo :size="20" /></span>
        <h1>PrismRAG 知识库</h1>
        <span class="header-sub">检索增强生成 · Agentic</span>
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
                <div v-if="!chat.messages.length" class="welcome">
                  <div class="welcome-orbit">
                    <span class="welcome-logo"><PrismLogo :size="30" color="#fff" /></span>
                    <span class="orbit-dot d1"></span>
                    <span class="orbit-dot d2"></span>
                    <span class="orbit-dot d3"></span>
                  </div>
                  <h2>你好，我是你的知识库助手</h2>
                  <p>基于检索增强生成（RAG），从你的文档中给出可信、可溯源的答案。</p>
                  <div class="welcome-actions">
                    <button
                      v-for="q in quickPrompts"
                      :key="q"
                      class="welcome-chip"
                      @click="chat.send(q)"
                    >
                      <MessageSquarePlus :size="14" /><span>{{ q }}</span>
                    </button>
                  </div>
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
  position: relative;
  background: var(--header-grad);
  color: var(--header-text);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  box-shadow: 0 3px 20px rgba(0, 0, 0, 0.18);
  overflow: hidden;
}
/* 顶栏玻璃高光：柔和光斑 + 顶部细亮边 */
.app-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(420px 160px at 18% 0%, var(--header-glow-a), transparent 60%),
    radial-gradient(360px 140px at 88% 0%, var(--header-glow-b), transparent 60%);
  pointer-events: none;
}
.app-header::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.35), transparent);
  pointer-events: none;
}
.header-title {
  display: flex;
  align-items: center;
  gap: 10px;
  position: relative;
  z-index: 1;
}
.header-title h1 {
  font-size: 19px;
  font-weight: 700;
  letter-spacing: 0.2px;
  color: #fff;
}
.header-sub {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.55);
  letter-spacing: 0.3px;
  padding-left: 12px;
  margin-left: 2px;
  border-left: 1px solid rgba(255, 255, 255, 0.18);
  font-weight: 400;
  white-space: nowrap;
}
.logo {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: var(--grad-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.header-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  position: relative;
  z-index: 1;
}
.header-btn {
  background: rgba(255, 255, 255, 0.1);
  color: var(--header-text);
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 9px;
  padding: 7px 13px;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  transition: background 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
}
.header-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  transform: translateY(-1px);
}
.status-dot {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 13px;
  opacity: 0.9;
  color: #fff;
  margin-right: 4px;
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

/* 消息气泡样式已迁移到 MessageItem.vue（.msg-ai 现为 .ai-row 子元素，需在子组件作用域定义）。
   此处仅保留 msgIn 动画，供空状态欢迎卡片 .welcome 使用。 */
@keyframes msgIn {
  from { opacity: 0; transform: translateY(10px) scale(0.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

/* 空状态欢迎卡片 */
.welcome {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 14px;
  padding: 40px 24px;
  animation: msgIn 0.4s cubic-bezier(0.21, 1.02, 0.73, 1) both;
}
.welcome-orbit {
  position: relative;
  width: 96px;
  height: 96px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.welcome-logo {
  width: 64px;
  height: 64px;
  border-radius: 18px;
  background: var(--grad-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 28px rgba(14, 165, 233, 0.4);
  position: relative;
  z-index: 2;
  animation: float 4s ease-in-out infinite;
}
.orbit-dot {
  position: absolute;
  border-radius: 50%;
  background: var(--primary);
  opacity: 0.5;
  animation: pulse 2.4s ease-in-out infinite;
}
.orbit-dot.d1 { width: 8px; height: 8px; top: 4px; right: 10px; animation-delay: 0s; }
.orbit-dot.d2 { width: 5px; height: 5px; bottom: 12px; left: 4px; animation-delay: 0.8s; }
.orbit-dot.d3 { width: 6px; height: 6px; top: 24px; left: -2px; animation-delay: 1.6s; }
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}
@keyframes pulse {
  0%, 100% { opacity: 0.5; transform: scale(1); }
  50% { opacity: 0.9; transform: scale(1.25); }
}
.welcome h2 {
  font-size: 22px;
  font-weight: 700;
  color: var(--heading);
  margin-top: 6px;
}
.welcome p {
  font-size: 14px;
  color: var(--text-muted);
  max-width: 420px;
  line-height: 1.7;
}
.welcome-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 6px;
}
.welcome-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 9px 16px;
  font-size: 13px;
  color: var(--text-secondary);
  background: var(--panel-bg);
  border: 1px solid var(--border-strong);
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.18s;
  font-weight: 500;
}
.welcome-chip:hover {
  color: var(--primary);
  border-color: var(--primary);
  background: var(--primary-soft);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}
</style>
