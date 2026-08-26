<script setup>
import { Plus } from '@lucide/vue'
import { useChatStore } from '../../stores/chat'

const chat = useChatStore()
</script>

<template>
  <div class="conv-sidebar">
    <button class="new-conv-btn" @click="chat.newConversation()">
      <Plus :size="15" /><span>新建对话</span>
    </button>
    <div class="conv-list">
      <div v-if="!chat.conversations.length" class="conv-empty">暂无历史对话</div>
      <div
        v-for="c in chat.conversations"
        :key="c.conversation_id"
        class="conv-item"
        :class="{ active: c.conversation_id === chat.currentConversationId }"
        :title="c.title"
        @click="chat.openConversation(c.conversation_id)"
      >
        <span class="conv-title">{{ c.title }}</span>
        <span class="conv-count">{{ c.message_count || 0 }}</span>
        <button
          class="conv-del"
          title="删除对话"
          @click.stop="chat.deleteConversation(c.conversation_id)"
        >×</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.conv-sidebar {
  width: 210px;
  flex-shrink: 0;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  background: var(--panel-sub-bg);
}
.new-conv-btn {
  margin: 10px;
  padding: 8px;
  border: 1px dashed var(--border-strong);
  border-radius: 8px;
  background: none;
  color: var(--primary);
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-weight: 500;
  transition: all 0.15s;
}
.new-conv-btn:hover {
  background: var(--grad-primary);
  color: #fff;
  border-color: transparent;
}
.conv-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px 8px;
}
.conv-empty {
  padding: 20px 10px;
  text-align: center;
  font-size: 12px;
  color: var(--text-faint);
}
.conv-item {
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: all 0.15s;
}
.conv-item:hover {
  background: rgba(128, 128, 128, 0.12);
  color: var(--text);
}
.conv-item.active {
  background: var(--grad-primary);
  color: #fff;
}
.conv-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.conv-count {
  font-size: 11px;
  opacity: 0.7;
  margin-left: 6px;
}
.conv-del {
  display: none;
  border: none;
  background: none;
  color: inherit;
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
  padding: 0 2px;
  opacity: 0.7;
  flex-shrink: 0;
}
.conv-item:hover .conv-del,
.conv-item.active .conv-del {
  display: block;
}
.conv-del:hover {
  opacity: 1;
}
</style>
