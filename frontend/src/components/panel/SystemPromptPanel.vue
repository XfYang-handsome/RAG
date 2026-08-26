<script setup>
import { ref, watch } from 'vue'
import { FileText, Save } from '@lucide/vue'
import { useSettingsStore } from '../../stores/settings'

const settings = useSettingsStore()
const saving = ref(false)

async function save() {
  saving.value = true
  try {
    await settings.saveSystemPrompt()
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="sp-panel">
    <div class="panel-head">
      <div class="panel-title"><FileText :size="15" /> 系统提示词</div>
      <button class="save-btn" :disabled="saving" @click="save">
        <Save :size="13" />{{ saving ? '保存中...' : '保存' }}
      </button>
    </div>
    <div class="sp-body">
      <textarea
        v-model="settings.systemPrompt"
        class="sp-textarea"
        placeholder="输入系统提示词（可选），将注入到每次对话的 system 角色中..."
      ></textarea>
    </div>
  </div>
</template>

<style scoped>
.sp-panel {
  display: flex;
  flex-direction: column;
  margin: 0 16px 16px;
  background: var(--panel-sub-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
}
.panel-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  font-size: 14px;
  color: var(--text);
}
.save-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  background: var(--panel-bg);
  color: var(--text-secondary);
  padding: 5px 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.save-btn:hover:not(:disabled) {
  border-color: var(--primary);
  color: var(--primary);
}
.save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.sp-body {
  padding: 12px 16px;
  display: flex;
}
.sp-textarea {
  flex: 1;
  min-height: 90px;
  resize: vertical;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.6;
  font-family: inherit;
  background: var(--panel-bg);
  color: var(--text);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.sp-textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14);
}
</style>
