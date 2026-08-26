<script setup>
import { computed } from 'vue'
import { Database, RefreshCw, Trash2, ChevronRight, FileText } from '@lucide/vue'
import { useSettingsStore } from '../../stores/settings'
import CollapsibleCard from './CollapsibleCard.vue'
import TreeNode from './TreeNode.vue'

const settings = useSettingsStore()

const dbOk = computed(() => settings.health.db_available)

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function isTreeDoc(p) {
  return !!p.is_tree_doc
}

async function delParent(p) {
  const title = isTreeDoc(p) ? '此结构树文档' : '此父块'
  if (!confirm(`确定删除${title}吗？`)) return
  await settings.deleteParent(p.parent_id)
}

async function delSource(source) {
  if (!confirm(`确定删除文件 "${source}" 及其所有分块吗？此操作不可恢复！`)) return
  await settings.deleteSource(source)
}

async function renameSource(source) {
  const newName = prompt(`重命名文件 "${source}"：`, source)
  if (newName === null) return
  const n = newName.trim()
  if (!n) return alert('文件名不能为空')
  if (n === source) return
  await settings.renameSource(source, n)
}

async function clearAll() {
  if (!confirm('确定清空全部数据吗？此操作不可恢复！')) return
  await settings.clearLocal()
}
</script>

<template>
  <CollapsibleCard title="数据库管理" persist-key="db">
    <template #icon><Database :size="15" /></template>
    <template #actions>
      <span class="health" :class="dbOk ? 'ok' : 'err'">
        <span class="dot"></span>{{ dbOk ? '已连接' : '未连接' }}
      </span>
      <button class="icon-btn" title="刷新" @click="settings.refreshParents(true)"><RefreshCw :size="14" /></button>
      <button class="icon-btn danger" title="清空全部" @click="clearAll"><Trash2 :size="14" /></button>
    </template>

    <div class="db-stats">
      父块 <b>{{ settings.health.parent_count }}</b> · 子块 <b>{{ settings.health.count }}</b>
    </div>
    <div v-if="settings.clearMsg" class="clear-msg">{{ settings.clearMsg }}</div>

    <div class="db-list">
      <div v-if="!settings.parentGroups.length" class="db-empty">暂无数据</div>
      <div v-for="g in settings.parentGroups" :key="g.source" class="src-group">
        <div class="src-head" @click="settings.toggleSource(g.source)">
          <ChevronRight
            :size="13"
            class="src-arrow"
            :style="{ transform: settings.expandedSources[g.source] ? 'rotate(90deg)' : '' }"
          />
          <FileText :size="13" class="src-icon" />
          <span class="src-name">{{ g.source }}</span>
          <span class="src-count">{{ g.parents.length }}</span>
          <span class="src-ops" @click.stop>
            <button class="mini-btn" title="重命名" @click="renameSource(g.source)">✎</button>
            <button class="mini-btn danger" title="删除文件" @click="delSource(g.source)">✕</button>
          </span>
        </div>

        <div v-if="settings.expandedSources[g.source]" class="src-body">
          <div v-for="p in g.parents" :key="p.parent_id" class="parent">
            <div class="parent-head" @click="settings.toggleParent(p)">
              <span class="parent-arrow">
                {{ settings.expandedParents[p.parent_id] ? '▼' : '▶' }}
              </span>
              <span class="parent-tag">{{ isTreeDoc(p) ? '🌳 结构树文档' : `[${p.parent_index}]` }}</span>
              <span class="parent-preview">{{ (p.text_preview || p.text || '').substring(0, 60) }}</span>
              <span class="parent-count">{{ p.child_count || 0 }}子块</span>
              <button class="mini-btn danger" title="删除" @click.stop="delParent(p)">✕</button>
            </div>

            <div v-if="settings.expandedParents[p.parent_id]" class="parent-children">
              <div v-if="isTreeDoc(p)" class="children-box">
                <TreeNode :nodes="settings.treeCache[p.parent_id] || []" :depth="0" />
              </div>
              <div v-else class="children-box">
                <div v-if="!(settings.childrenCache[p.parent_id] || []).length" class="children-empty">无子块</div>
                <div
                  v-for="c in settings.childrenCache[p.parent_id] || []"
                  :key="c.id"
                  class="child"
                >
                  <span class="child-text">{{ c.text_preview || c.text || '' }}</span>
                  <button class="mini-btn danger" title="删除此子块" @click="settings.deleteChild(c.id)">✕</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </CollapsibleCard>
</template>

<style scoped>
.head-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.health {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
}
.health .dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}
.health.ok { color: var(--success); }
.health.ok .dot { background: var(--success); }
.health.err { color: var(--danger); }
.health.err .dot { background: var(--danger); }
.icon-btn {
  background: none;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  padding: 4px 6px;
  cursor: pointer;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  transition: all 0.15s;
}
.icon-btn:hover { color: var(--primary); border-color: var(--primary); }
.icon-btn.danger:hover { color: var(--danger); border-color: var(--danger); }
.db-stats {
  padding: 8px 16px;
  font-size: 12px;
  color: var(--text-muted);
}
.db-stats b { color: var(--text); }
.clear-msg {
  padding: 4px 16px;
  font-size: 12px;
  color: var(--success);
}
.db-list {
  max-height: 300px;
  overflow-y: auto;
}
.db-empty {
  padding: 24px;
  text-align: center;
  font-size: 12px;
  color: var(--text-faint);
}
.src-group {
  border-top: 1px solid var(--border);
}
.src-head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  transition: background 0.15s;
}
.src-head:hover { background: rgba(128, 128, 128, 0.06); }
.src-arrow {
  color: var(--text-faint);
  transition: transform 0.2s;
  flex-shrink: 0;
}
.src-icon { color: var(--primary); flex-shrink: 0; }
.src-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.src-count {
  font-size: 11px;
  color: var(--text-faint);
  background: var(--panel-sub-bg);
  border-radius: 8px;
  padding: 1px 8px;
}
.src-ops { display: flex; gap: 2px; }
.mini-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 12px;
  padding: 2px 4px;
}
.mini-btn:hover { color: var(--primary); }
.mini-btn.danger:hover { color: var(--danger); }
.src-body { background: var(--panel-sub-bg); }
.parent { border-top: 1px solid var(--border-light); }
.parent-head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px 6px 24px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary);
  transition: background 0.15s;
}
.parent-head:hover { background: rgba(128, 128, 128, 0.06); }
.parent-arrow {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--text-faint);
  width: 12px;
}
.parent-tag { flex-shrink: 0; color: var(--text-muted); }
.parent-preview {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.parent-count {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-faint);
}
.parent-children { background: var(--panel-sub-bg); border-top: 1px solid var(--border-light); }
.children-box { padding-bottom: 2px; }
.children-empty {
  padding: 8px;
  font-size: 11px;
  color: var(--text-faint);
  text-align: center;
}
.child {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 4px 10px 4px 34px;
  font-size: 11px;
  color: var(--text-muted);
  border-top: 1px solid var(--border-light);
}
.child-text {
  flex: 1;
  word-break: break-all;
  line-height: 1.4;
}
</style>
