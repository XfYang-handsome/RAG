<script setup>
import { reactive } from 'vue'
import { useSettingsStore } from '../../stores/settings'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  depth: { type: Number, default: 0 },
})

const settings = useSettingsStore()
const expanded = reactive({})

const TYPE_LABEL = { paragraph: '段落', table: '表格', figure: '图片' }

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function toggle(n) {
  expanded[n.node_id] = !expanded[n.node_id]
}

function hasContent(n) {
  return (n.children && n.children.length > 0) || (n.chunks && n.chunks.length > 0)
}

async function delChunk(c) {
  if (!confirm('确定删除此子块吗？')) return
  await settings.deleteChild(c.id)
}
</script>

<template>
  <template v-for="n in nodes" :key="n.node_id || JSON.stringify(n)">
    <!-- section 节点 -->
    <div v-if="n.type === 'section'" class="tn">
      <div
        class="tn-section"
        :style="{ paddingLeft: (10 + depth * 16) + 'px' }"
        @click="hasContent(n) && toggle(n)"
        :class="{ clickable: hasContent(n) }"
      >
        <span class="tn-arrow">{{ hasContent(n) ? (expanded[n.node_id] ? '▼' : '▶') : '·' }}</span>
        <span class="tn-title">
          <span class="tn-mark">{{ '#'.repeat(Math.min(n.level + 1, 6)) }}</span>
          {{ n.title || '(未命名章节)' }}
        </span>
        <span class="tn-count">{{ n.subtree_chunk_count || 0 }} chunk</span>
      </div>
      <div v-if="n.summary" class="tn-summary" :style="{ paddingLeft: (10 + depth * 16 + 14) + 'px' }">
        摘要：{{ n.summary }}
      </div>
      <div v-if="expanded[n.node_id]">
        <template v-if="n.chunks && n.chunks.length">
          <div
            v-for="c in n.chunks"
            :key="c.id"
            class="tn-chunk"
            :style="{ paddingLeft: (10 + depth * 16 + 10) + 'px' }"
          >
            <span class="tn-chunk-text">🔹 {{ c.text_preview || '' }}</span>
            <button class="tn-del" @click="delChunk(c)">✕</button>
          </div>
        </template>
        <TreeNode v-if="n.children && n.children.length" :nodes="n.children" :depth="depth + 1" />
      </div>
    </div>

    <!-- 叶子节点（段落/表格/图片） -->
    <div v-else class="tn-leaf" :style="{ paddingLeft: (10 + depth * 16) + 'px' }">
      <span class="tn-leaf-tag">[{{ TYPE_LABEL[n.type] || n.type }}]</span>
      <span class="tn-leaf-text">{{ (n.text_preview || '').substring(0, 80) }}</span>
    </div>
  </template>
</template>

<style scoped>
.tn-section {
  padding: 5px 10px;
  font-size: 12px;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 6px;
  border-top: 1px solid var(--border-light);
}
.tn-section.clickable {
  cursor: pointer;
}
.tn-section:hover {
  background: rgba(128, 128, 128, 0.06);
}
.tn-arrow {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--text-faint);
  width: 12px;
  text-align: center;
}
.tn-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tn-mark {
  color: var(--primary);
  font-weight: 600;
}
.tn-count {
  flex-shrink: 0;
  margin-left: 8px;
  color: var(--text-faint);
  font-size: 11px;
}
.tn-summary {
  padding: 2px 10px;
  font-size: 11px;
  color: var(--text-secondary);
  font-style: italic;
}
.tn-chunk {
  padding: 3px 10px;
  font-size: 11px;
  color: var(--text-muted);
  display: flex;
  align-items: flex-start;
  gap: 6px;
  border-top: 1px dashed var(--border-light);
}
.tn-chunk-text {
  flex: 1;
  word-break: break-all;
  line-height: 1.4;
}
.tn-del {
  flex-shrink: 0;
  background: none;
  border: none;
  color: var(--danger);
  cursor: pointer;
  font-size: 11px;
  padding: 2px 4px;
}
.tn-leaf {
  padding: 4px 10px;
  font-size: 11px;
  color: var(--text-secondary);
  display: flex;
  align-items: flex-start;
  gap: 6px;
  border-top: 1px solid var(--border-light);
}
.tn-leaf-tag {
  color: var(--text-faint);
  flex-shrink: 0;
}
.tn-leaf-text {
  flex: 1;
  word-break: break-all;
  line-height: 1.4;
}
</style>
