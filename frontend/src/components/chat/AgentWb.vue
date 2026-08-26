<script setup>
import { computed } from 'vue'

const props = defineProps({
  trace: { type: Array, default: () => [] },
})

const ACTION_CN = {
  SEARCH: '检索',
  REFINE_QUERY: '改写查询',
  READ_PARENT: '读父块',
  READ_SECTION: '读章节',
  WEB_SEARCH: '联网搜索',
  ANSWER: '合成答案',
}

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

const last = computed(() => (props.trace.length ? props.trace[props.trace.length - 1] : null))

const reqChips = computed(() => {
  if (!last.value || !last.value.requirements) return []
  return last.value.requirements.map((r) => {
    const cls = r.status === 'SUPPORTED' ? 'wb-supported' : r.status === 'PARTIAL' ? 'wb-partial' : 'wb-missing'
    const icon = r.status === 'SUPPORTED' ? '✓' : r.status === 'PARTIAL' ? '△' : '✗'
    return { cls, icon, id: r.id, description: r.description }
  })
})

const rows = computed(() =>
  props.trace.map((t) => {
    const a = t.action
    let act
    if (a) {
      const cn = ACTION_CN[a.type] || a.type
      const q = a.query ? ` "${String(a.query).slice(0, 44)}"` : ''
      act = `${cn}${esc(q)}`
    } else {
      act = '（等待决策）'
    }
    const stop = t.stop_reason ? ` · ${t.stop_reason}` : ''
    return { iteration: t.iteration, act, stop }
  }),
)

const coverage = computed(() => {
  if (!last.value || last.value.coverage == null) return '—'
  return Math.round((last.value.coverage || 0) * 100) + '%'
})
</script>

<template>
  <div class="agent-wb">
    <div class="wb-head">
      <span>Agent 工作台</span>
      <span class="wb-meta">迭代 {{ last?.iteration }} · 证据 {{ last?.evidences }} · 覆盖 {{ coverage }}</span>
    </div>
    <div class="wb-reqs">
      <span
        v-for="(c, i) in reqChips"
        :key="i"
        class="wb-status"
        :class="c.cls"
        :title="c.description"
      >{{ c.icon }} {{ c.id }}</span>
    </div>
    <div class="wb-traj">
      <div v-for="(r, i) in rows" :key="i" class="wb-row">
        <span class="wb-iter">#{{ r.iteration }}</span><span v-html="r.act"></span><span class="wb-stop">{{ r.stop }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.agent-wb {
  margin: 8px 0;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  background: var(--panel-sub-bg);
  overflow: hidden;
  font-size: 12px;
}
.wb-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-bottom: 1px dashed var(--border-light);
  font-weight: 600;
  color: var(--text);
}
.wb-meta {
  font-weight: 400;
  font-size: 11px;
  color: var(--text-faint);
}
.wb-reqs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px 12px;
}
.wb-status {
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  line-height: 1.4;
  border: 1px solid transparent;
}
.wb-supported { color: #18a058; background: rgba(24, 160, 88, 0.1); border-color: rgba(24, 160, 88, 0.3); }
.wb-partial { color: #d88a00; background: rgba(216, 138, 0, 0.1); border-color: rgba(216, 138, 0, 0.3); }
.wb-missing { color: #d03050; background: rgba(208, 48, 80, 0.1); border-color: rgba(208, 48, 80, 0.3); }
.wb-traj {
  padding: 6px 12px 8px;
  border-top: 1px solid var(--border-light);
  max-height: 160px;
  overflow-y: auto;
}
.wb-row {
  padding: 2px 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.5;
}
.wb-iter {
  color: var(--primary);
  font-weight: 600;
  margin-right: 6px;
}
.wb-stop {
  color: var(--text-faint);
  font-style: italic;
}
</style>
