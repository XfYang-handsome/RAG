<script setup>
defineProps({
  citations: { type: Array, default: () => [] },
})

function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function locOf(c) {
  const src = c.source || {}
  if (src.origin === 'web') {
    return src.url ? `联网搜索 · ${src.url}` : '联网搜索'
  }
  const title = src.doc_title || src.doc_id || ''
  const secs = (src.section_titles && src.section_titles.length) ? src.section_titles : []
  if (secs.length) {
    return { doc: title, sec: secs.join(' / ') }
  }
  if (title) {
    return { doc: title, sec: src.section_path || '' }
  }
  return { doc: '未知来源', sec: '' }
}
</script>

<template>
  <div class="cite-box">
    <div class="cite-title">引用来源</div>
    <div v-for="c in citations" :key="c.num" class="cite-item">
      <span class="cite-num">{{ c.num }}</span>
      <span>
        <span class="cite-doc">{{ locOf(c).doc }}</span>
        <span v-if="locOf(c).sec" class="cite-sec">{{ locOf(c).sec }}</span>
      </span>
    </div>
  </div>
</template>

<style scoped>
.cite-box {
  margin-top: 10px;
  border-top: 1px solid var(--border-light);
  padding-top: 10px;
}
.cite-title {
  font-size: 11px;
  font-weight: 700;
  color: var(--text-faint);
  letter-spacing: 0.4px;
  margin-bottom: 6px;
  border-left: 3px solid var(--primary);
  padding-left: 8px;
}
.cite-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--panel-sub-bg);
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--text-secondary);
}
.cite-item:hover {
  border-color: var(--primary);
}
.cite-num {
  flex-shrink: 0;
  width: 18px;
  height: 18px;
  border-radius: 5px;
  background: var(--primary-soft);
  color: var(--primary);
  font-size: 11px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 1px;
}
.cite-doc {
  color: var(--text);
  font-weight: 600;
}
.cite-sec {
  color: var(--text-faint);
}
.cite-sec::before {
  content: ' · ';
  color: var(--text-faint);
}
</style>
