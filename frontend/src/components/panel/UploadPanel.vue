<script setup>
import { ref, computed } from 'vue'
import {
  Upload, FileText, Presentation, Table, FileImage, FileCode, File,
  X, RotateCcw, Trash2, Wand2,
} from '@lucide/vue'
import { useUploadStore } from '../../stores/upload'

const upload = useUploadStore()
const fileInput = ref(null)
const dragging = ref(false)

const fileIconMap = {
  pdf: { icon: FileText, cls: 'pdf' },
  doc: { icon: FileText, cls: 'doc' },
  docx: { icon: FileText, cls: 'doc' },
  ppt: { icon: Presentation, cls: 'ppt' },
  pptx: { icon: Presentation, cls: 'ppt' },
  xls: { icon: Table, cls: 'xls' },
  xlsx: { icon: Table, cls: 'xls' },
  csv: { icon: Table, cls: 'xls' },
  img: { icon: FileImage, cls: 'img' },
  code: { icon: FileCode, cls: 'code' },
  txt: { icon: FileText, cls: 'txt' },
  md: { icon: FileText, cls: 'txt' },
  other: { icon: File, cls: 'other' },
}

const CODE_EXT = ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'c', 'cpp', 'go', 'rs', 'html', 'css', 'json', 'xml', 'yaml', 'yml']
const IMG_EXT = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg']

function fileMeta(name) {
  const ext = ((name || '').split('.').pop() || '').toLowerCase()
  if (ext === 'pdf') return fileIconMap.pdf
  if (['doc', 'docx'].includes(ext)) return fileIconMap.doc
  if (['ppt', 'pptx'].includes(ext)) return fileIconMap.ppt
  if (['xls', 'xlsx', 'csv'].includes(ext)) return fileIconMap.xls
  if (IMG_EXT.includes(ext)) return fileIconMap.img
  if (CODE_EXT.includes(ext)) return fileIconMap.code
  if (['md', 'txt', 'rtf'].includes(ext)) return fileIconMap.txt
  return fileIconMap.other
}

function pickFiles() {
  fileInput.value.click()
}
function onFiles(e) {
  upload.addFiles(e.target.files)
  e.target.value = ''
}
function onDrop(e) {
  dragging.value = false
  upload.addFiles(e.dataTransfer.files)
}
function onDragOver(e) {
  e.preventDefault()
  dragging.value = true
}
function onDragLeave() {
  dragging.value = false
}

const canUpload = computed(() => upload.pendingFiles.length > 0 && !upload.uploading)

const TASK_STATUS = {
  DONE: '完成',
  FAILED: '失败',
  PARSING: '解析中',
  CHUNKING: '切分中',
  EMBEDDING: '向量化',
  INDEXING: '写入索引',
  PENDING: '排队中',
}

function taskStatusText(s) {
  return TASK_STATUS[s] || s || '未知'
}
</script>

<template>
  <div class="upload-panel">
    <div class="panel-head">
      <div class="panel-title"><Upload :size="15" /> 文件上传</div>
      <label class="enhance-toggle" title="增强解析：用 deepdoc 做布局/表格/结构识别（更慢但更准）">
        <input v-model="upload.enhance" type="checkbox" />
        <Wand2 :size="13" />
        <span>增强</span>
      </label>
    </div>

    <div
      class="drop-zone"
      :class="{ dragging }"
      @click="pickFiles"
      @drop.prevent="onDrop"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
    >
      <Upload :size="22" class="dz-icon" />
      <p>点击选择文件，或拖拽到此处</p>
    </div>
    <input ref="fileInput" type="file" multiple hidden @change="onFiles" />

    <!-- 待上传列表 -->
    <div v-if="upload.pendingFiles.length" class="file-list">
      <div v-for="f in upload.pendingFiles" :key="f.id" class="file-item">
        <component :is="fileMeta(f.name).icon" :size="16" class="f-icon" :class="fileMeta(f.name).cls" />
        <div class="file-main">
          <div class="file-row">
            <span class="file-name">{{ f.name }}</span>
            <span class="file-status" :class="f.state">{{ f.status }}</span>
          </div>
          <div class="progress" v-if="f.state !== 'done' && f.state !== 'fail'">
            <div
              class="progress-fill"
              :class="{ indeterminate: f.state === 'indeterminate' }"
              :style="{ width: f.progress + '%' }"
            ></div>
          </div>
        </div>
        <button class="mini-btn danger" title="移除" @click="upload.removeFile(f.id)"><X :size="13" /></button>
      </div>
    </div>

    <button class="upload-btn" :disabled="!canUpload" @click="upload.uploadAll()">
      {{ upload.uploading ? '上传中...' : `开始上传（${upload.pendingFiles.length}）` }}
    </button>

    <!-- 历史任务 -->
    <div class="history-head">
      <span>历史任务</span>
      <button class="mini-btn" title="刷新" @click="upload.loadHistory()"><RotateCcw :size="12" /></button>
    </div>
    <div class="history-list">
      <div v-if="!upload.uploadTasks.length" class="history-empty">暂无历史任务</div>
      <div v-for="t in upload.uploadTasks" :key="t.task_id" class="history-item">
        <span class="h-name">{{ t.file_name }}</span>
        <span class="h-status" :class="t.status">{{ taskStatusText(t.status) }}</span>
        <span v-if="t.status === 'FAILED'" class="h-ops">
          <button class="mini-btn" title="重试" @click="upload.retryTask(t.task_id)"><RotateCcw :size="12" /></button>
          <button class="mini-btn danger" title="删除" @click="upload.deleteTask(t.task_id)"><Trash2 :size="12" /></button>
        </span>
        <span v-else-if="t.status === 'DONE'" class="h-ops">
          <button class="mini-btn danger" title="删除" @click="upload.deleteTask(t.task_id)"><Trash2 :size="12" /></button>
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.upload-panel {
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
.enhance-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary);
  cursor: pointer;
  user-select: none;
}
.enhance-toggle input {
  accent-color: var(--primary);
  cursor: pointer;
}
.drop-zone {
  margin: 12px 16px;
  border: 2px dashed var(--border-strong);
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  color: var(--text-faint);
  font-size: 12px;
  transition: all 0.2s;
}
.drop-zone:hover,
.drop-zone.dragging {
  border-color: var(--primary);
  background: var(--primary-soft);
  color: var(--primary);
}
.dz-icon {
  margin-bottom: 6px;
}
.drop-zone p {
  margin: 0;
}
.file-list {
  margin: 0 16px 10px;
  max-height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.file-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel-sub-bg);
}
.f-icon {
  flex-shrink: 0;
}
.f-icon.pdf { color: #dc2626; }
.f-icon.doc { color: #2563eb; }
.f-icon.ppt { color: #ea580c; }
.f-icon.xls { color: #059669; }
.f-icon.img { color: #8b5cf6; }
.f-icon.code { color: #0ea5e9; }
.f-icon.txt { color: #64748b; }
.f-icon.other { color: var(--primary); }
.file-main { flex: 1; min-width: 0; }
.file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.file-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  color: var(--text);
}
.file-status {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-faint);
}
.file-status.done { color: var(--success); }
.file-status.fail { color: var(--danger); }
.progress {
  margin-top: 5px;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--grad-primary);
  border-radius: 2px;
  transition: width 0.4s ease;
}
.progress-fill.indeterminate {
  position: relative;
  overflow: hidden;
}
.progress-fill.indeterminate::after {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  width: 42%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
  animation: shimmer 1.4s infinite;
}
@keyframes shimmer {
  from { left: -42%; }
  to { left: 100%; }
}
.upload-btn {
  margin: 0 16px 12px;
  padding: 9px;
  border: none;
  border-radius: 8px;
  background: var(--grad-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.28);
  transition: transform 0.15s, box-shadow 0.2s;
}
.upload-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
.upload-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px 6px;
  border-top: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.history-list {
  max-height: 200px;
  overflow-y: auto;
  padding: 0 16px 12px;
}
.history-empty {
  padding: 16px;
  text-align: center;
  font-size: 12px;
  color: var(--text-faint);
}
.history-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border-light);
  font-size: 12px;
}
.h-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-secondary);
}
.h-status {
  flex-shrink: 0;
  font-size: 11px;
  color: var(--text-faint);
}
.h-status.DONE { color: var(--success); }
.h-status.FAILED { color: var(--danger); }
.h-status.PARSING,
.h-status.CHUNKING,
.h-status.EMBEDDING,
.h-status.INDEXING,
.h-status.PENDING { color: var(--primary); }
.h-ops { display: flex; gap: 2px; flex-shrink: 0; }
.mini-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 2px 4px;
  display: flex;
  align-items: center;
}
.mini-btn:hover { color: var(--primary); }
.mini-btn.danger:hover { color: var(--danger); }
</style>
