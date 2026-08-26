import { defineStore } from 'pinia'
import { api } from '../api'

const STAGE_TEXT = {
  PENDING: '排队中...',
  PARSING: '解析中...',
  CHUNKING: '切分中...',
  EMBEDDING: '向量化中...',
  INDEXING: '写入索引...',
  DONE: '完成',
  FAILED: '失败',
}

const STAGE_BASE = { PENDING: 3, PARSING: 15, CHUNKING: 38, EMBEDDING: 42, INDEXING: 90 }

let fileSeq = 0

export const useUploadStore = defineStore('upload', {
  state: () => ({
    pendingFiles: [], // {id, name, status, progress, file}
    uploadTasks: [], // 历史任务
    enhance: true,
    uploading: false,
  }),

  actions: {
    addFiles(fileList) {
      for (const f of fileList) {
        if (!f) continue
        if (this.pendingFiles.some((p) => p.name === f.name)) continue // 去重
        this.pendingFiles.push({
          id: ++fileSeq,
          name: f.name,
          status: '等待上传',
          progress: 0,
          state: null, // 'indeterminate' | 'done' | 'fail' | null
          file: f,
        })
      }
    },

    removeFile(id) {
      this.pendingFiles = this.pendingFiles.filter((p) => p.id !== id)
    },

    async uploadAll() {
      if (this.uploading || !this.pendingFiles.length) return
      this.uploading = true
      let ok = 0
      let fail = 0

      for (const item of this.pendingFiles) {
        const fd = new FormData()
        fd.append('file', item.file)
        fd.append('enhance', this.enhance ? 'true' : 'false')
        item.status = '提交中...'
        item.state = 'indeterminate'
        item.progress = 3
        try {
          const d = await api.upload(fd)
          if (d.success && d.task_id) {
            item.status = '已提交，等待处理...'
            item.taskId = d.task_id
            const success = await this.pollTask(item)
            if (success) ok++
            else fail++
          } else {
            item.status = '✗ ' + (d.message || '提交失败')
            item.state = 'fail'
            item.progress = 100
            fail++
          }
        } catch (e) {
          item.status = '✗ ' + e.message
          item.state = 'fail'
          item.progress = 100
          fail++
        }
      }

      this.uploading = false
      this.loadHistory()
      return { ok, fail }
    },

    async pollTask(item) {
      // 轮询直到 DONE / FAILED
      while (true) {
        try {
          const t = await api.getUploadStatus(item.taskId)
          if (t.status === 'DONE') {
            const cnt = (t.stats && t.stats.inserted_count) ?? 0
            item.status = `✓ ${cnt}条`
            item.state = 'done'
            item.progress = 100
            return true
          }
          if (t.status === 'FAILED') {
            item.status = '✗ ' + (t.error || '失败')
            item.state = 'fail'
            item.progress = 100
            return false
          }
          if (t.status === 'EMBEDDING' && (t.progress || 0) > 0) {
            item.status = `向量化 ${t.progress}%`
            item.progress = 42 + t.progress * 0.46
            item.state = null
          } else {
            item.status = STAGE_TEXT[t.status] || t.status
            item.progress = STAGE_BASE[t.status] || 10
            item.state = 'indeterminate'
          }
        } catch (e) {
          item.status = '✗ 状态查询失败'
          item.state = 'fail'
          item.progress = 100
          return false
        }
        await new Promise((res) => setTimeout(res, 1500))
      }
    },

    async loadHistory() {
      try {
        const d = await api.listUploadTasks()
        const now = Date.now()
        const doneTtl = 10 * 60 * 1000
        this.uploadTasks = (d.tasks || [])
          .filter((t) => !(t.status === 'DONE' && now - (t.created_at || 0) > doneTtl))
          .slice(0, 20)
      } catch (e) {
        this.uploadTasks = []
      }
    },

    async retryTask(taskId) {
      await api.retryUpload(taskId)
      this.loadHistory()
    },

    async deleteTask(taskId) {
      await api.deleteUpload(taskId)
      this.loadHistory()
    },
  },
})
