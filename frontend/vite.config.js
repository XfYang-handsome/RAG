import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端 FastAPI 服务地址（开发期 proxy 目标）
const backend = 'http://127.0.0.1:8000'

// 所有后端 API 前缀都代理到 FastAPI；SSE(/chat) 走 changeOrigin 透传
const apiPrefixes = [
  '/chat', '/conversations', '/upload', '/config', '/models',
  '/dbs', '/local', '/mcp', '/logs', '/health', '/reranker',
  '/browse', '/pick-directory',
]

export default defineConfig({
  plugins: [vue()],
  base: './', // 相对路径，便于 FastAPI 在任意子路径下服务静态资源
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      apiPrefixes.map((p) => [p, { target: backend, changeOrigin: true }])
    ),
  },
})
