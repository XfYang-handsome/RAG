import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'chat', component: () => import('../views/ChatView.vue') },
  { path: '/mcp', name: 'mcp', component: () => import('../views/McpView.vue') },
]

// hash 路由：后端无需 SPA fallback，刷新任意页面不 404，便于与旧版双轨共存
const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

export default router
