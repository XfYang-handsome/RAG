import { defineStore } from 'pinia'

const KEY = 'rag-theme'
const ORDER = ['system', 'light', 'dark']
const LABEL = { system: '跟随系统', light: '亮色', dark: '暗色' }

function systemPrefersDark() {
  return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
}

function readStored() {
  try {
    const v = localStorage.getItem(KEY)
    return ORDER.includes(v) ? v : 'system'
  } catch (e) {
    return 'system'
  }
}

export const useThemeStore = defineStore('theme', {
  state: () => ({
    mode: readStored(),
    // 系统偏好的响应式快照：让 isDark 在 system 模式下也能响应系统切换
    systemDark: systemPrefersDark(),
  }),
  getters: {
    label: (s) => LABEL[s.mode] || LABEL.system,
    // 实际是否暗色（system 模式跟随系统偏好）
    isDark: (s) => s.mode === 'dark' || (s.mode === 'system' && s.systemDark),
  },
  actions: {
    cycle() {
      const idx = ORDER.indexOf(this.mode)
      this.setMode(ORDER[(idx + 1) % ORDER.length])
    },
    setMode(m) {
      if (!ORDER.includes(m)) return
      this.mode = m
      try { localStorage.setItem(KEY, m) } catch (e) {}
      this.apply()
    },
    apply() {
      const root = document.documentElement
      if (this.mode === 'system') {
        root.removeAttribute('data-theme')
      } else {
        root.setAttribute('data-theme', this.mode)
      }
    },
    init() {
      this.apply()
      // system 模式下监听系统偏好变化：同时刷新 CSS 变量(data-theme) 与 Naive UI(isDark)
      if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
          this.systemDark = e.matches
          if (this.mode === 'system') this.apply()
        })
      }
    },
  },
})
