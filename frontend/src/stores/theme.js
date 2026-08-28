import { defineStore } from 'pinia'
import { api } from '../api'
import { shade, rgba, gradientOf } from '../theme/palette'

const KEY = 'rag-theme'
const ORDER = ['system', 'light', 'dark']
const LABEL = { system: '跟随系统', light: '亮色', dark: '暗色' }

const DEFAULT_PALETTE = {
  light: { gradient: true, color1: '#0ea5e9', color2: '#06b6d4' },
  dark: { gradient: true, color1: '#6366f1', color2: '#a855f7' },
}

const HEX_RE = /^#[0-9a-fA-F]{6}$/

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

function normalizePalette(p, fallback) {
  if (!p || typeof p !== 'object' || !p.color1 || !HEX_RE.test(p.color1)) {
    return { ...fallback }
  }
  const gradient = !!p.gradient
  let color2 = p.color2
  if (!color2 || !HEX_RE.test(color2)) {
    color2 = gradient ? fallback.color2 : p.color1
  }
  return { gradient, color1: p.color1.toLowerCase(), color2: color2.toLowerCase() }
}

export const useThemeStore = defineStore('theme', {
  state: () => ({
    mode: readStored(),
    // 系统偏好的响应式快照：让 isDark 在 system 模式下也能响应系统切换
    systemDark: systemPrefersDark(),
    // 亮/暗各自的配色（单色或双色渐变），默认值兜底，运行时从后端 config 加载
    palette: {
      light: { ...DEFAULT_PALETTE.light },
      dark: { ...DEFAULT_PALETTE.dark },
    },
    loaded: false,
  }),
  getters: {
    label: (s) => LABEL[s.mode] || LABEL.system,
    // 实际是否暗色（system 模式跟随系统偏好）
    isDark: (s) => s.mode === 'dark' || (s.mode === 'system' && s.systemDark),
    // 当前实际生效的配色对象
    activePalette: (s) => {
      const dark = s.mode === 'dark' || (s.mode === 'system' && s.systemDark)
      return dark ? s.palette.dark : s.palette.light
    },
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
      this.applyPalette()
    },
    // 根据当前配色运行时覆盖 CSS 变量（覆盖 global.css 的静态默认值）
    applyPalette() {
      const root = document.documentElement
      const p = this.activePalette
      if (!p || !p.color1) return
      const c1 = p.color1
      const c2 = p.gradient ? p.color2 : c1
      root.style.setProperty('--primary', c1)
      root.style.setProperty('--primary-hover', shade(c1, -0.15))
      root.style.setProperty('--primary-disabled', shade(c1, 0.35))
      root.style.setProperty('--primary-soft', rgba(c1, this.isDark ? 0.22 : 0.12))
      root.style.setProperty('--grad-primary', gradientOf(p, 135, false))
      root.style.setProperty('--grad-primary-strong', gradientOf(p, 135, true))
      root.style.setProperty('--card-hover', `0 10px 30px ${rgba(c1, this.isDark ? 0.28 : 0.14)}`)
      // 顶栏：主色加深后的渐变，保证白色标题/按钮的可读性
      const hc1 = shade(c1, -0.55)
      const hc2 = shade(c2, -0.42)
      const hc3 = shade(c1, -0.18)
      root.style.setProperty('--header-grad', `linear-gradient(135deg, ${hc1} 0%, ${hc2} 55%, ${hc3} 130%)`)
      root.style.setProperty('--header-glow-a', rgba(c1, 0.30))
      root.style.setProperty('--header-glow-b', rgba(c2, 0.24))
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
      this.loadPalette()
    },
    async loadPalette() {
      try {
        const d = await api.getTheme()
        this.palette = {
          light: normalizePalette(d.light, DEFAULT_PALETTE.light),
          dark: normalizePalette(d.dark, DEFAULT_PALETTE.dark),
        }
        this.loaded = true
        this.apply()
      } catch (e) {
        this.loaded = false
        this.apply()
      }
    },
    async savePalette(theme) {
      await api.saveTheme(theme)
      this.palette = {
        light: normalizePalette(theme.light, DEFAULT_PALETTE.light),
        dark: normalizePalette(theme.dark, DEFAULT_PALETTE.dark),
      }
      this.apply()
    },
    resetPalette() {
      this.palette = {
        light: { ...DEFAULT_PALETTE.light },
        dark: { ...DEFAULT_PALETTE.dark },
      }
      this.apply()
    },
  },
})