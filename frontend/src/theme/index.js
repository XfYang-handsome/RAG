// Naive UI 主题覆盖：对齐旧版 RAG 前端的配色（蓝紫主色）
// 亮/暗共用主色，差异主要靠 Naive UI 内置 darkTheme + 全局 CSS 变量控制

const common = {
  primaryColor: '#6366f1',
  primaryColorHover: '#4f46e5',
  primaryColorPressed: '#4338ca',
  primaryColorSuppl: '#6366f1',
  infoColor: '#6366f1',
  infoColorHover: '#4f46e5',
  successColor: '#059669',
  warningColor: '#d97706',
  errorColor: '#dc2626',
  borderRadius: '10px',
  borderRadiusSmall: '8px',
}

export const lightOverrides = { common }

export const darkOverrides = {
  common: {
    ...common,
    successColor: '#34d399',
    warningColor: '#fbbf24',
    errorColor: '#f87171',
  },
}
