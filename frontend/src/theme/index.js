// Naive UI 主题覆盖：主要色由调色板（亮/暗各自的单色或渐变）动态生成

import { shade } from './palette'

export function buildOverrides(palette, isDark) {
  const primary = palette.color1
  const hover = shade(primary, -0.15)
  const pressed = shade(primary, -0.25)
  return {
    common: {
      primaryColor: primary,
      primaryColorHover: hover,
      primaryColorPressed: pressed,
      primaryColorSuppl: primary,
      infoColor: primary,
      infoColorHover: hover,
      successColor: isDark ? '#34d399' : '#059669',
      warningColor: isDark ? '#fbbf24' : '#d97706',
      errorColor: isDark ? '#f87171' : '#dc2626',
      borderRadius: '10px',
      borderRadiusSmall: '8px',
    },
  }
}