// 颜色工具：hex 解析 / 混色 / 调亮调暗 / 转 rgba，由调色板动态生成主题相关颜色

function clamp(v) {
  return Math.max(0, Math.min(255, Math.round(v)))
}

function pad(v) {
  return clamp(v).toString(16).padStart(2, '0')
}

export function hexToRgb(hex) {
  const h = String(hex).replace('#', '')
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  }
}

export function toHex(r, g, b) {
  return `#${pad(r)}${pad(g)}${pad(b)}`
}

// 将 hex 与白(amt>0)/黑(amt<0)混合，返回新的 hex
export function shade(hex, amt) {
  const { r, g, b } = hexToRgb(hex)
  const target = amt >= 0 ? 255 : 0
  const t = Math.abs(amt)
  return toHex(
    r + (target - r) * t,
    g + (target - g) * t,
    b + (target - b) * t,
  )
}

// 两个 hex 混色，ratio 为第二个颜色的权重（0~1）
export function mix(hex1, hex2, ratio) {
  const a = hexToRgb(hex1)
  const b = hexToRgb(hex2)
  return toHex(
    a.r + (b.r - a.r) * ratio,
    a.g + (b.g - a.g) * ratio,
    a.b + (b.b - a.b) * ratio,
  )
}

export function rgba(hex, alpha) {
  const { r, g, b } = hexToRgb(hex)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

// 由一条配色 { gradient, color1, color2 } 生成渐变（或单色）字符串
// strong=true 会整体加深，用于渐变强调版（如按钮 hover）
export function gradientOf(p, from = 135, strong = false) {
  const c1 = p.color1
  const c2 = p.gradient ? p.color2 : c1
  if (!p.gradient) return strong ? shade(c1, -0.12) : c1
  const s1 = strong ? shade(c1, -0.12) : c1
  const s2 = strong ? shade(c2, -0.12) : c2
  return `linear-gradient(${from}deg, ${s1} 0%, ${s2} 100%)`
}