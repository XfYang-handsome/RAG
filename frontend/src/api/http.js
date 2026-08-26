// 统一 fetch 封装：JSON 请求/响应，FastAPI 错误（detail 字段）转成 Error

async function request(url, options = {}) {
  let res
  try {
    res = await fetch(url, options)
  } catch (e) {
    throw new Error(`网络请求失败: ${e.message}`)
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const data = await res.json()
      if (data && data.detail) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
      }
    } catch (e) {
      /* 非 JSON 错误体，保留默认 */
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }

  // 204 / 空响应体保护
  const text = await res.text()
  if (!text) return {}
  try {
    return JSON.parse(text)
  } catch (e) {
    return text
  }
}

export const http = {
  get: (url) => request(url),
  post: (url, body = {}) =>
    request(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  put: (url, body = {}) =>
    request(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  del: (url) => request(url, { method: 'DELETE' }),
  // multipart 上传：FormData 由 fetch 自动设置 boundary，勿手动设 Content-Type
  postForm: (url, formData) => request(url, { method: 'POST', body: formData }),
}
