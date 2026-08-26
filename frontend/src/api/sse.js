// SSE 流式读取：POST /chat，逐行解析 `data: {type, content}`，通过 onEvent 分发
//
// 事件类型（与后端 _sse 一致）：
//   token      正文 token
//   reasoning  深度思考内容
//   status     状态消息（检索中/评估中…）
//   warning    警告
//   citations  引用来源（content 为 JSON 数组字符串）
//   agent_trace Agent 决策轨迹（content 为 JSON 对象字符串）
//   error      错误
//   done       结束

export async function streamChat(body, onEvent) {
  let res
  try {
    res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
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
      /* 非 JSON */
    }
    throw new Error(detail)
  }

  if (!res.body) {
    throw new Error('浏览器不支持流式响应')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const data = JSON.parse(line.slice(6))
        onEvent(data)
      } catch (e) {
        /* 忽略无法解析的行 */
      }
    }
  }
}
