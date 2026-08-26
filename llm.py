"""
================================================================================
LLM 语言模型封装
================================================================================

支持两种模型后端：
  ┌─────────────────────────────────────────────────────────────────┐
  │ online=True  → ChatOpenAI（OpenAI 兼容 API，标准 Chat Completions）│
  │                流式输出基于 LangChain 的 .stream() 方法             │
  ├─────────────────────────────────────────────────────────────────┤
  │ online=False → DoubaoLLM（豆包 seed-evolving，火山方舟 API）       │
  │                使用 /api/v3/responses API（非标准格式）             │
  │                手动 SSE 流式解析                                   │
  └─────────────────────────────────────────────────────────────────┘

豆包 Responses API 特殊处理：
  豆包使用火山方舟的 Responses API（/api/v3/responses），而非标准 Chat Completions。
  SSE 事件类型：
    - response.reasoning_summary_text.delta  → 深度思考过程（流式输出，is_reasoning=True）
    - response.output_text.delta            → 最终输出文本（流式输出，is_reasoning=False）
    - response.completed                    → 流结束

  消息格式转换：
    豆包 API 的 input 格式与 OpenAI 不同：
    - 无独立 system role → system prompt 拼到第一条 user 消息前
    - user 消息需包装为 content: [{type: "input_text", text: "..."}]
    - assistant 消息直接为 content: "..." 字符串

依赖：
  - langchain-openai: 在线模式的标准流式调用
  - requests: 豆包模式的手动 SSE 解析
================================================================================
"""

import json
import os
from typing import Iterator, Optional

import httpx
import requests

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config_loader import config

# ============================================================================
# 禁用系统代理干扰
# ============================================================================
# Windows 上若配置了系统代理（Clash/V2Ray 等），httpx/requests 默认 trust_env=True
# 会把发往内网/外网 API 的请求也转发到代理，导致 Connection error / SSL EOF。
# 统一清除代理环境变量，确保 LLM 直连目标 API。
for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_proxy_var, None)


def _make_http_client() -> httpx.Client:
    """创建一个禁用系统代理的 httpx 客户端（trust_env=False 绕过 WinHTTP 注册表代理）。"""
    return httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0))


def _make_requests_session() -> requests.Session:
    """创建一个禁用系统代理的 requests 会话。"""
    s = requests.Session()
    s.trust_env = False
    return s

# ============================================================================
# 默认参数（具体模型配置从 models.json 读取）
# ============================================================================
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0        # 决策/评估/摘要等确定性任务（默认 0）
DEFAULT_ANSWER_TEMPERATURE = 0.7  # 生成答案（需要创造性/表达多样性）


# ============================================================================
# DoubaoStreamChunk — 豆包流式输出的兼容包装
# ============================================================================

class DoubaoStreamChunk:
    """
    模拟 LangChain AIMessageChunk 的简单包装。

    提供 .content 属性（使豆包流式输出与 ChatOpenAI.stream() 返回的
    chunk 接口一致），并额外提供 .is_reasoning 标记区分「深度思考内容」
    与「最终回答内容」。

    属性：
      content      : 文本片段（思考内容或正文内容）
      is_reasoning : True=深度思考过程，False=最终输出文本
    """
    def __init__(self, content: str, is_reasoning: bool = False):
        self.content = content
        self.is_reasoning = is_reasoning


# ============================================================================
# DoubaoLLM — 豆包模型（火山方舟 Responses API）
# ============================================================================

class DoubaoLLM:
    """
    豆包 doubao-seed-evolving 模型包装器。

    使用火山方舟 /api/v3/responses API（Responses API），
    通过 requests + SSE 流式解析实现。

    为什么不用 LangChain 的 ChatOpenAI？
      豆包的 Responses API 端点路径和请求/响应格式都与 OpenAI 不兼容，
      无法通过简单的 base_url 切换来适配。因此需要手动实现 SSE 解析。

    兼容 __main__.py 的 stream(messages) 调用方式。
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        self.model       = model
        self.base_url    = (base_url or "").rstrip("/")
        self.api_key     = api_key
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._requests   = _make_requests_session()  # 禁用系统代理

    # ------------------------------------------------------------------
    # 流式调用
    # ------------------------------------------------------------------
    def stream(self, messages: list) -> Iterator[DoubaoStreamChunk]:
        """
        流式调用豆包 Responses API。

        流程：
          1. 将 LangChain 消息列表转换为豆包 input 格式
          2. POST /api/v3/responses（stream=True）
          3. 逐行解析 SSE 事件，仅提取 output_text.delta
          4. 跳过 reasoning（深度思考）事件

        Args:
            messages: LangChain 消息列表 [SystemMessage, HumanMessage, AIMessage, ...]

        Yields:
            DoubaoStreamChunk（.content 属性为文本片段）
        """
        # ---- Step 1: 转换消息格式 ----
        input_content = self._convert_messages(messages)

        # ---- Step 2: 发送请求 ----
        payload = {
            "model": self.model,
            "input": input_content,
            "stream": True,
        }
        url     = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        response = self._requests.post(
            url, json=payload, headers=headers, stream=True, timeout=120,
        )
        response.raise_for_status()

        # ---- Step 3: 解析 SSE 流 ----
        # 用 iter_lines()（默认 decode_unicode=None → 返回 bytes 行）替代手写逐字节
        # buffer 拼接：
        #   1. iter_lines 按 \n 切分「完整字节行」，UTF-8 多字节字符不会跨行断裂
        #      （0x0A 不会出现在多字节序列中），且不做 decode_unicode 避免跨 chunk 断裂；
        #   2. iter_lines 在流结束时会把 buffer 里残留的「无换行最后一帧」也 yield
        #      出来——修复手写版「while b"\n" in buffer 循环后尾部残留被丢弃」的隐患。
        # 逐行 decode 用 errors="ignore" 容错：单个坏字节不拖垮整个流。
        for line in response.iter_lines(chunk_size=1024):
            if not line:
                continue
            line = line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # SSE 格式: "data: {...}" 或 "event: xxx"
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    event_type = data.get("type", "")

                    # 深度思考过程（流式输出给前端展示）
                    if event_type == "response.reasoning_summary_text.delta":
                        delta = data.get("delta", "")
                        if delta:
                            yield DoubaoStreamChunk(delta, is_reasoning=True)
                    # 最终输出文本的 delta
                    elif event_type in (
                        "response.output_text.delta",
                        "response.output_text.done",
                    ):
                        delta = data.get("delta", "")
                        if delta:
                            yield DoubaoStreamChunk(delta, is_reasoning=False)
                except json.JSONDecodeError:
                    continue  # 忽略解析失败的行

    # ------------------------------------------------------------------
    # 消息格式转换
    # ------------------------------------------------------------------
    def _convert_messages(self, messages: list) -> list:
        """
        将 LangChain 消息列表转换为豆包 Responses API 的 input 格式。

        转换规则：
          SystemMessage  → 拼接到第一条 HumanMessage 前面
          HumanMessage   → {"role":"user", "content":[{"type":"input_text","text":"..."}]}
          AIMessage      → {"role":"assistant", "content":"..."}

        注意：
          - 豆包 API 无独立 system role，需要合并到 user 消息
          - user 消息的 content 是数组格式 [{type, text}]
          - assistant 消息的 content 是纯字符串
        """
        input_content = []
        system_text   = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # 系统提示词先暂存，后续拼到第一条 user 消息
                system_text = msg.content

            elif isinstance(msg, HumanMessage):
                # 把 system prompt 拼到第一条 user 消息前面
                text = (system_text + "\n\n" + msg.content) if system_text else msg.content
                input_content.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                })
                system_text = ""  # 只拼一次

            elif isinstance(msg, AIMessage):
                # 助手消息直接传字符串
                input_content.append({
                    "role":    "assistant",
                    "content": msg.content,
                })

        # 兜底：如果只有 system prompt 没有 user 消息（极少见）
        if system_text and not input_content:
            input_content.append({
                "role": "user",
                "content": [{"type": "input_text", "text": system_text}],
            })

        return input_content


# ============================================================================
# 工厂函数
# ============================================================================

def create_chat_model(
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    temperature: float = 0.0,
    protocol: str = "openai",
    disable_thinking: bool = False,
):
    """
    创建聊天模型实例。

    根据 protocol 参数自动选择后端：
      - "openai" → ChatOpenAI（OpenAI 兼容 API，流式输出）
      - "doubao" → DoubaoLLM（豆包 Responses API，SSE 流式）

    Args:
        model:      模型名称
        base_url:   API 地址
        api_key:    API 密钥
        temperature: 温度参数（0=确定性输出）
        protocol:   API 协议类型（openai / doubao）

    Returns:
        ChatOpenAI 或 DoubaoLLM 实例
    """
    # 规范化 base_url：用户可能误填了 /chat/completions 等端点后缀，
    # 而 ChatOpenAI 会自动拼接 /chat/completions，导致 404 Not Found。
    if base_url:
        base_url = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break

    if protocol == "doubao":
        # --- 豆包 Responses API（不支持 function calling）---
        return DoubaoLLM(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
    else:
        # --- 标准 OpenAI 兼容 API（默认） ---
        kwargs = dict(
            model=model,
            base_url=base_url,
            api_key=api_key,
            streaming=True,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature if temperature != 0.0 else DEFAULT_TEMPERATURE,
            http_client=_make_http_client(),
        )
        # 关闭思考模式（DeepSeek V4 Pro 等 reasoning 模型默认 effort=high，
        # 决策/评估/摘要这类「要快」的场景会先跑一大段 thinking 再输出，极慢）。
        # 通过 extra_body 传 {"thinking": {"type": "disabled"}} 关闭。
        if disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(**kwargs)


# ============================================================================
# 消息构建工具
# ============================================================================

def build_messages(system_prompt: str, history: list) -> list:
    """
    构建 LangChain 标准消息列表。

    将对话历史从 JSON 格式转换为 LangChain Message 对象列表，
    方便传入 ChatOpenAI.stream() 或 DoubaoLLM.stream()。

    Args:
        system_prompt: 系统提示词（角色设定 + RAG 检索上下文）
        history:       对话历史，格式 [{"role":"user/assistant", "content":"..."}, ...]

    Returns:
        [SystemMessage, HumanMessage, AIMessage, ...]
    """
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        role    = msg["role"]
        content = msg["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


# ============================================================================
# 通用非流式调用
# ============================================================================

def invoke_llm(llm, messages: list) -> str:
    """
    通用非流式调用 LLM，返回完整文本。

    兼容两种后端：
      - ChatOpenAI（有 .invoke() 方法）→ 直接 invoke
      - DoubaoLLM（只有 .stream() 方法）→ 用 stream() 拼接完整文本

    供 grade_node / rewrite_node / MCP 工具等「需要完整文本」的场景统一使用。

    Args:
        llm:      由 create_chat_model 创建的实例
        messages: LangChain 消息列表

    Returns:
        完整响应文本（字符串）
    """
    # 有 invoke 方法 → 直接调用
    if hasattr(llm, "invoke"):
        resp = llm.invoke(messages)
        if hasattr(resp, "content"):
            return resp.content or ""
        return str(resp)

    # 无 invoke（如 DoubaoLLM）→ 用 stream 拼接
    if hasattr(llm, "stream"):
        parts = []
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", None)
            if content:
                # 跳过思考内容，只拼接正文
                if not getattr(chunk, "is_reasoning", False):
                    parts.append(content)
        return "".join(parts)

    raise AttributeError(f"LLM 实例无 invoke/stream 方法: {type(llm).__name__}")

