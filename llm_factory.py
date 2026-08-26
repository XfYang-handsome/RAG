# -*- coding: utf-8 -*-
"""
================================================================================
统一 LLM 实例工厂（消除各处重复的 _get_xxx_llm 复制粘贴）
================================================================================

背景：planner/controller/router/executor/evaluator/synthesizer/summarizer/
structure_resolver 等模块各自实现了几乎相同的「按回退链取模型 → create_chat_model」
逻辑，且大多每次调用都新建实例（无缓存），导致：
  1. 约 8 处 15 行的复制粘贴，改 bug（disable_thinking / protocol / temperature）
     要同步改多处，极易遗漏；
  2. Agentic 循环每轮（plan/choose_action/execute/evaluate）都重复新建 ChatModel
     客户端，浪费连接创建开销。

本模块提供唯一入口 get_model(*kinds, answer)，带进程级缓存：
  - 缓存 key = (回退链各 kind 的完整配置签名, answer)，配置变更时自动失效；
  - answer=True  → 生成答案模型（temperature 用答案温度，不 disable_thinking）；
  - answer=False → 决策/评估/摘要模型（disable_thinking 读配置，temperature 默认 0）。

用法：
    from llm_factory import get_model

    decision = get_model("tool_llm", "llm")          # 决策类（回退链）
    summary  = get_model("summary", "tool_llm", "llm")
    answer   = get_model("llm", answer=True)          # 生成答案
================================================================================
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

_lock = threading.RLock()
_cache = {}


def _pick_model(kind: str) -> Optional[dict]:
    """按 kind 取当前选中模型配置（按 name 查，回退第一个）。"""
    import store_config
    name = store_config.get_current(kind)
    m = store_config.get_model_by_name(kind, name) if name else None
    if m is None:
        models = store_config.list_models(kind)
        m = models[0] if models else None
    return m


def _model_sig(m: Optional[dict]) -> Optional[tuple]:
    """模型配置签名（用于缓存失效判断，配置内容变了缓存即失效）。"""
    if not m:
        return None
    return (
        m.get("model"),
        m.get("base_url"),
        m.get("api_key"),
        m.get("protocol"),
        m.get("disable_thinking"),
        m.get("temperature"),
    )


def get_model(*kinds: str, answer: bool = False):
    """按回退链获取模型实例（带进程级缓存）。

    Args:
        kinds:  模型类型回退链，如 ("tool_llm", "llm")、("summary", "tool_llm", "llm")。
                按顺序取第一个可用的模型。
        answer: True=生成答案模型（temperature 用答案温度，不 disable_thinking）；
                False=决策/评估/摘要模型（disable_thinking 读配置，temperature 默认 0）。

    Returns:
        ChatModel 实例；所有 kind 都无可用模型时返回 None。
    """
    from llm import create_chat_model, DEFAULT_ANSWER_TEMPERATURE

    chosen = None
    sig_parts = []
    for kind in kinds:
        m = _pick_model(kind)
        sig_parts.append((kind, _model_sig(m)))
        if chosen is None and m is not None:
            chosen = m

    cache_key = (tuple(sig_parts), answer)

    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

    if chosen is None:
        return None

    if answer:
        model = create_chat_model(
            model=chosen.get("model"),
            base_url=chosen.get("base_url"),
            api_key=chosen.get("api_key"),
            protocol=chosen.get("protocol", "openai"),
            temperature=chosen.get("temperature", DEFAULT_ANSWER_TEMPERATURE),
        )
    else:
        model = create_chat_model(
            model=chosen.get("model"),
            base_url=chosen.get("base_url"),
            api_key=chosen.get("api_key"),
            protocol=chosen.get("protocol", "openai"),
            disable_thinking=chosen.get("disable_thinking", False),
        )

    with _lock:
        _cache[cache_key] = model
    return model


def clear_cache() -> None:
    """清空缓存（可选；签名机制已能在配置变更时自动失效，此函数供强制刷新）。"""
    global _cache
    with _lock:
        old = _cache
        _cache = {}
    for m in old.values():
        try:
            m.close()
        except Exception:
            pass
