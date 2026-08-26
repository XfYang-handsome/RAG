# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — 统一配置入口
================================================================================

所有 Agentic RAG 的魔法数字 / 阈值 / 字符限制等字面量，一律从
``config/config.json`` 的 ``agentic`` 块读取，代码里不再硬编码。

用法：

    from .settings import get
    top_k = get("search.top_k", 5)

``get`` 的第二个参数是「config.json 缺省该键时的兜底值」。config.json 里
写全了 agentic 块后，兜底值只用于容错（配置被删 / 键名拼错）。
================================================================================
"""

from __future__ import annotations

from config_loader import cfg


def get(path: str, default=None):
    """读取 ``agentic.<path>``，缺省时返回 default。

    示例：
        get("search.top_k")                  → config.json 中 agentic.search.top_k
        get("search.top_k", 5)               → 缺省时返回 5
    """
    return cfg(f"agentic.{path}", default)
