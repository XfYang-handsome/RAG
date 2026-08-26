# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Complexity Router（Phase 9）
================================================================================

判断问题是否需要进入 Agent Loop：
  - SIMPLE  → 直接单轮检索 + Synthesis（≈ 现有 rag 链路，零额外循环）
  - COMPLEX → Planner 拆 requirements → Agent Loop

核心原则：不是所有问题都值得 Agentic RAG。

判据（规则优先 + LLM 兜底）：
  规则命中以下任一 → COMPLEX：
    1. 比较类：比较 / 对比 / 区别 / 差异 / 异同 / 优劣 / 优缺点 / 哪个更好 / 分别
    2. 多实体并列：出现多个并列连接（和 / 与 / 以及 / 、）
    3. 时间演进：演进 / 发展历程 / 历史 / 趋势 / 从...到 / 变化 / 阶段
    4. 多子问题：问号 > 1 个

  规则未命中 → LLM 兜底（可选）；LLM 不确定/不可用 → SIMPLE。

  权衡：COMPLEX 误判 SIMPLE 会导致漏答（更严重），SIMPLE 误判 COMPLEX 只多付
  几次 LLM 调用。因此规则命中才进循环，规则不命中时用 LLM 兜底把关。

用法：
    from agentic_rag.router import classify_complexity
    kind = classify_complexity(question)   # "SIMPLE" / "COMPLEX"
================================================================================
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------
# 明确需要「拆成多个 requirement」的信号 → COMPLEX
_COMPARE_PAT = re.compile(r"(比较|对比|区别|差异|异同|优劣|优缺点|哪个更好|谁更|分别|各自)")
_TEMPORAL_PAT = re.compile(r"(演进|发展历程|历史|趋势|从.{0,8}到|变化|阶段|演变)")
_MULTI_QUESTION_PAT = re.compile(r"[？?]")

# 多实体并列（但无对比语义）→ MEDIUM：可能需要多轮检索，但未必拆成多个需求
_MULTI_ENTITY_PAT = re.compile(r"(和|与|以及|、)[^。？！]{1,40}(和|与|以及)")


def _rule_classify(question: str) -> str:
    # 对比 / 时间演进 / 多子问题 → 需要 Planner 拆多个 requirement
    if _COMPARE_PAT.search(question):
        return "COMPLEX"
    if _TEMPORAL_PAT.search(question):
        return "COMPLEX"
    if len(_MULTI_QUESTION_PAT.findall(question)) > 1:
        return "COMPLEX"
    # 多实体并列 → 中等复杂度（多轮检索，单需求也能覆盖，拆解交给 LLM 兜底）
    if _MULTI_ENTITY_PAT.search(question):
        return "MEDIUM"
    return ""


# ---------------------------------------------------------------------------
# LLM 兜底
# ---------------------------------------------------------------------------
def _get_router_llm():
    from llm_factory import get_model
    return get_model("tool_llm", "llm")


def _llm_classify(question: str, llm) -> str:
    from llm import invoke_llm

    prompt = (
        "判断下面问题「是否需要拆解成多个信息需求分别检索」，只输出一个词：\n"
        "COMPLEX：问题包含多个需要分别回答的子问题（如比较多个对象、对比差异、"
        "跨时间段演进、多个并列实体需逐一说明），必须拆解才能答全；\n"
        "MEDIUM：单一需求，但可能需要 2~3 轮检索（换说法 / 补上下文）才能找全；\n"
        "SIMPLE：单一事实 / 定义 / 说明，一次检索即可回答，无需拆解。\n\n"
        f"问题：{question}"
    )
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是问题复杂度判断器，负责判断问题是否需要拆解成多个子需求。"),
            HumanMessage(content=prompt),
        ]).strip().upper()
        if "COMPLEX" in text:
            return "COMPLEX"
        if "MEDIUM" in text:
            return "MEDIUM"
        if "SIMPLE" in text:
            return "SIMPLE"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def classify_complexity(question: str, llm=None) -> str:
    """判断问题复杂度，返回 "SIMPLE" / "MEDIUM" / "COMPLEX"。

    策略：规则快速短路 + LLM 主导判断。
      - 规则明确命中 COMPLEX（对比 / 时间演进 / 多问号）→ 直接 COMPLEX，
        这些信号几乎不会误判，省一次 LLM 调用。
      - 其余情况（含多实体并列 MEDIUM、规则未命中）→ 交给 LLM 判断
        「是否需要拆解」，LLM 能识别规则覆盖不到的语义复杂度（如
        「有什么 X」的列举对比、「帮我全面分析 Y」的多维度）。
      - LLM 不可用 → 回退规则结果，再回退 SIMPLE。
    """
    rule = _rule_classify(question or "")
    if rule == "COMPLEX":
        return rule

    if llm is None:
        llm = _get_router_llm()
    if llm is not None:
        result = _llm_classify(question, llm)
        if result:
            return result

    return rule or "SIMPLE"
