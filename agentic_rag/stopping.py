# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Stopping Policy（Phase 7）
================================================================================

确定性地终止循环，且能解释「为什么停」。纯规则、无 LLM Judge（LLM 判定
只会引入「该停不停」的不确定性，留 V2）。

三层规则（与架构定稿一致）：

  第一层 Hard Stop（防死循环）
    - iteration  >= max_iterations（轮数限制，默认 5 轮）
    - tool_calls >= max_tool_calls

  第二层 Sufficiency（已足够）
    - 所有 importance >= IMPORTANCE_HIGH 的 requirement 均 SUPPORTED

  第三层 No-progress（继续无收益）
    - no_progress_rounds >= 2（连续两轮无 requirement 状态升级）

用法：
    from agentic_rag.stopping import should_stop, update_no_progress

    update_no_progress(state)            # 每轮 evaluate 后调用
    stop, reason = should_stop(state)    # 判断是否停
================================================================================
"""

from __future__ import annotations

from typing import Tuple

from .state import AgentState, IMPORTANCE_HIGH
from .settings import get as _get

# 连续无进展轮数阈值（第三层 No-progress）
NO_PROGRESS_THRESHOLD = _get("no_progress_threshold", 2)


def update_no_progress(state: AgentState) -> int:
    """每轮 evaluate 后调用：根据本轮进展更新 no_progress_rounds。

    进展定义（Gap Reduction）：
      本轮有任一 requirement 状态**升级**（MISSING→PARTIAL / PARTIAL→SUPPORTED
      / MISSING→SUPPORTED）才算有进展；单纯新增 evidence（哪怕检索到一堆
      不相关证据）不算进展。

    这是根治「空转」的关键：把「检索到了」和「解决了」区分开——
    检索到垃圾证据不推进任何 requirement，就应尽快停。

    Returns:
        更新后的 no_progress_rounds。
    """
    if state.newly_upgraded_ids():
        state.no_progress_rounds = 0
    else:
        state.no_progress_rounds += 1
    return state.no_progress_rounds


def should_stop(state: AgentState) -> Tuple[bool, str]:
    """判断是否终止循环，返回 (是否停止, 停止原因)。"""
    # 第一层：Hard Stop（以轮数为主，不再用时间限制强制结束）
    if state.iteration >= state.budget.max_iterations:
        return True, f"hard: 已达最大迭代次数 {state.budget.max_iterations}"
    if state.budget.tool_calls >= state.budget.max_tool_calls:
        return True, f"hard: 已达最大工具调用次数 {state.budget.max_tool_calls}"

    # 无 requirement 时不进入 Sufficiency 判定（空 requirements 由编排层处理，
    # 这里保守返回不停止，交给 hard stop 兜底）
    if not state.requirements:
        return False, ""

    # 第二层：Sufficiency（高重要度 requirement 全部解决）
    if not state.high_importance_unsupported():
        return True, "sufficiency: 高重要度信息需求全部满足"

    # 第三层：No-progress（连续多轮无 requirement 状态升级）
    if state.no_progress_rounds >= NO_PROGRESS_THRESHOLD:
        return True, f"no_progress: 连续 {NO_PROGRESS_THRESHOLD} 轮无需求状态升级"

    return False, ""
