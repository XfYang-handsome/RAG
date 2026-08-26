# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Controller 决策（Phase 5）
================================================================================

Controller = π(State)：根据当前 State（gaps + queries 历史 + iteration）决定
下一个 Action。

核心设计（与架构定稿一致）：

  1. Action 必须「Gap→Action」，而非「Question→Action」：
     每个非 ANSWER 的 Action 必须带 target_gap，指向某个 requirement_id。
  2. Action 空间最小化（V1 仅 5 个）：
     SEARCH / REFINE_QUERY / READ_PARENT / READ_SECTION / ANSWER
  3. 终止条件：
       - 无 gap（全部 SUPPORTED）→ ANSWER
       - iteration 已达上限 → ANSWER（由 Stopping 兜底，这里做最后一道保险）
  4. 查询去重：SEARCH/REFINE_QUERY 的 query 若与历史 query 完全相同，
     强制降级为 READ_PARENT / ANSWER，避免在死胡同里转圈。
  5. 容错：pydantic 校验 + 重试 1 次 + 兜底（非法输出 → ANSWER，宁停勿错）。

用法：
    from agentic_rag.controller import choose_action
    action = choose_action(state)  # 返回 Action，未就地改 state
================================================================================
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .state import (
    AgentState,
    Action,
    ActionType,
    RetrievalTool,
    IMPORTANCE_HIGH,
)
from .settings import get as _get

from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# 模型获取（tool_llm 优先，回退 llm）
# ============================================================================
def _get_controller_llm():
    from llm_factory import get_model
    return get_model("tool_llm", "llm")


# ============================================================================
# 结构化输出模型（pydantic 校验 + 归一）
# ============================================================================
_ACTION_ALIAS = {
    "SEARCH": ActionType.SEARCH,
    "REFINE_QUERY": ActionType.REFINE_QUERY,
    "REFINE": ActionType.REFINE_QUERY,
    "READ_PARENT": ActionType.READ_PARENT,
    "READ_SECTION": ActionType.READ_SECTION,
    "READ": ActionType.READ_SECTION,
    "WEB_SEARCH": ActionType.WEB_SEARCH,
    "WEB": ActionType.WEB_SEARCH,
    "ANSWER": ActionType.ANSWER,
    "STOP": ActionType.ANSWER,
    "SYNTHESIZE": ActionType.ANSWER,
}

_TOOL_ALIAS = {
    "vector": RetrievalTool.VECTOR,
    "hybrid": RetrievalTool.HYBRID,
    "bm25": RetrievalTool.HYBRID,
    "tree": RetrievalTool.TREE,
    "hierarchical": RetrievalTool.TREE,
}


class _ActionOut(BaseModel):
    type: ActionType
    target_gap: str = ""
    query: str = ""
    tool: RetrievalTool = RetrievalTool.HYBRID

    @field_validator("type", mode="before")
    @classmethod
    def _norm_type(cls, v):
        if isinstance(v, ActionType):
            return v
        return _ACTION_ALIAS.get(str(v).strip().upper(), ActionType.ANSWER)

    @field_validator("tool", mode="before")
    @classmethod
    def _norm_tool(cls, v):
        if isinstance(v, RetrievalTool):
            return v
        return _TOOL_ALIAS.get(str(v).strip().lower(), RetrievalTool.HYBRID)


# ============================================================================
# Prompt
# ============================================================================
_CONTROLLER_SYSTEM = (
    "你是多轮检索的决策器（Controller）。根据当前信息缺口（Gap）决定下一步动作，"
    "只输出一个 JSON 对象，不要任何解释、代码块标记或其他文字。"
    "生成检索 query 时：针对 target_gap 的具体信息点精准措辞，"
    "不要堆砌所有需求的关键词；若知识库文档可能为英文，可用英文关键词提升召回。"
)


def _build_prompt(state: AgentState) -> str:
    # 缺口行：除「缺什么」外，补充该缺口「已关联的证据摘要」，让 Controller 知道
    # 已有证据为什么不足，据此决定是换说法 / 换工具 / 读上下文 / 联网，而非盲目 refine。
    gap_lines = []
    for g in state.gaps:
        line = f"- {g.requirement_id}（重要度 {g.importance:.2f}）：缺 {g.missing_what}"
        item = state.requirement_status.get(g.requirement_id)
        if item and item.evidence_ids:
            ev_summaries = []
            for eid in item.evidence_ids[:3]:  # 最多展示 3 条，控制 prompt 体积
                ev = state.evidences.get(eid)
                if ev is None:
                    continue
                t = ev.text.replace("\n", " ").strip()
                ev_summaries.append(f"    · {t[:120]}")
            if ev_summaries:
                line += "\n  已有证据（不足够）：\n" + "\n".join(ev_summaries)
        gap_lines.append(line)
    gap_lines = "\n".join(gap_lines) if gap_lines else "（无缺口）"

    supported = [
        rid for rid, item in state.requirement_status.items()
        if item.status.value == "SUPPORTED"
    ]
    hist_lines = "\n".join(f"- {q.text} [{q.tool.value}]" for q in state.queries) or "（空）"

    web_hint = ""
    if state.can_web_search():
        web_hint = (
            f"当前已连续 {state.no_progress_rounds} 轮无需求状态升级，"
            f"且知识库检索已无新信息。若判断该缺口可能超出知识库范围，"
            f"可用 WEB_SEARCH 联网搜索补充（剩余次数 {_get('web_search.max_calls', 2) - state.web_search_count}）。\n"
        )

    return (
        f"用户问题：{state.question}\n\n"
        f"当前信息缺口（按重要度排序，含已有但不足的证据）：\n{gap_lines}\n\n"
        f"已解决的 requirement：{', '.join(supported) or '（无）'}\n\n"
        f"已执行过的检索 query（不要重复）：\n{hist_lines}\n\n"
        f"{web_hint}"
        "请输出下一个动作，JSON 格式：\n"
        '{"type": "SEARCH", "target_gap": "R3", "query": "检索文本", "tool": "hybrid"}\n\n'
        "可选 type（严格大写）：\n"
        "  - SEARCH：检索新证据（tool 只能是 vector/hybrid/tree，默认 hybrid）\n"
        "  - REFINE_QUERY：改写上一个检索 query（检索结果太差时）\n"
        "  - READ_PARENT / READ_SECTION：已有证据上下文不足，读父块/章节补上下文\n"
        "  - WEB_SEARCH：联网搜索（仅当知识库检索已无新进展、且问题可能超出知识库范围时）\n"
        "  - ANSWER：信息已足够，可以合成答案\n\n"
        "规则：\n"
        "1. target_gap 必须指向上面某个缺口（如 R3）；ANSWER 时可为空。\n"
        "2. SEARCH 必须给出 query；query 只针对 target_gap 的核心信息点，"
        "3~5 个关键词即可，不要堆砌所有需求的关键词。\n"
        "3. 若连续多轮中文 query 检索效果差（新增证据少），可改用英文关键词 query。\n"
        "4. 若缺口已有证据但不足以支撑，优先考虑 READ_PARENT / READ_SECTION 补上下文，"
        "而非重复检索；只有确实换方向才 SEARCH。\n"
        "5. WEB_SEARCH 必须给出 query，且 query 用面向搜索引擎的自然语言"
        "（中文问题用中文）；只有在知识库检索已无新进展时才用它，不要一上来就联网。\n"
        "6. 若所有高重要度缺口都已解决，输出 ANSWER。\n"
        "7. 只输出一个动作。"
    )


# ============================================================================
# 核心逻辑
# ============================================================================
def _recent_queries(state: AgentState) -> set:
    return {q.text.strip() for q in state.queries if q.text.strip()}


def choose_action(state: AgentState, llm=None) -> Action:
    """根据 State 决定下一个 Action。

    Args:
        state: 当前状态（gaps / queries / iteration 已填充）
        llm:   决策模型（None=内部取 tool_llm）

    Returns:
        Action（不会就地修改 state；由 Executor 负责写回）
    """
    # 1. 硬终止：无 gap 或已达 iteration 上限 → ANSWER
    if not state.gaps:
        return Action(type=ActionType.ANSWER)
    if state.iteration >= state.budget.max_iterations:
        return Action(type=ActionType.ANSWER)

    # 2. LLM 决策
    if llm is None:
        llm = _get_controller_llm()
    if llm is None:
        return Action(type=ActionType.ANSWER)

    from llm import invoke_llm
    from common.text_utils import parse_json

    prompt = _build_prompt(state)
    recent = _recent_queries(state)

    # retry 表示「失败后重试次数」，总尝试次数 = retry + 1
    for _ in range(_get("controller.retry", 1) + 1):
        try:
            text = invoke_llm(llm, [SystemMessage(content=_CONTROLLER_SYSTEM), HumanMessage(content=prompt)])
            obj = parse_json(text)
            if obj is None:
                continue
            out = _ActionOut.model_validate(obj)

            # 校验 target_gap：非 ANSWER 时必须指向一个真实存在的 gap
            valid_gap_ids = {g.requirement_id for g in state.gaps}
            if out.type != ActionType.ANSWER and out.target_gap not in valid_gap_ids:
                # 若 target_gap 非法，尝试回退到第一个 gap
                if state.gaps:
                    out.target_gap = state.gaps[0].requirement_id
                else:
                    out = _ActionOut(type=ActionType.ANSWER)

            # SEARCH/REFINE_QUERY/WEB_SEARCH 必须有 query
            if out.type in (ActionType.SEARCH, ActionType.REFINE_QUERY, ActionType.WEB_SEARCH) and not out.query.strip():
                continue  # 非法，重试

            action = Action(
                type=out.type,
                target_gap=out.target_gap,
                query=out.query.strip(),
                tool=out.tool,
            )

            # 查询去重：SEARCH/REFINE_QUERY 的 query 与历史完全相同 → 降级 READ_PARENT
            if action.type in (ActionType.SEARCH, ActionType.REFINE_QUERY):
                if action.query in recent:
                    action = Action(type=ActionType.READ_PARENT, target_gap=action.target_gap)

            # WEB_SEARCH 硬约束（守住底线，防止 LLM 过早/过度联网）：
            #   1) 总开关关闭 / 次数用尽 → 降级 SEARCH（用原 query 继续搜知识库）
            #   2) 尚未出现「无进展」信号（还在正常推进）→ 降级 SEARCH，不要一上来就联网
            if action.type == ActionType.WEB_SEARCH:
                if not _get("web_search.enabled", True) or not state.can_web_search():
                    action = Action(
                        type=ActionType.SEARCH,
                        target_gap=action.target_gap,
                        query=action.query,
                        tool=RetrievalTool.HYBRID,
                    )
                elif state.no_progress_rounds < _get("web_search.min_no_progress_before", 1):
                    action = Action(
                        type=ActionType.SEARCH,
                        target_gap=action.target_gap,
                        query=action.query,
                        tool=RetrievalTool.HYBRID,
                    )

            return action
        except Exception:
            continue

    # 3. 兜底：非法输出 → ANSWER（宁停勿错，避免死循环）
    return Action(type=ActionType.ANSWER)
