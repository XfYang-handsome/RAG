# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — 编排层（Phase 10）
================================================================================

把 planner / retriever / evaluator / controller / executor / stopping /
synthesizer 串成完整的 Requirement–Evidence–Gap 循环。

流程：

    Question → Complexity Router
      ├── SIMPLE  → 单轮检索 + 评估 + Synthesis（零额外循环）
      └── COMPLEX → Planner 拆 requirements → Agent Loop：
                        ┌──────────────────────────────┐
                        │  snapshot → Controller 决策   │
                        │  → Executor 执行 → Evaluator   │
                        │  评估 → no_progress → Stop?    │
                        └──────────────┬───────────────┘
                                       │ STOP
                                       ▼
                                  Synthesis

实现说明：
  - 用纯 Python while 循环实现（而非 LangGraph 条件边）：循环逻辑简单直观，
    且每轮可打印 trace_line，天然满足「可观测性」要求。
  - status_callback 用于把每轮决策轨迹推给前端（可观测性落地点）。

用法：
    from agentic_rag.agent import run_agentic
    result = run_agentic("比较 OpenAI 与 Anthropic 的 Agent 路线",
                         reranker=reranker,
                         status_callback=print)
    # result = {"answer": ..., "citations": [...], "state": AgentState, "complexity": "COMPLEX"}
================================================================================
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from .state import (
    AgentState,
    Action,
    ActionType,
    Requirement,
    RetrievalTool,
)
from .settings import get as _get
from config_loader import cfg as _cfg
from .router import classify_complexity
from .planner import plan, expand_deferred
from .controller import choose_action
from .executor import execute
from .evaluator import evaluate
from .stopping import should_stop, update_no_progress
from .synthesizer import synthesize
from .retriever import mode_to_tool


def _effective_tool_label(action, retrieval_mode: str = "") -> str:
    """返回「实际执行」的检索工具标签（日志可观测用）。

    execute() 内部会 force_tool = mode_to_tool(retrieval_mode) 覆盖 Controller
    决策出的 tool，因此日志展示必须用覆盖后的值，否则会出现「用户选树导航、
    日志却显示 SEARCH [hybrid]」的误导。
    """
    if action is None:
        return "-"
    if action.type == ActionType.WEB_SEARCH:
        return "web"
    if retrieval_mode:
        return mode_to_tool(retrieval_mode).value
    return action.tool.value if action.tool else "-"


# ---------------------------------------------------------------------------
# 需求级缓存（会话内相似问题复用）
# ---------------------------------------------------------------------------
# 结构：{conversation_id: [(normalized_question, complexity, AgentState), ...]}
# 命中后跳过整个检索循环，直接用已解决的 requirement_status + evidences 合成答案，
# 让「追问 / 换种问法」类问题不必重新检索，显著降低延迟与成本。
# 用简单 LRU 上限防内存膨胀；会话结束 / 服务重启自动清空（进程内缓存）。
_cache: Dict[str, List[Tuple[str, str, AgentState]]] = {}


def _norm_question(q: str) -> str:
    """问题归一化：去空白 + 小写，作为精确命中 key。"""
    return re.sub(r"\s+", "", (q or "")).lower()


def _cache_lookup(conversation_id: str, question: str, reranker) -> Optional[Tuple[str, AgentState]]:
    """查缓存：先精确命中，再 reranker 相似度命中（超阈值则复用）。

    Returns:
        (complexity, state) 或 None（未命中）。
    """
    if not _get("cache.enabled", True) or not conversation_id:
        return None
    entries = _cache.get(conversation_id)
    if not entries:
        return None

    norm = _norm_question(question)
    # ① 精确命中（完全相同问题）
    for nq, complexity, state in entries:
        if nq == norm:
            return complexity, state

    # ② 相似命中：reranker 对「历史问题」打分，超过阈值复用
    if reranker is not None:
        hist_qs = [nq for nq, _, _ in entries]
        try:
            ranked = reranker.rerank(question, hist_qs)
        except Exception:
            ranked = []
        best = None
        for r in ranked:
            idx = r.get("index")
            score = float(r.get("score", 0.0) or 0.0)
            if 0 <= idx < len(entries) and (best is None or score > best[0]):
                best = (score, idx)
        if best and best[0] >= float(_get("cache.sim_threshold", 0.9)):
            _, complexity, state = entries[best[1]]
            return complexity, state
    return None


def _cache_store(conversation_id: str, question: str, complexity: str, state: AgentState) -> None:
    """写入缓存（LRU 上限，超出淘汰最早条目）。"""
    if not _get("cache.enabled", True) or not conversation_id:
        return
    max_entries = int(_get("cache.max_entries", 32))
    entries = _cache.setdefault(conversation_id, [])
    norm = _norm_question(question)
    # 去重：同一问题覆盖旧条目
    entries[:] = [(nq, c, s) for nq, c, s in entries if nq != norm]
    entries.append((norm, complexity, state))
    if len(entries) > max_entries:
        entries.pop(0)


def _fmt_requirement(r: Requirement) -> str:
    """需求的可读表示：R1(1.0) 描述。"""
    return f"{r.id}({r.importance:.1f}) {r.description}"


def _fmt_status(state: AgentState) -> str:
    """每个 requirement 的评估结果：R1=SUPPORTED, R2=MISSING, ..."""
    parts = []
    for r in state.requirements:
        item = state.requirement_status.get(r.id)
        st = item.status.value if item is not None else "MISSING"
        parts.append(f"{r.id}={st}")
    return ", ".join(parts) if parts else "（无需求）"


def _fmt_new_evidence(new_ids: list) -> str:
    """新增证据的可读表示（截断，避免刷屏）。"""
    if not new_ids:
        return "0 条"
    shown = ", ".join(new_ids[:5])
    if len(new_ids) > 5:
        shown += f" ...（共 {len(new_ids)} 条）"
    return f"{len(new_ids)} 条（{shown}）"


def _fmt_queries(queries: list) -> str:
    """实际检索 query 的可读表示（展示 multi-query 展开 + 英化后的真实 query）。"""
    if not queries:
        return "（无）"
    parts = [f"\"{q.text[:60]}\"" for q in queries]
    if len(parts) > 6:
        parts = parts[:6] + [f"...（共 {len(parts)} 个）"]
    return " / ".join(parts)


def _state_snapshot(state: AgentState, action=None, stop_reason: str = "") -> dict:
    """把当前状态序列化为结构化 dict（供前端「Agent 工作台」可视化）。

    包含：requirements 状态矩阵、evidence 累积、coverage、本轮 action、停止原因。
    这是任务 6「Agent 决策轨迹前端可视化」的数据源。
    """
    reqs = []
    for r in state.requirements:
        item = state.requirement_status.get(r.id)
        status = item.status.value if item is not None else "MISSING"
        ev_count = len(item.evidence_ids) if item is not None else 0
        reqs.append({
            "id": r.id,
            "description": r.description,
            "importance": r.importance,
            "status": status,
            "evidence_count": ev_count,
        })
    snap = {
        "iteration": state.iteration,
        "requirements": reqs,
        "evidences": len(state.evidences),
        "coverage": round(state.compute_coverage(), 3),
        "gaps": [g.requirement_id for g in state.gaps],
        "no_progress_rounds": state.no_progress_rounds,
        "stop_reason": stop_reason,
        "action": None,
    }
    if action is not None:
        snap["action"] = {
            "type": action.type.value,
            "target_gap": action.target_gap,
            "query": action.query,
            "tool": action.tool.value if action.tool else "",
        }
    return snap


def _run_light(question: str, reranker, status_callback, token_callback, top_k: int,
               retrieval_mode: str = None, max_rounds: int = 2,
               complexity: str = "SIMPLE", trace_callback: Optional[Callable[[dict], None]] = None) -> dict:
    """SIMPLE / MEDIUM 路径：单需求 + 轻量循环（最多 max_rounds 轮）。

    与 COMPLEX 的区别：不拆 requirements（单 R1），但保留 Controller 决策与
    「检索失败补救」能力。第一轮固定 SEARCH；evaluate 后若 R1 仍未 SUPPORTED，
    由 Controller 再决策 REFINE / 换工具 / READ / WEB_SEARCH / ANSWER，直到
    SUPPORTED 或达到轮数上限。根治 SIMPLE「单轮失败零补救」的问题。
    """
    label = "中等（多轮检索）" if complexity == "MEDIUM" else "简单（单轮检索）"
    if status_callback:
        status_callback(f"问题类型：{label}")

    state = AgentState(question=question)
    state.requirements = [Requirement(id="R1", description=question, importance=1.0)]
    state.start_budget()
    state.derive_gaps()
    # 轻量循环轮数上限：SIMPLE=2（首轮 + 1 次补救），MEDIUM=读配置（默认 3）
    state.budget.max_iterations = max_rounds

    while True:
        state.snapshot_round()
        if status_callback:
            status_callback(state.trace_line())

        # 首轮固定 SEARCH；之后交给 Controller 决策（含补救 / 换工具 / 联网）
        if state.iteration == 0:
            action = Action(type=ActionType.SEARCH, target_gap="R1",
                            query=question, tool=mode_to_tool(retrieval_mode) if retrieval_mode else RetrievalTool.HYBRID)
            if status_callback:
                status_callback(
                    f"Controller：SEARCH [{_effective_tool_label(action, retrieval_mode)}] → target_gap=R1"
                )
        else:
            action = choose_action(state)
            if action.type == ActionType.ANSWER:
                if status_callback:
                    status_callback("Controller：信息已充分，进入合成")
                if trace_callback:
                    trace_callback(_state_snapshot(state, action=action, stop_reason="sufficiency"))
                break
            if status_callback:
                desc = action.query or ""
                if action.type in (ActionType.SEARCH, ActionType.REFINE_QUERY, ActionType.WEB_SEARCH):
                    tool_label = _effective_tool_label(action, retrieval_mode)
                    status_callback(
                        f"Controller：{action.type.value} [{tool_label}] \"{desc[:40]}\" → target_gap={action.target_gap}"
                    )
                else:
                    status_callback(f"Controller：{action.type.value} → target_gap={action.target_gap}")

        # Executor 执行
        before = set(state.evidences.keys())
        q_before = len(state.queries)
        added = execute(state, action, top_k=top_k, retrieval_mode=retrieval_mode, reranker=reranker)
        if status_callback:
            new_queries = state.queries[q_before:]
            status_callback(
                f"检索：实际 query {_fmt_queries(new_queries)} → 新增 "
                f"{_fmt_new_evidence(sorted(set(state.evidences.keys()) - before))}"
            )

        # Evaluator 评估（无新增证据则跳过）
        if added > 0:
            if status_callback:
                status_callback("评估：正在评估证据相关性（LLM 判定）...")
            evaluate(state, reranker=reranker)
            if status_callback:
                status_callback(f"评估：{_fmt_status(state)}")
        else:
            if status_callback:
                status_callback("无新增证据，跳过评估")

        update_no_progress(state)

        stop, reason = should_stop(state)
        if trace_callback:
            trace_callback(_state_snapshot(state, action=action, stop_reason=reason))
        if stop:
            if status_callback:
                status_callback(f"停止：{reason}")
            break

    result = synthesize(state, stream_callback=token_callback)
    if status_callback:
        status_callback(f"合成答案（基于 {len(result['citations'])} 条证据）")
    result["state"] = state
    result["complexity"] = complexity
    return result


def _run_simple(question: str, reranker, status_callback, token_callback, top_k: int,
                retrieval_mode: str = None, trace_callback=None) -> dict:
    """SIMPLE 路径：轻量循环（最多 2 轮 = 首轮检索 + 1 次失败补救）。"""
    return _run_light(question, reranker, status_callback, token_callback, top_k,
                      retrieval_mode=retrieval_mode, max_rounds=2, complexity="SIMPLE",
                      trace_callback=trace_callback)


def _run_medium(question: str, reranker, status_callback, token_callback, top_k: int,
                retrieval_mode: str = None, trace_callback=None) -> dict:
    """MEDIUM 路径：单需求 + 多轮轻量循环（不拆 requirements）。"""
    max_rounds = int(_get("medium.max_iterations", 3))
    return _run_light(question, reranker, status_callback, token_callback, top_k,
                      retrieval_mode=retrieval_mode, max_rounds=max_rounds,
                      complexity="MEDIUM", trace_callback=trace_callback)


def _run_complex(question: str, reranker, status_callback, token_callback, top_k: int,
                 retrieval_mode: str = None, trace_callback=None) -> dict:
    """COMPLEX 路径：Planner 拆需求 → Agent Loop。"""
    if status_callback:
        status_callback("问题类型：复杂（拆解信息需求）")

    requirements = plan(question)
    if status_callback:
        status_callback(f"拆解出 {len(requirements)} 个需求: "
                        + " / ".join(_fmt_requirement(r) for r in requirements))

    state = AgentState(question=question, requirements=requirements)
    state.start_budget()
    state.derive_gaps()

    while True:
        state.snapshot_round()

        if status_callback:
            status_callback(state.trace_line())

        # 1. Controller 决策（读 gaps）
        action = choose_action(state)
        if action.type == ActionType.ANSWER:
            if status_callback:
                status_callback("Controller：信息已充分，进入合成")
            if trace_callback:
                trace_callback(_state_snapshot(state, action=action, stop_reason="sufficiency"))
            break

        if status_callback:
            desc = action.query or ""
            if action.type in (ActionType.SEARCH, ActionType.REFINE_QUERY, ActionType.WEB_SEARCH):
                tool_label = _effective_tool_label(action, retrieval_mode)
                status_callback(
                    f"Controller：{action.type.value} [{tool_label}] \"{desc[:40]}\" → target_gap={action.target_gap}"
                )
            else:
                status_callback(
                    f"Controller：{action.type.value} → target_gap={action.target_gap}"
                )

        # 2. Executor 执行
        before = set(state.evidences.keys())
        q_before = len(state.queries)
        added = execute(state, action, top_k=top_k, retrieval_mode=retrieval_mode, reranker=reranker)
        if status_callback:
            new_queries = state.queries[q_before:]
            status_callback(
                f"检索：实际 query {_fmt_queries(new_queries)} → 新增 "
                f"{_fmt_new_evidence(sorted(set(state.evidences.keys()) - before))}"
            )

        # 3. Evaluator 评估（无新增证据则跳过：证据未变，评估结果不会变，省 rerank + LLM）
        if added > 0:
            if status_callback:
                status_callback("评估：正在评估证据相关性（LLM 判定）...")
            evaluate(state, reranker=reranker)
            if status_callback:
                status_callback(f"评估：{_fmt_status(state)}")
        else:
            if status_callback:
                status_callback("无新增证据，跳过评估")

        # 3.5 枚举-展开：depends_on 已 SUPPORTED 的 deferred 对比需求 → 抽取真实清单动态展开
        expanded = expand_deferred(state)
        if expanded and status_callback:
            status_callback(
                f"展开对比需求：据检索到的真实清单生成 {len(expanded)} 个对比子需求 "
                f"({' / '.join(r.description[:20] for r in expanded)})"
            )

        # 3.6 综合归纳：依赖需求已 SUPPORTED 的关系/联系类需求 → 标记 SUPPORTED（由 Synthesis 归纳，不空转检索）
        resolved = state.resolve_synthetic()
        if resolved and status_callback:
            status_callback(
                f"综合归纳：{' / '.join(resolved)} 由依赖需求证据归纳，无需独立检索"
            )

        # 4. no_progress
        update_no_progress(state)

        # 5. Stopping
        stop, reason = should_stop(state)
        if trace_callback:
            trace_callback(_state_snapshot(state, action=action, stop_reason=reason))
        if stop:
            if status_callback:
                status_callback(f"停止：{reason}")
            break

    result = synthesize(state, stream_callback=token_callback)
    if status_callback:
        status_callback(f"合成答案（基于 {len(result['citations'])} 条证据）")
    result["state"] = state
    result["complexity"] = "COMPLEX"
    return result


def run_agentic(
    question: str,
    reranker=None,
    status_callback: Optional[Callable[[str], None]] = None,
    token_callback: Optional[Callable[[str, bool], None]] = None,
    top_k: int = None,
    retrieval_mode: str = None,
    conversation_id: str = "",
    trace_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Agentic RAG 主入口。

    Args:
        question:       用户问题
        reranker:       Reranker 实例（None=跳过 rerank，用检索分数，仅测试用）
        status_callback: 状态回调 fn(status_str)，用于前端可观测
        token_callback:  流式 token 回调 fn(content, is_reasoning)，用于前端逐字输出
        top_k:          检索返回数量（None=读配置 agentic.search.top_k）
        retrieval_mode: 检索模式（"vector"/"hybrid"/"tree"，None=读 config 默认）
        conversation_id: 会话 ID（用于需求级缓存，可为空=不缓存）
        trace_callback:  结构化轨迹回调 fn(dict)，每轮评估后推状态快照（前端工作台）

    Returns:
        {"answer": str, "citations": [...], "state": AgentState, "complexity": str}
    """
    if top_k is None:
        top_k = _get("search.top_k", 5)
    if not retrieval_mode:
        # 检索模式在顶层 search.retrieval_mode（非 agentic 块内），用 cfg 直读
        retrieval_mode = _cfg("search.retrieval_mode", "hybrid")

    question = (question or "").strip()
    if not question:
        return {"answer": "", "citations": [], "complexity": "SIMPLE"}

    # 需求级缓存：命中则跳过检索循环，直接用上次已解决的证据合成
    cached = _cache_lookup(conversation_id, question, reranker)
    if cached is not None:
        cached_complexity, cached_state = cached
        if status_callback:
            status_callback("命中需求级缓存：复用上次已解决的证据，跳过检索")
        if trace_callback:
            trace_callback(_state_snapshot(cached_state, stop_reason="cache_hit"))
        result = synthesize(cached_state, stream_callback=token_callback)
        result["state"] = cached_state
        result["complexity"] = cached_complexity
        result["from_cache"] = True
        return result

    kind = classify_complexity(question)
    if kind == "COMPLEX":
        result = _run_complex(question, reranker, status_callback, token_callback, top_k,
                              retrieval_mode, trace_callback=trace_callback)
    elif kind == "MEDIUM":
        result = _run_medium(question, reranker, status_callback, token_callback, top_k,
                             retrieval_mode, trace_callback=trace_callback)
    else:
        result = _run_simple(question, reranker, status_callback, token_callback, top_k,
                             retrieval_mode, trace_callback=trace_callback)

    # 缓存本次结果（有证据才缓存，空结果不污染缓存）
    if result.get("citations"):
        _cache_store(conversation_id, question, result["complexity"], result["state"])
    return result
