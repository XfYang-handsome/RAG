# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Evaluator 结构化判断（Phase 3）
================================================================================

每次 Retrieval 后，评估「这些证据到底解决了哪些 Requirement」。

设计（与架构定稿一致）：

  1. LLM 只做**离散判断**（SUPPORTED / PARTIAL / MISSING），不打浮点分。
  2. 所有浮点指标（coverage / quality / novelty）由代码从结构化结果导出。
  3. 检索返回的 score 语义不可比（COSINE 距离「越小越好」、RRF 分数「越大越好」，
     量级差百倍），因此 Evaluator 先统一用 **Reranker 分数覆盖** evidence.score，
     使所有证据落到同一可比尺度（quality 指标的来源）。
  4. 复用 grade 节点「reranker 分数兜底跳过 LLM」的思路：
       - 无相关证据（score 全部低于阈值）→ 直接判 MISSING，不调 LLM。
  5. 单调性约束：evidence 只增不减，已 SUPPORTED 的 requirement **不允许被降级**，
     防止 LLM 每轮判断抖动导致 coverage 忽高忽低、Stopping 乱判。
  6. 容错：pydantic 校验 + 重试 1 次 + 兜底（解析失败判 PARTIAL，让循环继续补救）。

用法：

    from agentic_rag.evaluator import evaluate
    status = evaluate(state, reranker=reranker, llm=tool_llm)  # 就地写回 state
================================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from common.text_utils import parse_json

from .state import (
    AgentState,
    Requirement,
    RequirementStatus,
    RequirementStatusItem,
)
from .settings import get as _get

from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# 模型获取（tool_llm 优先，回退 llm）
# ============================================================================
def _get_evaluator_llm():
    """获取评估模型（tool_llm → llm）；无可用模型返回 None。"""
    from llm_factory import get_model
    return get_model("tool_llm", "llm")


# ============================================================================
# 结构化输出模型（pydantic 校验 + 容错归一）
# ============================================================================
_STATUS_ALIAS = {
    "SUPPORTED": RequirementStatus.SUPPORTED,
    "PARTIAL": RequirementStatus.PARTIAL,
    "PARTIALLY": RequirementStatus.PARTIAL,
    "MISSING": RequirementStatus.MISSING,
    "NONE": RequirementStatus.MISSING,
    "EMPTY": RequirementStatus.MISSING,
}


class _EvalItem(BaseModel):
    id: str
    status: RequirementStatus
    evidence_ids: List[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v):
        """大小写 / 同义词归一，非法值兜底 PARTIAL（保守，不误判 SUPPORTED）。"""
        if isinstance(v, RequirementStatus):
            return v
        return _STATUS_ALIAS.get(str(v).strip().upper(), RequirementStatus.PARTIAL)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _norm_ids(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [x.strip() for x in v.split(",") if x.strip()] if v else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


class _EvalResult(BaseModel):
    requirements: List[_EvalItem] = Field(default_factory=list)


# ============================================================================
# Prompt
# ============================================================================
_EVAL_SYSTEM = (
    "你是检索证据评估器。请逐条判断给定需求是否被证据充分支持，只输出 JSON，"
    "不要任何解释、代码块标记或其他文字。"
    "证据可能是英文（知识库为英文文档），请依据语义判断相关性，"
    "不要因证据与需求语言不同而判低。"
)


def _build_prompt(question: str, requirements: List[Requirement], relevant: Dict[str, "Evidence"], state=None):
    """构造评估 prompt，返回 (prompt, id_map)。

    上下文分层压缩（借鉴 deepagents「分层压缩」思想）：
      - 本轮新增证据：全文 500 字符（LLM 重点判断这批是否补齐缺口）
      - 历史 pending 证据：截短 150 字符（pending 需求既有的 PARTIAL 证据，判断完整性）
      - 已 SUPPORTED 需求的证据：**不再逐条进 prompt**，用一行「已解决」状态摘要替代
        （这些证据对 pending 判断无贡献，逐条展示会让 prompt 随轮数线性膨胀）

    剔除安全约束：仅剔除「只被已 SUPPORTED 需求引用、且未被任何 pending 需求引用」
    的证据；pending 需求引用的历史证据完整保留，避免 LLM 因看不到既有证据而
    误判完整性（退化为永远 PARTIAL）。

    id_map: 短编号 -> 真实 evidence_id，避免 LLM 引用错长 id。
    """
    pending_ids = {r.id for r in requirements}

    # 1. 计算可剔除证据：只被已 SUPPORTED 需求引用、且未被任何 pending 需求引用
    droppable_eids: set = set()
    if state is not None:
        supported_eids: set = set()
        pending_eids: set = set()
        for r in state.requirements:
            item = state.requirement_status.get(r.id)
            if item is None:
                continue
            if item.status == RequirementStatus.SUPPORTED:
                supported_eids.update(item.evidence_ids)
            elif r.id in pending_ids:
                pending_eids.update(item.evidence_ids)
        droppable_eids = supported_eids - pending_eids

    # 2. 需求展示：pending 需求完整展示 + 已 SUPPORTED 需求一行状态摘要
    req_lines = [
        f"- {r.id}（重要度 {r.importance:.2f}）：{r.description}" for r in requirements
    ]
    if state is not None:
        for r in state.requirements:
            if r.id in pending_ids:
                continue
            item = state.requirement_status.get(r.id)
            if item is not None and item.status == RequirementStatus.SUPPORTED:
                note = (item.note or "").strip()[:60]
                req_lines.append(f"- {r.id}（已解决）：{note}")

    # 3. 证据展示：剔除 droppable 证据后再连续编号
    new_eids = set(state.new_evidence_ids()) if state is not None else set()
    new_chars = _get("evaluator.new_evidence_chars", 500)
    hist_chars = _get("evaluator.history_evidence_chars", 150)
    summ_chars = _get("evaluator.summary_chars", 80)
    id_map: Dict[str, str] = {}
    ev_lines: List[str] = []
    kept = {eid: ev for eid, ev in relevant.items() if eid not in droppable_eids}
    for num, (eid, ev) in enumerate(kept.items(), 1):
        id_map[str(num)] = eid
        # 新增证据全文，历史证据截短（控制 prompt 长度）
        is_new = eid in new_eids
        text = ev.text[:new_chars] if is_new else ev.text[:hist_chars]
        tag = "新增" if is_new else "历史"
        # 附带章节摘要（中文），帮助 LLM 理解英文 chunk 所属章节语义
        summary = (ev.source.summary or "")[:summ_chars]
        line = f"[{num}]({tag}) {text}"
        if summary:
            line += f"\n   └ 章节摘要：{summary}"
        ev_lines.append(line)
    return (
        f"用户问题：{question}\n\n"
        f"信息需求：\n{chr(10).join(req_lines)}\n\n"
        f"候选证据：\n{chr(10).join(ev_lines) or '（无）'}\n\n"
        "请输出 JSON，格式：\n"
        '{"requirements": [{"id": "R1", "status": "SUPPORTED", "evidence_ids": ["1"], "note": "..."}]}\n\n'
        "规则：\n"
        "1. status 只能是 SUPPORTED / PARTIAL / MISSING 之一（严格大写）。\n"
        "2. evidence_ids 只能填候选证据的编号（如 \"1\"、\"2\"），不得编造；无证据支撑则空数组。\n"
        "3. note 简要说明判断依据，MISSING/PARTIAL 时说明「缺什么」。\n"
        "4. 必须覆盖全部未标「已解决」的信息需求，一个都不能漏。\n"
        "5. 标「新增」的证据是本轮刚检索到的，标「历史」的是之前已检索的（已截短展示）。\n"
        "6. 标「已解决」的需求已完成评估，无需再判断，也不要给它输出任何状态。"
    ), id_map


# ============================================================================
# 核心逻辑
# ============================================================================
def _rerank_evidences(state: AgentState, reranker) -> None:
    """用 Reranker 对本轮**新增** evidence 打分，历史已 rerank 过的分数保持不变。

    优化：不再每轮全量 rerank。跨轮累积后全量 rerank 会线性变慢（第 4 轮 16 条，
    CPU 上跑 cross-encoder 尤其明显）。历史 evidence 的 score 在首次加入时已 rerank
    且保持同一尺度，无需重复计算，因此只 rerank 本轮新增。
    """
    if reranker is None or not state.evidences:
        return
    # 本轮新增的 evidence id（基于 snapshot_round 的快照）
    new_ids = state.new_evidence_ids()
    if not new_ids:
        return
    texts = [state.evidences[i].text for i in new_ids]
    try:
        ranked = reranker.rerank(state.question, texts)
    except Exception:
        return  # reranker 失败不阻塞，沿用原 score
    for item in ranked:
        idx = item.get("index")
        if idx is not None and 0 <= idx < len(new_ids):
            state.evidences[new_ids[idx]].score = float(item.get("score", 0.0))


def _prune_new_evidences(state: AgentState, top_n: Optional[int] = None) -> None:
    """rerank 后对本轮**新增**证据按 score 截断，丢弃低分证据（控制 prompt 膨胀）。

    动机：Multi-Query 一轮最多灌进 4×top_k 条证据，全部进 state 会让 evaluator
    的 prompt 线性膨胀。rerank 后只保留 score 最高的 top_n 条，其余丢弃。

    安全约束：
      - 只丢弃本轮新增：历史证据可能已被 requirement_status 引用，不能删。
      - 仅当新增证据数量 > top_n 时才截断（少量证据不误删，保护召回）。
      - top_n <= 0 表示不截断。

    注意：丢弃发生在 LLM 评估之前，被丢弃的证据不会被 requirement_status 引用，
    顺序安全。
    """
    if top_n is None:
        top_n = _get("evaluator.prune_top_n", 8)
    if not top_n or top_n <= 0:
        return
    new_ids = state.new_evidence_ids()
    if len(new_ids) <= top_n:
        return
    # 按 score 降序，保留 top_n，丢弃其余低分证据
    scored = sorted(new_ids, key=lambda i: state.evidences[i].score, reverse=True)
    for i in scored[top_n:]:
        state.evidences.pop(i, None)


def _all_missing(requirements: List[Requirement]) -> Dict[str, RequirementStatusItem]:
    return {
        r.id: RequirementStatusItem(
            requirement_id=r.id,
            status=RequirementStatus.MISSING,
            evidence_ids=[],
            note=f"{r.description} 尚无相关证据",
        )
        for r in requirements
    }


def _match_req_id(raw_id: str, req_ids: set) -> Optional[str]:
    """把 LLM 返回的 requirement id 精确匹配回真实 id。

    LLM 可能返回带空格/全角符号/重命名的 id，做归一化后精确匹配：
      - 精确命中（含 strip）→ 直接返回
      - 归一化后命中（如 "R1：" → "R1"）→ 返回
      - 完全无法匹配 → None（由调用方丢弃，走补全兜底）
    """
    rid = str(raw_id).strip() if raw_id else ""
    if rid in req_ids:
        return rid
    # 归一化：去掉全角/半角冒号、空格、下划线等常见噪声
    norm = rid.replace("：", "").replace(":", "").replace(" ", "").replace("_", "")
    if norm in req_ids:
        return norm
    for real in req_ids:
        if norm == real.replace("_", "").replace(":", ""):
            return real
    return None


def _llm_evaluate(
    question: str,
    requirements: List[Requirement],
    relevant: Dict[str, "Evidence"],
    llm,
    state=None,
) -> Dict[str, RequirementStatusItem]:
    """调 LLM 做离散判断，含 pydantic 校验 + 重试 + 兜底。"""
    from llm import invoke_llm

    prompt, id_map = _build_prompt(question, requirements, relevant, state=state)

    # retry 表示「失败后重试次数」，总尝试次数 = retry + 1
    for _ in range(_get("evaluator.retry", 1) + 1):
        try:
            text = invoke_llm(llm, [SystemMessage(content=_EVAL_SYSTEM), HumanMessage(content=prompt)])
            obj = parse_json(text)
            if obj is None:
                continue
            result = _EvalResult.model_validate(obj)
            status_map: Dict[str, RequirementStatusItem] = {}
            req_ids = {r.id for r in requirements}
            for item in result.requirements:
                # 短编号映射回真实 evidence id，同时过滤 LLM 幻觉编造的编号
                clean_ids = [id_map[e] for e in item.evidence_ids if e in id_map]
                # 关键修复：把 LLM 返回的 requirement id 精确匹配回真实 id。
                # LLM 可能返回带空格/后缀（"R1 "、"R1："）或重命名（"req1"），
                # 若直接用其原始 id 作 key，会导致 requirement_status 里出现
                # 与 requirements 不一致的 key，compute_coverage 恒为 0。
                rid = _match_req_id(item.id, req_ids)
                if rid is None:
                    continue  # 无法匹配到任何真实需求 → 丢弃（由下方补全兜底）
                # 不变量：SUPPORTED 必须至少引用一条有效证据；声称支持却无证据
                # 是矛盾的（会让 Synthesis 收集不到引用），此时降级 PARTIAL
                status = item.status
                note = item.note
                if status == RequirementStatus.SUPPORTED and not clean_ids:
                    status = RequirementStatus.PARTIAL
                    note = (note + "；判定支持但无有效证据引用，降级 PARTIAL").strip("；")
                status_map[rid] = RequirementStatusItem(
                    requirement_id=rid,
                    status=status,
                    evidence_ids=clean_ids,
                    note=note,
                )
            # 以 requirements 为准补全漏掉的项
            for r in requirements:
                status_map.setdefault(
                    r.id,
                    RequirementStatusItem(
                        requirement_id=r.id,
                        status=RequirementStatus.MISSING,
                        note="LLM 未给出该需求的评估",
                    ),
                )
            return status_map
        except Exception:
            continue

    # 兜底：解析/校验持续失败 → 全判 PARTIAL（保守，让循环继续 READ/SEARCH 补救）
    return {
        r.id: RequirementStatusItem(
            requirement_id=r.id,
            status=RequirementStatus.PARTIAL,
            note="评估失败，默认 PARTIAL",
        )
        for r in requirements
    }


def evaluate(
    state: AgentState,
    reranker=None,
    llm=None,
    threshold: Optional[float] = None,
) -> Dict[str, RequirementStatusItem]:
    """评估 evidence 对 requirements 的满足程度，就地写回 state.requirement_status。

    Args:
        state:     当前状态（requirements + evidences 已填充）
        reranker:  Reranker 实例（None=跳过 rerank，直接用检索分数，仅测试用）
        llm:       评估模型（None=内部取 tool_llm）
        threshold: 已废弃（保留签名兼容）：相关性判定不再依赖 reranker 硬阈值，
                   而是交给 LLM 依据文本独立判断（跨语言场景 reranker 分数不可靠）

    Returns:
        {requirement_id: RequirementStatusItem}，同时写回 state.requirement_status
    """
    requirements = state.requirements
    if not requirements:
        state.requirement_status = {}
        return {}

    # 1. 统一用 reranker 分数覆盖（仅作排序/质量参考，不再用于硬过滤）
    _rerank_evidences(state, reranker)
    # 1.5 rerank 后截断低分新增证据（控制 evaluator prompt 膨胀）
    _prune_new_evidences(state)

    # 2. 只评估「尚未 SUPPORTED」的 requirement（已 SUPPORTED 由单调性保留，
    #    无需重判，降低 LLM prompt 长度与判断负担）。
    #    综合型需求（synthetic）不做独立检索、也不由 LLM 评估：其状态由
    #    resolve_synthetic 依据依赖需求派生，避免 LLM 对其误判 MISSING 浪费判断。
    pending = [
        r for r in requirements
        if not r.synthetic
        and (state.requirement_status.get(r.id) is None
             or state.requirement_status[r.id].status != RequirementStatus.SUPPORTED)
    ]
    if not pending:
        state.derive_gaps()
        return state.requirement_status

    # 3. 证据池 = 全部 evidence（但历史证据在 prompt 里截短，见 _build_prompt）
    #    不只传新增：LLM 需看到全局证据才能判断 requirement 是否充分满足。
    #    只传新增会导致 LLM 永远判 PARTIAL（每轮只看 2~3 条，无法判断完整性）。
    relevant = state.evidences

    # 4. 完全没有证据 → 全 MISSING（不调 LLM）；有证据则调 LLM 判定
    if not relevant:
        new_status = {r.id: _all_missing([r])[r.id] for r in pending}
    else:
        if llm is None:
            llm = _get_evaluator_llm()
        if llm is None:
            new_status = {r.id: _all_missing([r])[r.id] for r in pending}
        else:
            new_status = _llm_evaluate(state.question, pending, relevant, llm, state=state)

    # 5. 合并：单调性约束 —— 状态只升不降
    #    SUPPORTED > PARTIAL > MISSING，新评估结果不得低于历史状态
    #    （evidence 只增不减，不应越搜越差。PARTIAL → MISSING 是之前 LLM 抖动
    #    导致空转的根因）
    _RANK = {RequirementStatus.SUPPORTED: 2, RequirementStatus.PARTIAL: 1, RequirementStatus.MISSING: 0}
    merged: Dict[str, RequirementStatusItem] = {}
    for r in requirements:
        # 综合型需求不参与 LLM 评估：保持既有状态（None 或 resolve_synthetic 写入的
        # SUPPORTED），避免被强制置 MISSING 污染 requirement_status。
        if r.synthetic:
            if r.id in state.requirement_status:
                merged[r.id] = state.requirement_status[r.id]
            continue
        old = state.requirement_status.get(r.id)
        new = new_status.get(r.id)
        if old is not None and new is not None:
            if _RANK[new.status] < _RANK[old.status]:
                # 新评估低于历史 → 保持历史（不降级）
                merged[r.id] = old
            else:
                merged[r.id] = new
        elif new is not None:
            merged[r.id] = new
        elif old is not None:
            merged[r.id] = old
        else:
            merged[r.id] = RequirementStatusItem(
                requirement_id=r.id, status=RequirementStatus.MISSING,
            )

    state.requirement_status = merged
    # 6. 派生 gaps（保持 state 一致；Gap 是派生态，Controller 直接读 state.gaps）
    state.derive_gaps()
    return merged
