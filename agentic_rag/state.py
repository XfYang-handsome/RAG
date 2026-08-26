# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — 显式状态定义（Phase 0 & 1）
================================================================================

这是整个 Agentic RAG 的**唯一事实源**。planner / evaluator / controller /
executor / stopping / synthesizer 全部围绕它读写。

核心设计（与架构定稿一致）：

  1. Requirement Status 与 Gap 分离：
     - requirement_status 由 Evaluator **唯一写入**（离散判断 SUPPORTED/PARTIAL/MISSING）
     - gaps 由 requirement_status **派生**（代码计算），LLM 绝不直接生成 Gap，
       避免 status 与 gap 相互矛盾。
  2. 所有浮点指标（coverage / quality / novelty）均由代码从结构化结果导出，
     LLM 只做离散判断，不直接打浮点分（浮点分噪声大、不可比、会让 Stop 乱停）。
  3. Evidence id 全局唯一且稳定（chunk_id / Milvus 主键），
     这是 novelty 差集计算与去重的基础。

用法（供后续 Phase 引用）：

    from agentic_rag.state import AgentState, Requirement, Action, ...

    state = AgentState(question="...")
    state.start_budget()
    state.evidences["E1"] = Evidence(id="E1", ...)
    state.requirement_status["R1"] = RequirementStatusItem(...)
    state.derive_gaps()
    state.compute_coverage()

================================================================================
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from .settings import get as _get


# ============================================================================
# Phase 0 — 枚举与常量（定界与命名）
# ============================================================================

class RequirementStatus(str, Enum):
    """Requirement 满足状态（Evaluator 的唯一离散输出，避免 ✓△✗ 符号歧义）。"""
    SUPPORTED = "SUPPORTED"   # 已有充分证据支撑
    PARTIAL    = "PARTIAL"    # 有部分证据，但不完整
    MISSING    = "MISSING"    # 完全没有证据


# 状态等级（用于单调性比较与「状态升级」判定）：rank 越大越「充分」
STATUS_RANK = {
    RequirementStatus.MISSING: 0,
    RequirementStatus.PARTIAL: 1,
    RequirementStatus.SUPPORTED: 2,
}


class ActionType(str, Enum):
    """Controller 可选 Action（V1 收敛到最小 6 个，DECOMPOSE / VERIFY 留 V2）。"""
    SEARCH       = "SEARCH"        # 检索新证据（走 Retrieval Router）
    REFINE_QUERY = "REFINE_QUERY"  # 重写查询（复用 rewrite_llm）
    READ_PARENT  = "READ_PARENT"   # 树上下文扩展：读父块
    READ_SECTION = "READ_SECTION"  # 树上下文扩展：读章节
    WEB_SEARCH   = "WEB_SEARCH"    # 知识库无新信息时联网搜索（受控，有次数上限）
    ANSWER       = "ANSWER"        # 终止循环，进入 Synthesis


class RetrievalTool(str, Enum):
    """检索工具（Retrieval Router 按规则选择，不由 LLM 自由选）。"""
    VECTOR = "vector"    # 纯向量（dense）
    HYBRID = "hybrid"    # 混合检索 dense + BM25 + RRF（默认）
    TREE   = "tree"      # 层级树检索（section_path 过滤）
    WEB    = "web"       # 联网搜索（非知识库，仅用于 Query 记录）


# 高重要度阈值：importance >= 该值时，Stopping 的 Sufficiency 判据要求其必须 SUPPORTED
IMPORTANCE_HIGH = _get("importance_high", 0.8)

# 默认预算（强制结束以轮数为准：最多 N 轮）
DEFAULT_MAX_ITERATIONS = _get("max_iterations", 5)
DEFAULT_MAX_TOOL_CALLS = _get("max_tool_calls", 12)
DEFAULT_MAX_LATENCY_MS = _get("max_latency_ms", 60_000)  # 已弃用：强制结束不再看时间，仅保留字段兼容


# ============================================================================
# Phase 1 — 值对象（不可变语义，字段即契约）
# ============================================================================

class Requirement(BaseModel):
    """Planner 产出的一个信息需求（最小充分拆解）。"""
    id: str
    description: str
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    # 枚举-展开（「有哪些 X + 它们的区别」类问题）：
    deferred: bool = False   # True=延迟对比需求：待 depends_on 解决后，从证据抽取真实清单动态展开
    depends_on: str = ""     # 依赖的枚举需求 id（deferred=True 时有意义）
    # 综合型需求（「A 与 B 的关系/联系/关联」类问题）：
    synthetic: bool = False  # True=综合归纳需求：不做独立检索，待依赖需求解决后由 Synthesis 归纳
    synthesize_from: List[str] = Field(default_factory=list)  # 依赖的需求 id 列表（synthetic=True 时有意义）


class EvidenceSource(BaseModel):
    """Evidence 的来源元数据（用于溯源 / 前端跳转 / 树上下文扩展）。"""
    doc_id: str = ""
    section_path: str = ""   # 章节路径字符串，如 "0/1"
    parent_id: str = ""
    chunk_id: str = ""
    summary: str = ""        # 所属章节的中文摘要（跨语言评估辅助：LLM 借中文摘要理解英文 chunk）
    origin: str = "kb"       # 来源类型："kb"（知识库）/ "web"（联网搜索）
    url: str = ""            # 联网来源的链接（origin="web" 时有值）


class Evidence(BaseModel):
    """一条被检索到的证据（跨轮累积，按 id 去重）。"""
    id: str = ""             # 稳定唯一 id（chunk_id / Milvus 主键），novelty 计算依据
    text: str = ""
    score: float = 0.0       # reranker / 检索分数（quality 指标来源）
    source: EvidenceSource = Field(default_factory=EvidenceSource)
    produced_by_round: int = 0


class RequirementStatusItem(BaseModel):
    """单个 requirement 的评估结果（Evaluator 唯一写这里）。"""
    requirement_id: str
    status: RequirementStatus = RequirementStatus.MISSING
    evidence_ids: List[str] = Field(default_factory=list)
    note: str = ""           # "缺什么" / "证据不足在哪"，派生 Gap 的 missing_what 来源


class Gap(BaseModel):
    """信息缺口（由 requirement_status 派生，LLM 不直接生成）。"""
    requirement_id: str
    missing_what: str = ""
    importance: float = 1.0


class Action(BaseModel):
    """Controller 的单个决策（必须指向某个 Gap）。"""
    type: ActionType
    target_gap: str = ""     # 指向某个 requirement_id（Gap→Action，而非 Question→Action）
    query: str = ""          # SEARCH / REFINE_QUERY 用
    tool: RetrievalTool = RetrievalTool.HYBRID
    produced_evidence_ids: List[str] = Field(default_factory=list)


class Query(BaseModel):
    """一次检索记录（去重 / 避免重复检索）。"""
    text: str
    tool: RetrievalTool = RetrievalTool.HYBRID
    round: int = 0


class Budget(BaseModel):
    """循环预算（Hard Stop 依据）。"""
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS
    started_at_ms: int = 0
    tool_calls: int = 0

    @property
    def elapsed_ms(self) -> int:
        if self.started_at_ms <= 0:
            return 0
        return int(time.time() * 1000) - self.started_at_ms


# ============================================================================
# Phase 1 — 主状态（AgentState）
# ============================================================================

class AgentState(BaseModel):
    """
    循环的唯一事实源。字段"谁写、谁读"约定：

      question           : 入口写，全程只读
      requirements       : Planner 写（V1 初期可为空/单条）
      evidences          : Executor 增量合并（add_evidence 去重）
      requirement_status : Evaluator 唯一写
      gaps               : derive_gaps() 派生（代码写，LLM 不碰）
      actions            : Executor 追加
      queries            : Executor 追加
      iteration          : Executor 自增
      no_progress_rounds : Executor 计算
      budget             : start_budget() 初始化，Executor 累加 tool_calls
    """

    question: str = ""
    requirements: List[Requirement] = Field(default_factory=list)
    evidences: Dict[str, Evidence] = Field(default_factory=dict)  # id -> Evidence，跨轮累积
    requirement_status: Dict[str, RequirementStatusItem] = Field(default_factory=dict)
    gaps: List[Gap] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    queries: List[Query] = Field(default_factory=list)
    iteration: int = 0
    no_progress_rounds: int = 0
    web_search_count: int = 0   # 已联网搜索次数（受 web_search.max_calls 限制，防无限联网）
    budget: Budget = Field(default_factory=Budget)
    # 已展开的 deferred 需求 id（枚举-展开轨迹，用于可观测 + 防重复展开）
    expanded_requirements: List[str] = Field(default_factory=list)

    # 上轮快照（内部工作状态）：snapshot_round() 在每轮开始时更新，
    # 用于计算「本轮新增 evidence / 新解决 requirement」，即 no-progress 判据依据
    last_round_evidence_ids: List[str] = Field(default_factory=list)
    last_round_supported_ids: List[str] = Field(default_factory=list)
    # 上轮 requirement 状态等级快照（requirement_id -> rank），用于「状态升级」判定
    last_round_status: Dict[str, int] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # 派生（代码计算，LLM 不写）
    # ------------------------------------------------------------------
    def derive_gaps(self) -> List[Gap]:
        """从 requirements 派生 gaps：status 非 SUPPORTED（含尚未评估）的项，importance 降序。

        必须遍历 requirements 而非 requirement_status：初始状态 requirement_status
        为空时，所有 requirement 都应视为 gap（尚未评估 = 尚无证据 = MISSING），
        否则 Controller 在首轮拿不到任何 gap 输入。
        """
        gaps: List[Gap] = []
        for r in self.requirements:
            # 综合型需求不做独立检索：知识库通常没有直接论述「A 与 B 关系」的段落，
            # 直接检索必然空转。它由依赖需求解决后、Synthesis 归纳，故不产生 gap。
            if r.synthetic:
                continue
            item = self.requirement_status.get(r.id)
            if item is not None and item.status == RequirementStatus.SUPPORTED:
                continue
            note = item.note if item else ""
            gaps.append(Gap(
                requirement_id=r.id,
                missing_what=note or f"{r.description} 仍缺少证据",
                importance=r.importance,
            ))
        gaps.sort(key=lambda g: g.importance, reverse=True)
        self.gaps = gaps
        return gaps

    def resolve_synthetic(self) -> List[str]:
        """把依赖已满足的综合型需求标记为 SUPPORTED（由 Synthesis 归纳，不做独立检索）。

        关系/联系类需求（synthetic=True）本身没有独立文档论述，只要它依赖的
        单点需求（synthesize_from）都已 SUPPORTED，即可由最终 Synthesis 归纳出
        二者关系，无需再检索。这里把该需求状态直接置为 SUPPORTED，从而：
          - coverage / high_importance_unsupported 正确反映「已可归纳」；
          - Sufficiency 判定得以触发，循环不再空转。

        Returns:
            本轮新变为 SUPPORTED 的 synthetic 需求 id 列表。
        """
        resolved: List[str] = []
        for r in self.requirements:
            if not r.synthetic:
                continue
            deps = r.synthesize_from or []
            if not deps:
                continue  # 无依赖可归纳（边界：应已被 planner 兜底取消 synthetic）
            if not all(
                self.requirement_status.get(d) is not None
                and self.requirement_status[d].status == RequirementStatus.SUPPORTED
                for d in deps
            ):
                continue
            cur = self.requirement_status.get(r.id)
            if cur is not None and cur.status == RequirementStatus.SUPPORTED:
                continue
            self.requirement_status[r.id] = RequirementStatusItem(
                requirement_id=r.id,
                status=RequirementStatus.SUPPORTED,
                evidence_ids=[],
                note="综合型需求：由依赖需求（" + ",".join(deps) + "）的证据归纳，无需独立检索",
            )
            resolved.append(r.id)
        return resolved

    def compute_coverage(self) -> float:
        """coverage = SUPPORTED 的 requirement 占比（可解释、可观测）。"""
        total = len(self.requirements)
        if total == 0:
            return 0.0
        supported = sum(
            1 for r in self.requirements
            if self.requirement_status.get(r.id) is not None
            and self.requirement_status[r.id].status == RequirementStatus.SUPPORTED
        )
        return supported / total

    def _status_rank(self, rid: str) -> int:
        """返回 requirement 当前状态的等级（未评估视为 MISSING=0）。"""
        item = self.requirement_status.get(rid)
        if item is None:
            return STATUS_RANK[RequirementStatus.MISSING]
        return STATUS_RANK.get(item.status, 0)

    def snapshot_round(self) -> None:
        """每轮开始时调用：保存本轮起点快照（evidence id + requirement 状态等级）。"""
        self.last_round_evidence_ids = list(self.evidences.keys())
        self.last_round_supported_ids = [
            r.id for r in self.requirements
            if self.requirement_status.get(r.id) is not None
            and self.requirement_status[r.id].status == RequirementStatus.SUPPORTED
        ]
        self.last_round_status = {
            r.id: self._status_rank(r.id) for r in self.requirements
        }

    def new_evidence_ids(self) -> List[str]:
        """本轮新增 evidence id 集 = 当前集 - 上轮快照集。"""
        return list(set(self.evidences.keys()) - set(self.last_round_evidence_ids))

    def newly_supported_ids(self) -> List[str]:
        """本轮新解决（新变为 SUPPORTED）的 requirement id 集。"""
        cur = {
            r.id for r in self.requirements
            if self.requirement_status.get(r.id) is not None
            and self.requirement_status[r.id].status == RequirementStatus.SUPPORTED
        }
        return list(cur - set(self.last_round_supported_ids))

    def newly_upgraded_ids(self) -> List[str]:
        """本轮状态升级（rank 提升）的 requirement id 集（Gap Reduction 判据）。

        与 newly_supported_ids 不同：这里检测**任意等级提升**
        （MISSING→PARTIAL / PARTIAL→SUPPORTED / MISSING→SUPPORTED），
        用于判定「本轮检索是否真正推进了需求满足」，而非「是否新增 evidence」。
        这是根治空转的关键：检索到一堆不相关证据（等级未提升）不算进展。
        """
        upgraded: List[str] = []
        for r in self.requirements:
            prev = self.last_round_status.get(r.id, STATUS_RANK[RequirementStatus.MISSING])
            if self._status_rank(r.id) > prev:
                upgraded.append(r.id)
        return upgraded

    def compute_novelty(self) -> int:
        """本轮新增 evidence 数（基于上轮快照，供 Stopping 的 no-progress 判据）。"""
        return len(self.new_evidence_ids())

    def high_importance_unsupported(self) -> List[Requirement]:
        """返回 importance >= IMPORTANCE_HIGH 且非 SUPPORTED 的 requirement（Stopping 判据）。"""
        result: List[Requirement] = []
        for r in self.requirements:
            if r.importance < IMPORTANCE_HIGH:
                continue
            item = self.requirement_status.get(r.id)
            if item is None or item.status != RequirementStatus.SUPPORTED:
                result.append(r)
        return result

    # ------------------------------------------------------------------
    # 更新（Executor 调用，顺序见 PLAN Phase 6）
    # ------------------------------------------------------------------
    def add_evidence(self, ev: Evidence) -> bool:
        """增量合并 evidence；返回 True 表示新增，False 表示已存在（去重）。"""
        if not ev.id:
            return False
        if ev.id in self.evidences:
            return False
        self.evidences[ev.id] = ev
        return True

    def register_action(self, action: Action) -> None:
        self.actions.append(action)
        # tool_calls 统计所有触发外部检索/读取的动作（SEARCH / REFINE_QUERY / READ_* / WEB_SEARCH）
        # REFINE_QUERY 执行时也会触发检索，必须计入，否则 max_tool_calls 上限形同虚设
        if action.type in (
            ActionType.SEARCH,
            ActionType.REFINE_QUERY,
            ActionType.READ_PARENT,
            ActionType.READ_SECTION,
            ActionType.WEB_SEARCH,
        ):
            self.budget.tool_calls += 1

    def can_web_search(self, max_calls: int = None) -> bool:
        """是否还能联网搜索（受 web_search.max_calls 限制，防止无限联网）。"""
        if max_calls is None:
            max_calls = _get("web_search.max_calls", 2)
        return self.web_search_count < int(max_calls)

    def add_query(self, q: Query) -> None:
        self.queries.append(q)

    def start_budget(self) -> None:
        """首次调用时记录起始时间戳（idempotent）。"""
        if self.budget.started_at_ms <= 0:
            self.budget.started_at_ms = int(time.time() * 1000)

    # ------------------------------------------------------------------
    # 可观测性
    # ------------------------------------------------------------------
    def trace_line(self) -> str:
        """生成单轮状态的可读摘要（前端/日志展示 Agent 决策轨迹）。"""
        parts = [
            f"第 {self.iteration} 轮",
            f"coverage={self.compute_coverage():.2f}",
            f"evidence={len(self.evidences)}",
            f"gaps={len(self.gaps)}",
        ]
        unsupported = self.high_importance_unsupported()
        if unsupported:
            parts.append("高重要度未解决: " + ",".join(r.id for r in unsupported))
        return " | ".join(parts)
