# -*- coding: utf-8 -*-
"""
================================================================================
通用树导航检索（Tree Retrieval）—— 纯树导航 + 文档路由 + 三级降级
================================================================================

把「纯树导航检索」从 agentic_rag 下沉为根目录通用能力，供：
  - 普通知识库问答（server.py 的 rag 模式，retrieval_mode=tree）
  - MCP 知识库工具（search_knowledge_base 的 tree 模式）
  - Agentic RAG（内部经薄包装转 Evidence 使用）

职责分工（纯树检索 = 不用向量召回做候选生成，但允许 reranker/LLM/关键词匹配）：
  - tree_store（SQLite）      → 结构存储 + 节点原文
  - 文档树 TreeNode           → 搜索空间
  - NodeState / TreeNavState  → 状态机（本文件）
  - LLM                       → 搜索策略（Policy），只提议 descend / backtrack
  - 本地 reranker             → Node Rerank（第一层 Gate）+ Leaf Rerank

统一返回结构（dict，与 db_service 检索结果风格一致，字段自包含）：
  {
    "id": 稳定去重 id,
    "text": 原文,
    "score": 相关度分,
    "doc_id": 文档 ID,
    "parent_id": 父节点 ID,
    "section_path": 章节路径字符串（如 "0/1"）,
    "summary": 所属章节摘要,
    "is_neighbor": 是否相邻补读,
  }

八条原则（见 AGENTIC_RAG_PLAN.md）：
    1. 单一 stack（栈顶 = current）
    2. NodeState 增加 exhausted
    3. is_searchable = status ∈ {unvisited, expanded}
    4. 叶子判断写死成节点类型（section=container，paragraph/table/figure=leaf）
    5. 叶子自动 Leaf Rerank → read，不让 LLM 决定 read
    6. Stop Policy 由代码触发
    7. LLM 提议动作，代码执行和裁决
    8. trajectory 保留完整搜索轨迹

参数语义：
    reranker=None → 跳过 Node/Leaf Rerank（仅测试用）
    llm=None      → 固定策略选第一个 searchable section
    max_* = None  → 读 config agentic.tree_nav.*（未配置时用内置兜底值）
================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tree_store

from common.text_utils import contains_cjk, parse_json, translate_to_en_keywords
from config_loader import cfg as _cfg

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置读取（等价 agentic_rag.settings.get：读 config agentic.<path>）
# ---------------------------------------------------------------------------
def _get(path: str, default=None):
    """读 ``agentic.<path>`` 配置，缺省时返回 default。

    与 agentic_rag.settings.get 语义一致，但不依赖 agentic 包。
    """
    return _cfg(f"agentic.{path}", default)


# ---------------------------------------------------------------------------
# 模型获取（复用 llm_factory，不依赖 agentic_rag.planner/executor）
# ---------------------------------------------------------------------------
def _get_router_llm():
    """文档路由/决策 LLM：tool_llm 优先，回退 llm。"""
    from llm_factory import get_model
    return get_model("tool_llm", "llm")


# ---------------------------------------------------------------------------
# 节点类型（叶子判断「写死成节点类型」，不用 if not children）
# ---------------------------------------------------------------------------
_CONTAINER_TYPES = {"section", "document"}


def is_container_kind(kind: str) -> bool:
    """判断节点类型是否为容器（section / document）。"""
    return (kind or "") in _CONTAINER_TYPES


def is_leaf_kind(kind: str) -> bool:
    """判断节点类型是否为叶子（paragraph/table/figure/... 等非容器类型）。"""
    return not is_container_kind(kind)


# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
@dataclass
class NodeState:
    """树导航过程中单个节点的状态。"""
    node_id: str
    kind: str = ""                  # section / paragraph / table / figure ...
    parent_id: str = ""
    doc_id: str = ""                # 所属文档（体量统计缓存按 doc 聚合用）
    depth: int = 0
    title: str = ""
    summary: str = ""
    score: float = 0.0              # Node Rerank 分（仅 section 用）
    scored: bool = False            # 是否已做 Node Rerank（显式标志，避免 score==0.0 误判）
    ambiguous: bool = False         # True=reranker 分数落在模糊区（不剪枝，交 LLM 兜底判断）
    status: str = "unvisited"       # unvisited / expanded / exhausted / read / pruned
    children: List[str] = field(default_factory=list)


@dataclass
class TreeNavState:
    """纯树导航的全局状态（单一控制中心）。"""
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    stack: List[str] = field(default_factory=list)          # 单一栈，栈顶 = current
    read_candidates: List[dict] = field(default_factory=list)  # 已读叶子（转统一 dict）
    pruned_candidates: List[dict] = field(default_factory=list)  # 被剪枝的叶子（空召回兜底抢救用）
    trajectory: List[dict] = field(default_factory=list)    # 完整搜索轨迹
    llm_calls: int = 0              # 已用 LLM 次数
    expansions: int = 0             # 已 expand 次数
    leaf_reads: int = 0             # 已读叶子数
    step: int = 0                   # 轨迹步号
    stats_cache: Dict[str, dict] = field(default_factory=dict)  # {doc_id: {node_id: 体量统计}}
    rep_cache: Dict[str, dict] = field(default_factory=dict)    # {doc_id: {node_id: 代表性叶子文本}}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _is_searchable(ns: NodeState) -> bool:
    """is_searchable = status ∈ {unvisited, expanded}（含未展开节点）。"""
    return ns.status in ("unvisited", "expanded")


def _make_ns(info: dict, depth: int = 0, parent_id: str = "") -> NodeState:
    return NodeState(
        node_id=info["node_id"],
        kind=info.get("type", ""),
        parent_id=parent_id or info.get("parent_node_id", ""),
        doc_id=info.get("doc_id", ""),
        depth=depth,
        title=info.get("title", "") or "",
        summary=info.get("summary", "") or "",
    )


def _subtree_stats(state: TreeNavState, node_id: str, doc_id: str) -> dict:
    """取节点子树体量（按 doc 惰性缓存，避免每次决策全量拉文档重复统计）。"""
    if doc_id and doc_id not in state.stats_cache:
        try:
            state.stats_cache[doc_id] = tree_store.get_subtree_stats_all(doc_id)
        except Exception:
            state.stats_cache[doc_id] = {}
    stats = state.stats_cache.get(doc_id, {}).get(node_id)
    if stats is not None:
        return stats
    # 缓存未命中（如全库检索未按 doc 预聚）：回退单点查询
    try:
        return tree_store.get_subtree_stats(node_id)
    except Exception:
        return {"section_count": 0, "leaf_count": 0, "max_depth": 0}


def _ensure_node(state: TreeNavState, node_id: str) -> Optional[NodeState]:
    """取节点状态，不在 nodes 里则用 tree_store.get_node 读取后创建。"""
    if node_id in state.nodes:
        return state.nodes[node_id]
    info = tree_store.get_node(node_id)
    if info is None:
        return None
    state.nodes[node_id] = _make_ns(info)
    return state.nodes[node_id]


def _traj(state: TreeNavState, node_id: str, action: str,
          candidates: Optional[List[str]] = None, reason: str = "",
          score: float = 0.0, llm_call: bool = False) -> None:
    """记录一条搜索轨迹（TrajStep）。"""
    state.step += 1
    state.trajectory.append({
        "step": state.step,
        "node_id": node_id,
        "action": action,
        "score": score,
        "candidates": candidates or [],
        "reason": reason,
        "llm_call": llm_call,
    })


def _expand(state: TreeNavState, ns: NodeState) -> None:
    """expand：读当前节点的直接子节点，填充 children（纯 SQLite）。"""
    children = tree_store.get_children(ns.node_id)
    for c in children:
        cid = c["node_id"]
        if cid not in state.nodes:
            state.nodes[cid] = _make_ns(c, depth=ns.depth + 1, parent_id=ns.node_id)
        ns.children.append(cid)
    state.expansions += 1


def _split_children(state: TreeNavState, ns: NodeState):
    """expand 后按节点类型分流：section 子节点 vs leaf 子节点。"""
    section_children: List[str] = []
    leaf_children: List[str] = []
    for cid in ns.children:
        cns = state.nodes[cid]
        if is_container_kind(cns.kind):
            section_children.append(cid)
        else:
            leaf_children.append(cid)
    return section_children, leaf_children


# ---------------------------------------------------------------------------
# Node Rerank（第一层 Gate）+ 剪枝
# ---------------------------------------------------------------------------
def _section_rep_text(state: TreeNavState, section_node_id: str, doc_id: str,
                      max_chars: int = 200) -> str:
    """取 section 的「代表性叶子文本」（第一个叶子的前 max_chars），按 doc 惰性缓存。

    优化：原 _section_representative_text 对每个候选 section 递归 get_children +
    get_node 找第一个叶子，N 个候选 = N 次递归 N+1 查询。改为一次性批量算
    全文档（tree_store.get_representative_texts_all），后续 O(1) 命中缓存。
    """
    if doc_id and doc_id not in state.rep_cache:
        try:
            state.rep_cache[doc_id] = tree_store.get_representative_texts_all(doc_id, max_chars)
        except Exception:
            state.rep_cache[doc_id] = {}
    cached = state.rep_cache.get(doc_id, {}).get(section_node_id)
    if cached is not None:
        return cached

    # 缓存未命中（如 doc_id 为空的全库检索）→ 回退递归
    children = tree_store.get_children(section_node_id)
    for c in children:
        if c.get("type") != "section":
            info = tree_store.get_node(c["node_id"])
            t = (info or {}).get("text", "") or ""
            if t.strip():
                return t[:max_chars]
    for c in children:
        if c.get("type") == "section":
            t = _section_rep_text(state, c["node_id"], doc_id, max_chars)
            if t:
                return t
    return ""


def _node_rerank(state: TreeNavState, query: str, reranker,
                 section_ids: List[str], node_min_score: float,
                 node_high_score: float = None) -> None:
    """对 section 子节点批量打分，按三区间分流。

    reranker=None 时跳过（不剪枝）。
    评分对象 = title + summary + 代表性叶子正文（前 200 字符）。

    三区间：
      - score < node_min_score              → 明确不相关，剪枝
      - node_min_score ≤ score < high_score → 模糊区，不剪枝，标记 ambiguous
      - score ≥ high_score                  → 明确相关，保留
    """
    if reranker is None:
        return
    to_score = [
        sid for sid in section_ids
        if not state.nodes[sid].scored and state.nodes[sid].status != "pruned"
    ]
    if not to_score:
        return

    texts = []
    for sid in to_score:
        ns = state.nodes[sid]
        t = f"{ns.title} {ns.summary}".strip()
        rep = _section_rep_text(state, sid, ns.doc_id)
        if rep:
            t = f"{t} {rep}".strip()
        texts.append(t if t else ns.title)

    try:
        res = reranker.rerank(query, texts)
        score_map = {r["index"]: float(r["score"]) for r in res}
        for i, sid in enumerate(to_score):
            state.nodes[sid].score = score_map.get(i, 0.5)
    except Exception:
        for sid in to_score:
            state.nodes[sid].score = 0.5

    # 标记已评分（显式标志，替代 score==0.0 的隐式判断）
    for sid in to_score:
        state.nodes[sid].scored = True

    for sid in to_score:
        ns = state.nodes[sid]
        if ns.score < node_min_score:
            ns.status = "pruned"
            _traj(state, sid, "prune", score=ns.score,
                  reason=f"Node Rerank 明确低分 {ns.score:.3f} < {node_min_score}")
        elif node_high_score is not None and ns.score < node_high_score:
            ns.ambiguous = True
            _traj(state, sid, "keep", score=ns.score,
                  reason=f"Node Rerank 模糊区 {ns.score:.3f} ∈ [{node_min_score}, {node_high_score})，标记 ambiguous 交 LLM 判断")


# ---------------------------------------------------------------------------
# Leaf Rerank → read / prune
# ---------------------------------------------------------------------------
def _dispose_leaf(state: TreeNavState, query: str, reranker, leaf_ns: NodeState,
                  info: dict, score: float, ambiguous: bool,
                  leaf_min_score: float, leaf_high_score: float,
                  neighbor_window: int) -> None:
    """叶子打分后的统一处置：read / prune + 相邻补读。

    info 为节点详情 dict（含 text/doc_id/section_path），由调用方传入，避免
    批量路径下每个叶子重复 get_node。reranker=None 时 score 无意义，不剪枝。
    """
    text = (info or {}).get("text", "") or ""

    # 明确低分 → 剪枝（仅 reranker 存在时）
    if reranker is not None and score < leaf_min_score:
        leaf_ns.status = "pruned"
        _traj(state, leaf_ns.node_id, "prune", score=score,
              reason=f"Leaf Rerank 明确低分 {score:.3f} < {leaf_min_score}")
        # 记录被剪叶子（供「空召回兜底」抢救：短文档叶子文本都短，
        # bge-reranker 会系统性打低分，全部剪枝导致 0 召回）
        state.pruned_candidates.append({
            "node_id": leaf_ns.node_id,
            "text": text,
            "doc_id": (info or {}).get("doc_id", ""),
            "section_path": (info or {}).get("section_path", []),
            "parent_id": leaf_ns.parent_id,
            "score": score,
        })
        return

    # 模糊区 → read 但标记 ambiguous（交下游 LLM 兜底）
    if reranker is not None and leaf_high_score is not None and score < leaf_high_score:
        ambiguous = True

    state.read_candidates.append({
        "node_id": leaf_ns.node_id,
        "text": text,
        "doc_id": (info or {}).get("doc_id", ""),
        "section_path": (info or {}).get("section_path", []),
        "parent_id": leaf_ns.parent_id,
        "score": score,
        "ambiguous": ambiguous,
        "is_neighbor": False,
    })
    leaf_ns.status = "read"
    leaf_ns.ambiguous = ambiguous
    leaf_ns.score = score
    state.leaf_reads += 1
    if ambiguous:
        _traj(state, leaf_ns.node_id, "read", score=score,
              reason=f"叶子读（模糊区 {score:.3f} ∈ [{leaf_min_score}, {leaf_high_score})，交下游 LLM 兜底）")
    else:
        _traj(state, leaf_ns.node_id, "read", score=score,
              reason="叶子自动读（Leaf Rerank 达标）" if reranker else "叶子自动读（无 reranker）")

    if neighbor_window and neighbor_window > 0:
        _read_neighbor_leaves(state, leaf_ns, neighbor_window)


def _leaf_rerank_batch(state: TreeNavState, query: str, reranker,
                       leaf_ids: List[str], leaf_min_score: float,
                       leaf_high_score: float = None,
                       neighbor_window: int = None) -> None:
    """对一批叶子批量打分并处置（read / prune + 相邻补读）。

    核心优化：cross-encoder 批量推理替代逐个 batch=1。GPU 上 batch=1 利用率极低，
    一个 section 展开出的 N 个叶子合并成一次 rerank 调用，推理耗时下降一个数量级。
    reranker=None 时跳过打分直接逐个 read。
    """
    # 过滤掉已处理（非 unvisited）的叶子
    pending = [lid for lid in leaf_ids if state.nodes[lid].status == "unvisited"]
    if not pending:
        return

    # 先一次性拉取所有叶子的节点详情（含 text），供打分与处置复用
    infos: Dict[str, dict] = {}
    for lid in pending:
        infos[lid] = tree_store.get_node(lid) or {}

    if reranker is None:
        # 无 reranker：跳过打分，直接逐个 read（等价旧逻辑）
        for lid in pending:
            _dispose_leaf(state, query, None, state.nodes[lid], infos[lid],
                          0.0, False, leaf_min_score, leaf_high_score, neighbor_window)
        return

    # 批量打分
    to_score = [lid for lid in pending if (infos[lid].get("text") or "").strip()]
    score_map: Dict[int, float] = {}
    if to_score:
        texts = [infos[lid].get("text", "") or "" for lid in to_score]
        try:
            res = reranker.rerank(query, texts)
            score_map = {r["index"]: float(r["score"]) for r in res}
        except Exception:
            score_map = {i: 0.5 for i in range(len(to_score))}

    for i, lid in enumerate(to_score):
        score = score_map.get(i, 0.5)
        ambiguous = leaf_high_score is not None and score < leaf_high_score
        _dispose_leaf(state, query, reranker, state.nodes[lid], infos[lid],
                      score, ambiguous, leaf_min_score, leaf_high_score, neighbor_window)

    # 空文本叶子（罕见）：reranker 打不了分，跳过打分直接 read
    for lid in pending:
        if lid in infos and (infos[lid].get("text") or "").strip():
            continue  # 已在 to_score 里处理
        _dispose_leaf(state, query, reranker, state.nodes[lid], infos[lid],
                      0.0, False, leaf_min_score, leaf_high_score, neighbor_window)


def _leaf_rerank_read(state: TreeNavState, query: str, reranker,
                      leaf_ns: NodeState, leaf_min_score: float,
                      leaf_high_score: float = None,
                      neighbor_window: int = None) -> None:
    """单叶子打分入口（仅防御分支：stack 顶意外出现叶子时用）。

    正常路径走 _leaf_rerank_batch（批量），本函数保留给主循环的叶子防御分支。
    """
    if leaf_ns.status != "unvisited":
        return
    info = tree_store.get_node(leaf_ns.node_id) or {}
    text = info.get("text", "") or ""
    score = 0.0
    ambiguous = False
    if reranker is not None and text:
        try:
            res = reranker.rerank(query, [text], top_n=1)
            score = float(res[0]["score"]) if res else 0.0
        except Exception:
            score = 0.5
        if leaf_high_score is not None and score < leaf_high_score:
            ambiguous = True
    _dispose_leaf(state, query, reranker, leaf_ns, info, score, ambiguous,
                  leaf_min_score, leaf_high_score, neighbor_window)


def _read_neighbor_leaves(state: TreeNavState, leaf_ns: NodeState, window: int) -> None:
    """补读命中叶子同父的相邻叶子（跨节点答案补全）。

    补读叶子不 rerank（作为上下文补全），标记 is_neighbor=True，score 记 0.0
    （相邻叶子不是独立命中，不应继承命中叶子的分数误导下游排序）。
    Stop Policy 不把相邻叶子算作「高分证据」。
    """
    parent_id = leaf_ns.parent_id
    if not parent_id:
        return
    siblings = tree_store.get_children(parent_id)
    leaf_siblings = [c for c in siblings if c.get("type") != "section"]
    if not leaf_siblings:
        return
    ids = [c["node_id"] for c in leaf_siblings]
    if leaf_ns.node_id not in ids:
        return
    idx = ids.index(leaf_ns.node_id)
    lo = max(0, idx - window)
    hi = min(len(ids), idx + window + 1)

    for nb_id in ids[lo:hi]:
        if nb_id == leaf_ns.node_id:
            continue
        if nb_id in state.nodes and state.nodes[nb_id].status != "unvisited":
            continue
        info = tree_store.get_node(nb_id)
        if info is None:
            continue
        text = (info or {}).get("text", "") or ""
        if not text:
            continue
        if nb_id not in state.nodes:
            state.nodes[nb_id] = _make_ns(info, depth=leaf_ns.depth, parent_id=parent_id)
        nb_ns = state.nodes[nb_id]
        state.read_candidates.append({
            "node_id": nb_id,
            "text": text,
            "doc_id": (info or {}).get("doc_id", ""),
            "section_path": (info or {}).get("section_path", []),
            "parent_id": parent_id,
            "score": 0.0,
            "ambiguous": False,
            "is_neighbor": True,
        })
        nb_ns.status = "read"
        state.leaf_reads += 1
        _traj(state, nb_id, "read", score=0.0,
              reason=f"相邻叶子补读（同父，窗口 ±{window}）")


# ---------------------------------------------------------------------------
# read 候选 → 统一结果 dict
# ---------------------------------------------------------------------------
def _to_result(cand: dict, summary_cache: Dict[str, str] = None) -> dict:
    """把 read 候选 dict 转成统一结果 dict（稳定 id + section_path 字符串 + 父章节摘要）。

    summary_cache：可选父章节摘要缓存。相邻补读叶子共享同一 parent，命中叶子与
    相邻叶子同父，若逐条 get_node 会重复查询同一 parent 的 summary。传入缓存可
    把 parent→summary 的查询次数降到「去重后的 parent 数」。
    """
    sp = cand.get("section_path") or []
    section_path_str = "/".join(str(x) for x in sp) if sp else ""
    parent_id = cand.get("parent_id") or ""
    summary = ""
    if parent_id:
        if summary_cache is not None:
            if parent_id not in summary_cache:
                pinfo = tree_store.get_node(parent_id)
                summary_cache[parent_id] = (pinfo or {}).get("summary", "") or ""
            summary = summary_cache[parent_id]
        else:
            pinfo = tree_store.get_node(parent_id)
            summary = (pinfo or {}).get("summary", "") or ""
    node_id = cand.get("node_id") or ""
    eid = f"tree:{node_id}"
    return {
        "id": eid,
        "text": cand.get("text") or "",
        "score": float(cand.get("score", 0.0) or 0.0),
        "doc_id": cand.get("doc_id") or "",
        "section_path": section_path_str,
        "parent_id": parent_id,
        "chunk_id": eid,
        "summary": summary,
        "is_neighbor": bool(cand.get("is_neighbor", False)),
    }


# ---------------------------------------------------------------------------
# LLM Policy（只 descend / backtrack）
# ---------------------------------------------------------------------------
def _choose_next(state: TreeNavState, query: str, llm,
                 current_ns: NodeState, searchable: List[str],
                 max_llm_calls: int, status_callback=None,
                 node_high_score: float = None) -> Optional[str]:
    """从 searchable 中选下一个 descend 目标。

    - llm=None：固定策略选第一个 searchable。
    - searchable 仅 1 个：确定性 descend，跳过 LLM（唯一候选无需决策，省一次调用）。
    - reranker 唯一明确高分：searchable 中只有一个 score ≥ node_high_score 且非模糊
      的章节（其余都是模糊区），确定性 descend 到它，跳过 LLM。
    - llm 非 None：LLM 决策，输出 {"action":"descend","choice":N} 或 {"action":"backtrack"}。
      返回 None 表示 backtrack；非法输出保守回退为 backtrack。
    """
    if llm is None or len(searchable) == 1:
        return searchable[0]

    # 唯一明确高分 → 确定性 descend（reranker 已明确，无需 LLM 再判断）
    if node_high_score is not None:
        high = [sid for sid in searchable
                if not state.nodes[sid].ambiguous
                and state.nodes[sid].scored
                and state.nodes[sid].score >= node_high_score]
        if len(high) == 1:
            _traj(state, current_ns.node_id, "descend", candidates=[high[0]],
                  reason=f"reranker 唯一明确高分 {state.nodes[high[0]].score:.3f} ≥ {node_high_score}，跳过 LLM")
            return high[0]

    choice_map: Dict[str, str] = {}
    lines = []
    for i, sid in enumerate(searchable, 1):
        ns = state.nodes[sid]
        choice_map[str(i)] = sid
        title = ns.title or "(无标题)"
        summ = (ns.summary or "")[:60]
        score_str = f"[{ns.score:.2f}]" if ns.score else ""
        flag = " [模糊]" if ns.ambiguous else ""
        stats = _subtree_stats(state, sid, ns.doc_id)
        vol = (f"(子章节{stats['section_count']} 叶子{stats['leaf_count']} "
               f"深{stats['max_depth']})")
        lines.append(f"{i}. {title} {score_str}{flag} {vol} — {summ}".strip())

    recent = state.read_candidates[-3:]
    ev_lines = []
    for c in recent:
        t = (c.get("text") or "").replace("\n", " ")[:80]
        ev_lines.append(f"- {t}")
    ev_block = "\n".join(ev_lines) if ev_lines else "（暂无）"

    prompt = (
        f"用户问题：{query}\n"
        f"当前章节：{current_ns.title or '(根)'}"
        + (f" — {current_ns.summary[:60]}" if current_ns.summary else "") + "\n\n"
        f"候选子章节（编号 / 标题 / 相关度分 / 体量 / 摘要）：\n" + "\n".join(lines) + "\n\n"
        f"已找到的证据摘要：\n{ev_block}\n\n"
        f"说明：\n"
        f"1. 候选行括号内是章节体量：「子章节」=该章节下有几层子章节节点，"
        f"「叶子」=该章节下有多少个正文叶子（段落/表格等），"
        f"「深」=往下最多还有几层子章节（深 0 表示下面只剩叶子、无子章节可再深入）。\n"
        f"2. 带「[模糊]」标记的候选，其相关度分来自语义相似度小模型（reranker），"
        f"分数不可靠，请根据标题与摘要的实际语义仔细判断是否值得深入，不要被分数误导。\n\n"
        f"请决定下一步：深入某个候选子章节（descend），或返回上级（backtrack）。\n"
        f'只输出 JSON：{{"action":"descend","choice":1}} 或 {{"action":"backtrack"}}，不要任何解释。'
    )

    try:
        from llm import invoke_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        state.llm_calls += 1
        if status_callback:
            status_callback(f"树导航决策（第 {state.llm_calls} 次 LLM）：{len(searchable)} 个候选")

        text = invoke_llm(llm, [
            SystemMessage(content="你是文档树导航器，决定深入哪个子章节或返回上级。只输出 JSON。"),
            HumanMessage(content=prompt),
        ])
        obj = parse_json(text)
        if obj is None:
            _traj(state, current_ns.node_id, "backtrack", reason="LLM 输出无法解析，保守回退")
            return None

        action = str(obj.get("action", "")).strip().lower()
        if action == "backtrack":
            _traj(state, current_ns.node_id, "backtrack", reason="LLM 决策 backtrack",
                  llm_call=True)
            return None

        if action == "descend":
            choice = str(obj.get("choice", "")).strip()
            target = choice_map.get(choice)
            if target is not None:
                return target
            _traj(state, current_ns.node_id, "backtrack",
                  reason=f"LLM choice={choice!r} 非法，保守回退", llm_call=True)
            return None

        _traj(state, current_ns.node_id, "backtrack",
              reason=f"LLM action={action!r} 未知，保守回退", llm_call=True)
        return None
    except Exception as e:
        _traj(state, current_ns.node_id, "backtrack",
              reason=f"LLM 调用异常 {type(e).__name__}，保守回退", llm_call=True)
        return None


# ---------------------------------------------------------------------------
# Stop Policy（代码触发）
# ---------------------------------------------------------------------------
def _check_stop(state: TreeNavState, min_evidences: int,
                leaf_min_score: float, reranker,
                leaf_high_score: float = None) -> Optional[str]:
    """判断是否该停止（返回停止原因；None=继续）。

    只认「非模糊 + 非相邻补读 + 分数 ≥ leaf_high_score」的叶子为高分证据。
    """
    high_threshold = leaf_high_score if leaf_high_score is not None else leaf_min_score
    if reranker is None:
        high = [c for c in state.read_candidates if not c.get("is_neighbor")]
    else:
        high = [c for c in state.read_candidates
                if not c.get("ambiguous")
                and not c.get("is_neighbor")
                and float(c.get("score", 0.0) or 0.0) >= high_threshold]
    if len(high) >= min_evidences:
        return f"证据足够：{len(high)} 条高分证据 >= {min_evidences}"
    return None


# ---------------------------------------------------------------------------
# 初始化 entries（优先起点，非强制入口）
# ---------------------------------------------------------------------------
def _llm_select_entries(query: str, llm, doc_id: Optional[str] = None) -> List[str]:
    """LLM 语义入口选择（match_sections 关键词无命中时的兜底）。

    返回顶层 section 的 node_id 作为搜索起点；失败返回空列表。
    """
    tops = tree_store.list_top_sections(doc_id)
    if not tops or llm is None:
        return []

    lines = []
    for i, s in enumerate(tops, 1):
        title = s["title"] or "(无标题)"
        summ = (s["summary"] or "")[:60]
        line = f"{i}. {title}"
        if summ:
            line += f" — {summ}"
        lines.append(line)
    section_text = "\n".join(lines)

    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是文档章节路由助手。请根据用户问题，从给定的顶层章节列表中选出"
        "最可能包含答案的章节（最多 3 个）。严格只输出 JSON："
        '{"sections": [1, 3]}，编号从 1 开始，按相关度从高到低，不要任何解释。'
    )
    prompt = (
        f"用户问题：{query}\n\n"
        f"顶层章节列表（编号 / 标题 / 摘要）：\n{section_text}\n\n"
        f"请选出与问题最相关的章节编号。"
    )

    try:
        text = invoke_llm(llm, [SystemMessage(content=system),
                                HumanMessage(content=prompt)]).strip()
        obj = parse_json(text)
        if obj is None:
            return []
        indices = obj.get("sections") or []
    except Exception:
        return []

    entries: List[str] = []
    seen = set()
    for idx in indices:
        try:
            i = int(idx)
        except (ValueError, TypeError):
            continue
        if 1 <= i <= len(tops):
            nid = tops[i - 1]["node_id"]
            if nid and nid not in seen:
                seen.add(nid)
                entries.append(nid)
    return entries


def _init_entries(query: str, doc_id: Optional[str] = None, llm=None,
                  match_queries: Optional[List[str]] = None) -> List[str]:
    """确定搜索起点。

    - doc_id 指定 → 只在该文档内检索（文档级路由后的单文档树检索）
    - 未指定 doc_id（全库检索）：
      * match_queries 关键词匹配标题（原文优先 + 英化兜底，合并去重）→ 命中 section
        本身 + 顶层祖先作为起点
      * 无命中 → LLM 语义入口选择（用原始 query，兜底关键词失效）
      * 仍无 → 所有文档的顶层 section（从 root 宽搜）
    """
    if doc_id:
        entries: List[str] = []
        root_id = tree_store.get_document_root_id(doc_id)
        if root_id:
            for c in tree_store.get_children(root_id):
                if c.get("type") == "section":
                    entries.append(c["node_id"])
        return entries

    if match_queries is None:
        match_queries = [query]

    # 按查询序列依次匹配标题（原文优先、英化兜底），跨查询合并去重。
    # 修复：中文专有名词英化可能被 LLM 错译（如「幽境危战」→「Imaginarium Theater」），
    # 只靠英化 query 会漏召回；保留原文匹配一次，再叠加英化结果，专名不丢失。
    matched: List[dict] = []
    seen_matched: set = set()
    for q in match_queries:
        for m in tree_store.match_sections(q, limit=8):
            if m["node_id"] not in seen_matched:
                seen_matched.add(m["node_id"])
                matched.append(m)
    if matched:
        entries: List[str] = []
        seen = set()
        # ① 命中的 section 本身（精确命中，直接 descend 到命中处，避免从根逐层重走）
        for m in matched:
            nid = m["node_id"]
            if nid and nid not in seen:
                seen.add(nid)
                entries.append(nid)
        # ② 顶层祖先（保留 descend/backtrack 空间，覆盖命中 section 的兄弟分支）
        for m in matched:
            top = tree_store.get_top_ancestor(m["node_id"])
            if top and top not in seen:
                seen.add(top)
                entries.append(top)
        if entries:
            return entries

    llm_entries = _llm_select_entries(query, llm, doc_id=None)
    if llm_entries:
        return llm_entries

    entries = []
    for doc in tree_store.list_documents():
        root_id = tree_store.get_document_root_id(doc["doc_id"])
        if not root_id:
            continue
        for c in tree_store.get_children(root_id):
            if c.get("type") == "section":
                entries.append(c["node_id"])
    return entries


# ---------------------------------------------------------------------------
# 主循环（单一控制中心）
# ---------------------------------------------------------------------------
def tree_navigate(
    query: str,
    reranker=None,
    llm=None,
    doc_id: Optional[str] = None,
    node_min_score: float = None,
    node_high_score: float = None,
    leaf_min_score: float = None,
    leaf_high_score: float = None,
    max_depth: int = None,
    max_expansions: int = None,
    max_llm_calls: int = None,
    max_leaf_reads: int = None,
    min_evidences: int = None,
    neighbor_window: int = None,
    status_callback=None,
) -> dict:
    """纯树导航检索入口（完整状态机）。

    Returns:
        {"evidences": [统一结果 dict], "trajectory": [...], "stats": {...}}
    """
    node_min_score = _get("tree_nav.node_min_score", 0.2) if node_min_score is None else node_min_score
    node_high_score = _get("tree_nav.node_high_score", 0.5) if node_high_score is None else node_high_score
    leaf_min_score = _get("tree_nav.leaf_min_score", 0.2) if leaf_min_score is None else leaf_min_score
    leaf_high_score = _get("tree_nav.leaf_high_score", 0.5) if leaf_high_score is None else leaf_high_score
    max_depth = _get("tree_nav.max_depth", 4) if max_depth is None else max_depth
    max_expansions = _get("tree_nav.max_expansions", 8) if max_expansions is None else max_expansions
    max_llm_calls = _get("tree_nav.max_llm_calls", 6) if max_llm_calls is None else max_llm_calls
    max_leaf_reads = _get("tree_nav.max_leaf_reads", 20) if max_leaf_reads is None else max_leaf_reads
    min_evidences = _get("tree_nav.min_evidences", 2) if min_evidences is None else min_evidences
    neighbor_window = _get("tree_nav.neighbor_window", 1) if neighbor_window is None else neighbor_window

    state = TreeNavState()

    # 构造 match_sections 的查询序列：中文原文优先，英化 query 兜底。
    # 修复：中文专有名词英化可能被 LLM 错译（如「幽境危战」→「Imaginarium Theater」），
    # 先保留原文匹配一次，再叠加英化结果，避免专名错译导致漏召回。
    match_queries: List[str] = [query]
    if contains_cjk(query):
        translated = translate_to_en_keywords(query, llm)
        if translated and translated != query and translated not in match_queries:
            match_queries.append(translated)

    # reranker 打分与 LLM 决策用原始 query（bge-reranker-v2-m3 跨语言能力强）
    rerank_query = query

    # 初始化 entries（match_sections 用「原文 + 英化」序列；doc_id 指定则单文档），逆序压栈
    entries = _init_entries(query, doc_id=doc_id, llm=llm, match_queries=match_queries)
    for e in reversed(entries):
        state.stack.append(e)

    # 主循环：栈顶 = current
    while state.stack:
        current_id = state.stack[-1]
        ns = _ensure_node(state, current_id)
        if ns is None:
            state.stack.pop()
            continue

        # ---- 叶子（防御：理论上不入栈，若出现则自动 Leaf Rerank）----
        if is_leaf_kind(ns.kind):
            _leaf_rerank_read(state, rerank_query, reranker, ns, leaf_min_score, leaf_high_score, neighbor_window)
            state.stack.pop()
            continue

        # ---- 容器节点（section）----
        if ns.status == "unvisited":
            _expand(state, ns)
            ns.status = "expanded"
            _traj(state, ns.node_id, "expand",
                  reason=f"展开 {len(ns.children)} 个子节点")

        section_children, leaf_children = _split_children(state, ns)

        # ① leaf 子节点：批量 Leaf Rerank → read / prune（一次 rerank 调用打一批）
        _leaf_rerank_batch(state, rerank_query, reranker, leaf_children,
                           leaf_min_score, leaf_high_score, neighbor_window)

        # ② section 子节点：Node Rerank（第一层 Gate，三区间）+ 剪枝
        _node_rerank(state, rerank_query, reranker, section_children, node_min_score, node_high_score)

        searchable = [c for c in section_children if _is_searchable(state.nodes[c])]

        if not searchable:
            ns.status = "exhausted"
            _traj(state, ns.node_id, "backtrack",
                  reason="无可探索 section 子节点，标记 exhausted")
            state.stack.pop()
            continue

        # ---- Stop Policy（代码每轮自动检查，LLM 不决定）----
        stop_reason = _check_stop(state, min_evidences, leaf_min_score, reranker, leaf_high_score)
        if stop_reason:
            _traj(state, ns.node_id, "stop", reason=stop_reason)
            break

        # ---- 预算检查（硬停止）----
        if state.expansions >= max_expansions:
            _traj(state, ns.node_id, "stop", reason=f"expand 次数达上限 {max_expansions}")
            break
        if state.llm_calls >= max_llm_calls:
            _traj(state, ns.node_id, "stop", reason=f"LLM 调用达上限 {max_llm_calls}")
            break
        if state.leaf_reads >= max_leaf_reads:
            _traj(state, ns.node_id, "stop", reason=f"读叶子数达上限 {max_leaf_reads}")
            break

        # ---- LLM Policy（只 descend / backtrack，用原始 query）----
        target = _choose_next(state, rerank_query, llm, ns, searchable, max_llm_calls,
                              status_callback, node_high_score)
        if target is None:
            ns.status = "exhausted"
            state.stack.pop()
            continue

        # 深度检查：超过 max_depth 则不再 descend
        if state.nodes[target].depth >= max_depth:
            _traj(state, ns.node_id, "backtrack",
                  reason=f"目标深度 {state.nodes[target].depth} >= 上限 {max_depth}")
            ns.status = "exhausted"
            state.stack.pop()
            continue

        _traj(state, ns.node_id, "descend", candidates=searchable,
              reason="LLM 决策 descend" if llm else "固定策略选第一个 section",
              llm_call=bool(llm))
        state.stack.append(target)

    # 空召回兜底：短文档的叶子文本都短，bge-reranker-v2-m3（交叉编码器）对超短
    # 文本的 logits 系统性偏低，导致所有叶子 score < leaf_min_score 被剪枝 →
    # 树导航 0 召回（明明文档有相关内容）。此时抢救分数最高的 top-K 个被剪叶子，
    # 标记 ambiguous 交下游 LLM 兜底，避免短文档「明明有内容却全军覆没」。
    if not state.read_candidates and state.pruned_candidates and reranker is not None:
        empty_recall_top_k = int(_get("tree_nav.empty_recall_top_k", 3))
        pruned = sorted(state.pruned_candidates,
                        key=lambda c: -float(c.get("score", 0.0) or 0.0))
        for c in pruned[:empty_recall_top_k]:
            state.read_candidates.append({
                "node_id": c["node_id"],
                "text": c["text"],
                "doc_id": c.get("doc_id", ""),
                "section_path": c.get("section_path", []),
                "parent_id": c.get("parent_id", ""),
                "score": float(c.get("score", 0.0) or 0.0),
                "ambiguous": True,   # 兜底抢救，标记模糊，不算高分证据
                "is_neighbor": False,
            })
        _traj(state, "", "rescue",
              reason=f"空召回兜底：抢救 {len(state.read_candidates)} 条被剪叶子（短文本 rerank 低分）")

    # 结果按相关度排序：非相邻补读在前、按 score 降序；相邻补读（score=0）恒在后。
    # 下游（agentic 评估 / rag 生成）若按顺序取 top_k，能拿到最相关的证据。
    summary_cache: Dict[str, str] = {}  # parent_id → summary（去重查询，消除 N+1）
    evidences = [_to_result(c, summary_cache) for c in state.read_candidates]
    evidences.sort(key=lambda e: (bool(e.get("is_neighbor")), -float(e.get("score", 0.0) or 0.0)))

    return {
        "evidences": evidences,
        "trajectory": state.trajectory,
        "stats": {
            "visited": len(state.nodes),
            "read_leafs": len(state.read_candidates),
            "llm_calls": state.llm_calls,
            "expansions": state.expansions,
            "steps": state.step,
        },
    }


# ---------------------------------------------------------------------------
# 文档级路由（Doc Router）
# ---------------------------------------------------------------------------
def _parse_abstract(abstract: str) -> str:
    """把 documents.abstract（JSON 字符串）解析成可读文本。"""
    if not abstract:
        return ""
    try:
        obj = json.loads(abstract)
        parts = []
        if isinstance(obj, dict):
            if obj.get("abstract"):
                parts.append(str(obj["abstract"]))
            kws = obj.get("keywords") or []
            if kws:
                parts.append("关键词：" + "、".join(str(k) for k in kws))
            return "；".join(parts)
    except Exception:
        pass
    return abstract


def list_doc_cards() -> List[dict]:
    """列出所有文档的「卡片」（doc_id + 标题 + 主旨 + 关键词），供路由判断。"""
    cards = []
    for d in tree_store.list_documents():
        cards.append({
            "doc_id": d.get("doc_id") or "",
            "title": d.get("title") or "",
            "abstract": _parse_abstract(d.get("abstract") or ""),
            "node_count": d.get("node_count") or 0,
        })
    return cards


_ROUTE_SYSTEM = (
    "你是知识库文档路由助手。请根据用户问题，从给定的文档卡片中选出"
    "最可能包含答案的文档。严格只输出 JSON："
    '{"docs": ["doc_id1", "doc_id2"]}，按相关度从高到低排序，不要任何解释。'
)


def route_to_docs(question: str, llm=None, top_n: int = 1) -> List[str]:
    """基于文档卡片，用 LLM 选出与问题最相关的 top-N 个 doc_id。"""
    cards = list_doc_cards()
    if not cards:
        return []
    if len(cards) == 1:
        return [cards[0]["doc_id"]]

    if llm is None:
        llm = _get_router_llm()
    if llm is None:
        return [cards[0]["doc_id"]]

    card_lines = []
    for i, c in enumerate(cards, 1):
        line = f"{i}. doc_id={c['doc_id']}\n   标题：{c['title'] or '（无标题）'}"
        if c["abstract"]:
            line += f"\n   主旨：{c['abstract']}"
        card_lines.append(line)
    cards_text = "\n\n".join(card_lines)

    prompt = (
        f"用户问题：{question}\n\n"
        f"知识库文档卡片：\n{cards_text}\n\n"
        f"请选出与问题最相关的文档（最多 {top_n} 篇），按相关度从高到低输出其 doc_id。"
    )

    try:
        from llm import invoke_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        text = invoke_llm(llm, [SystemMessage(content=_ROUTE_SYSTEM),
                                HumanMessage(content=prompt)]).strip()
        obj = parse_json(text)
        if obj is None:
            return [cards[0]["doc_id"]]
        docs = obj.get("docs") or []
        valid_ids = {c["doc_id"] for c in cards}
        result = []
        for did in docs:
            did = str(did).strip()
            if did in valid_ids and did not in result:
                result.append(did)
        return result[:top_n] or [cards[0]["doc_id"]]
    except Exception:
        return [cards[0]["doc_id"]]


def _merge_evidences(evs: List[dict], new_evs: List[dict]) -> List[dict]:
    """按 id 去重合并两组证据，返回原 evs 列表（原地追加）。"""
    seen = {e["id"] for e in evs}
    for e in new_evs or []:
        if e.get("id") not in seen:
            seen.add(e["id"])
            evs.append(e)
    return evs


def build_coverage_evidence(evs: List[dict], top_k: int = 5) -> List[dict]:
    """从完整证据列表构建「轻量知识覆盖集合」，供 DocNovelty 判断用。

    与最终 Evidence 分离的原因（方案 C 核心）：最终 Evidence 会被 top_k 截断，
    而 DocNovelty 需要「完整覆盖集合」作判断基准——否则 D1 搜到的后几条知识
    不在最终 top_k 里，会导致 D2 被误判 uncovered → 重复搜索。

    规则：
      1. 排除相邻补读（is_neighbor=True，它们只是上下文，不是独立命中）
      2. 按 score 降序取前 top_k 条
      3. 每条只留三字段：path(主题) + summary(讲了什么) + text[:150](细节)
    """
    # 1. 排除相邻补读
    cands = [e for e in (evs or []) if not e.get("is_neighbor")]
    if not cands:
        return []
    # 2. 按 score 降序取前 top_k
    cands = sorted(cands, key=lambda e: -float(e.get("score", 0.0) or 0.0))[:top_k]
    # 3. 只留三字段
    out = []
    for e in cands:
        path = (e.get("section_path") or "").strip()
        summary = (e.get("summary") or "").strip()
        text = (e.get("text") or "")[:150].strip()
        out.append({"path": path, "summary": summary, "text": text})
    return out


def judge_doc_novelty(card: dict, coverage_evidence: List[dict], reranker,
                      coverage_high: float = 0.55) -> str:
    """DocNovelty：候选文档相对当前证据的信息重复度判断（V1 只二分）。

    返回 "covered"（跳过）或 "uncovered"（继续搜）。

    判断方向必须是 D → E（文档主旨 → 已有证据），不能用 question 当 query：
    因为 D 和 E 都是被 question 筛出来的，二者天然都与 question 高相关，
    无法区分「D 的独特贡献是否已被上一篇覆盖」。

    一期策略「宁可多搜、绝不误跳」：所有不确定 / 缺失的边界一律判 uncovered，
    只有 reranker 明确给出「高度覆盖」才判 covered。
    """
    # 边界：无 reranker / 无证据 / 无卡片 → 一律 uncovered（保守搜）
    if reranker is None or not coverage_evidence:
        return "uncovered"

    # 构造 query = D 的主旨 + 关键词（card 的 abstract 已是 _parse_abstract 后的可读文本）
    query = (card.get("abstract") or "").strip() if card else ""
    if not query:
        query = (card.get("title") or "").strip()
    if not query:
        return "uncovered"  # 主旨与标题都空 → 保守搜

    # documents = coverage_evidence 的文本（path + summary + text 拼接）
    docs = []
    for c in coverage_evidence:
        joined = f"{c.get('path', '')} {c.get('summary', '')} {c.get('text', '')}".strip()
        if joined:
            docs.append(joined)
    if not docs:
        return "uncovered"

    # 批量 rerank，取 max score
    try:
        res = reranker.rerank(query, docs)
        max_score = max((float(r["score"]) for r in res), default=0.0)
    except Exception:
        return "uncovered"  # rerank 失败 → 保守搜

    # 二分：只有「非常确定覆盖」才跳过（宁可多搜）
    return "covered" if max_score >= coverage_high else "uncovered"


def retrieve_by_doc_routing(question: str, reranker=None, llm=None,
                            top_k: int = None, status_callback=None) -> List[dict]:
    """文档级路由 + 纯树导航检索（轻量编排，把 tree_navigate 当 skill 调用）。

    流程（方案 C V1：DocNovelty 覆盖驱动遍历）：
      1. route_to_docs 选 top-N 文档（相关度降序）
      2. 第一篇永远搜（无对比基准）
      3. 后续每篇：judge_doc_novelty 判断「其核心知识是否已被当前证据覆盖」
         - uncovered → 继续搜，合并证据
         - covered   → 跳过（连续 coverage_max_skip 篇 covered 则启发式早停）
      4. 结果按 score 降序返回

    若 novelty_enabled=false 或 reranker 不可用，回退旧的「条数早停」逻辑。

    Returns:
        统一结果 dict 列表（可能跨多篇文档，按 score 降序）。
    """
    fallback_top_n = int(_get("doc_router.fallback_top_n", 3))
    min_ev = int(_get("doc_router.min_evidences_fallback", 3))
    enabled = bool(_get("doc_router.enabled", True))
    novelty_enabled = bool(_get("doc_router.novelty_enabled", True))
    coverage_high = float(_get("doc_router.coverage_high", 0.55))
    coverage_max_skip = int(_get("doc_router.coverage_max_skip", 2))
    coverage_evidence_top_k = int(_get("doc_router.coverage_evidence_top_k", 5))

    if not enabled:
        if status_callback:
            status_callback("文档路由关闭，退回全库树导航")
        return tree_navigate(question, reranker=reranker, llm=llm)["evidences"]

    doc_ids = route_to_docs(question, llm=llm, top_n=fallback_top_n)
    if not doc_ids:
        if status_callback:
            status_callback("知识库无文档，树导航无结果")
        return []

    # 旧「条数」逻辑（novelty 关闭时保留，完全兼容旧行为）
    if not novelty_enabled:
        primary = doc_ids[0]
        if status_callback:
            status_callback(f"文档路由：选中 {primary}（top-1），开始单文档树检索")
        evs = tree_navigate(question, reranker=reranker, llm=llm, doc_id=primary)["evidences"]
        if len(evs) >= min_ev:
            return evs[:top_k] if top_k else evs
        for did in doc_ids[1:]:
            if status_callback:
                status_callback(f"top-1 证据不足（{len(evs)} < {min_ev}），fallback 到 {did}")
            evs2 = tree_navigate(question, reranker=reranker, llm=llm, doc_id=did)["evidences"]
            _merge_evidences(evs, evs2)
            if len(evs) >= min_ev:
                break
        return evs[:top_k] if top_k else evs

    # 方案 C V1：DocNovelty 覆盖驱动遍历
    evs: List[dict] = []                 # 最终 Evidence
    coverage_evidence: List[dict] = []   # 知识覆盖集合（独立维护）
    skipped_docs: List[str] = []         # 被 DocNovelty 判 covered 而跳过的文档
    skip_streak = 0

    # 一次性构建 doc_id → 卡片映射，消除循环内 _find_card 的全量扫描（O(N²)→O(N)）
    card_map = {c.get("doc_id"): c for c in list_doc_cards()}

    for i, did in enumerate(doc_ids):
        # 第一篇永远搜（无对比基准）
        if i == 0:
            if status_callback:
                status_callback(f"文档路由：选中 {did}（top-1），开始单文档树检索")
            r = tree_navigate(question, reranker=reranker, llm=llm, doc_id=did)
            _merge_evidences(evs, r.get("evidences") or [])
            coverage_evidence = build_coverage_evidence(evs, coverage_evidence_top_k)
            continue

        card = card_map.get(did, {})
        verdict = judge_doc_novelty(card, coverage_evidence, reranker, coverage_high)

        if verdict == "covered":
            skip_streak += 1
            skipped_docs.append(did)
            if status_callback:
                status_callback(f"DocNovelty：{did} 已被当前证据覆盖，跳过（连续 {skip_streak} 篇）")
            # 启发式 early-stop：连续 N 篇 covered 提前停（非正确性保证）
            if skip_streak >= coverage_max_skip:
                if status_callback:
                    status_callback(f"DocNovelty 连续 {skip_streak} 篇 covered，启发式早停")
                break
        else:
            skip_streak = 0
            if status_callback:
                status_callback(f"DocNovelty：{did} 有新增信息，继续检索")
            r = tree_navigate(question, reranker=reranker, llm=llm, doc_id=did)
            _merge_evidences(evs, r.get("evidences") or [])
            coverage_evidence = build_coverage_evidence(evs, coverage_evidence_top_k)

    # 结果按 score 降序（非相邻补读在前、相邻补读在后由 tree_navigate 已保证；
    # 这里跨文档合并后重排，保证整体相关度排序）
    evs.sort(key=lambda e: (bool(e.get("is_neighbor")), -float(e.get("score", 0.0) or 0.0)))
    return evs[:top_k] if top_k else evs


# ---------------------------------------------------------------------------
# 三级降级检索（tree_search）：树导航 → 章节定位 → 以文检文 hybrid 补齐
# ---------------------------------------------------------------------------
def _build_result_id(doc_id: str, parent_id: str, chunk_seq: int) -> str:
    """构造稳定结果 id（去重基础）。"""
    if doc_id:
        return f"{doc_id}:c{chunk_seq}"
    return f"parent:{parent_id}"


def _db_doc_to_result(r: dict) -> dict:
    """把 db_service 检索结果 dict 映射为统一结果 dict。"""
    doc_id = r.get("doc_id") or ""
    parent_id = r.get("parent_id") or ""
    chunk_seq = int(r.get("chunk_seq", 0) or 0)
    rid = _build_result_id(doc_id, parent_id, chunk_seq)
    return {
        "id": rid,
        "text": r.get("text") or "",
        "score": float(r.get("score", 0.0) or 0.0),
        "doc_id": doc_id,
        "section_path": r.get("section_path_str") or r.get("section_path") or "",
        "parent_id": parent_id,
        "chunk_id": rid,
        "summary": r.get("section_summary") or "",
        "is_neighbor": bool(r.get("is_neighbor", False)),
    }


def _build_tree_anchor(merged: List[dict], query: str, max_chars: int = 600) -> str:
    """用树导航已命中的证据正文构造「以文检文」增强 query。"""
    texts = [e.get("text", "").strip() for e in (merged or []) if e.get("text", "").strip()]
    if not texts:
        return query
    anchor = " ".join(texts)
    if len(anchor) > max_chars:
        anchor = anchor[:max_chars]
    return f"{query} {anchor}"


def tree_search(query: str, top_k: int = None, reranker=None, llm=None) -> List[dict]:
    """纯树导航检索统一入口（三级降级，不碰向量召回，最终兜底才 hybrid）。

    降级策略（结果不足 top_k 时逐层补齐）：
      ① 文档级路由 + 单文档树检索（retrieve_by_doc_routing）
      ② 章节定位检索（db_service.retrieve_by_section_entry，仍不碰向量召回）
      ③ 以文检文协同（树命中正文 + 原 query 增强，喂 hybrid 补齐）

    Returns:
        统一结果 dict 列表。
    """
    if top_k is None:
        top_k = _get("search.top_k", 5)

    merged: List[dict] = []
    seen = set()

    def _add(results):
        for r in results or []:
            if r.get("id") and r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)

    # ① 文档级路由 + 单文档树检索
    try:
        _add(retrieve_by_doc_routing(query, reranker=reranker, llm=llm))
    except Exception as e:
        _logger.warning("树导航①文档路由检索失败: %s", e)

    if len(merged) >= top_k:
        return merged[:top_k]

    # ② 章节定位（不碰向量召回）
    try:
        from db_service import retrieve_by_section_entry
        res = retrieve_by_section_entry(query, top_k=top_k)
        _add([_db_doc_to_result(r) for r in (res.get("docs") or [])])
    except Exception as e:
        _logger.warning("树导航②章节定位检索失败: %s", e)

    if len(merged) >= top_k:
        return merged[:top_k]

    # ③ 以文检文协同兜底（补齐到 top_k）
    anchor = _build_tree_anchor(merged, query)
    try:
        from db_service import search_documents
        _add([_db_doc_to_result(r) for r in
              (search_documents(anchor, top_k=top_k, expand_neighbors=False, hybrid=True) or [])])
    except Exception as e:
        _logger.warning("树导航③以文检文兜底检索失败: %s", e)
    return merged[:top_k]
