# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Retrieval 抽象 + Retrieval Router（Phase 2）
================================================================================

把现有 milvus_store 的三路检索（vector / hybrid / tree）封装成统一接口，
并让 Retrieval Router 用「纯规则」选择检索路径（Controller 不直接选 tool）。

设计要点（与架构定稿一致）：

  1. 复用 db_service 高层函数（已封装 embedding + 邻近扩展 + 章节路径恢复），
     不重写检索逻辑。
  2. SEARCH 显式关闭邻近扩展（expand_neighbors=False）：上下文补全应由
     READ_PARENT / READ_SECTION 显式触发，而非 SEARCH 自动带上，否则
     Evidence 粒度会混入邻近块，READ 动作也失去意义。
  3. Evidence id 必须是「稳定 id」，不能直接用 Milvus auto_id（重新入库会变）：
       - 结构树 chunk： f"{doc_id}:c{chunk_seq}"
       - 普通父块：     f"parent:{parent_id}"
     这是 novelty 差集去重的基础（古早 bug 高发区）。
  4. tree 检索无命中时自动 fallback 到 hybrid，避免浪费一轮。

说明：
  - READ_PARENT / READ_SECTION 的底层能力在 Phase 6（Executor）实现，
    本模块只提供 vector / hybrid / tree 三路检索与路由。
================================================================================
"""

from __future__ import annotations

import re
from typing import List

from .state import Evidence, EvidenceSource, RetrievalTool
from .settings import get as _get


# ---------------------------------------------------------------------------
# 章节定位型判断（Router 规则：query/gap 明显指向章节/目录 → TREE）
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = [
    re.compile(r"第[零一二三四五六七八九十百千\d]+[章节部分]"),
    re.compile(r"(目录|大纲|有哪些章节|哪一?节|第几章)"),
    re.compile(r"\d+(\.\d+)*\s*[章节]"),
]


def _looks_like_section(text: str) -> bool:
    """判断文本是否明确指向「章节 / 目录」定位。"""
    return any(p.search(text) for p in _SECTION_PATTERNS)


# ---------------------------------------------------------------------------
# 检索模式 → RetrievalTool 映射
# ---------------------------------------------------------------------------
_RETRIEVAL_MODES = {
    "vector": RetrievalTool.VECTOR,
    "hybrid": RetrievalTool.HYBRID,
    "tree": RetrievalTool.TREE,
}


def mode_to_tool(mode: str) -> RetrievalTool:
    """把「检索模式」字符串映射为 RetrievalTool。

    支持：vector（普通检索）/ hybrid（混合检索）/ tree（树导航检索）。
    非法值回退 hybrid（默认覆盖绝大多数场景）。
    """
    return _RETRIEVAL_MODES.get((mode or "").strip().lower(), RetrievalTool.HYBRID)


# ---------------------------------------------------------------------------
# Retrieval Router（纯规则）
# ---------------------------------------------------------------------------
def route(query: str, target_gap: str = "", gap_context: str = "") -> RetrievalTool:
    """根据 query / gap 选择检索路径（纯规则，无 LLM）。

    规则（V1 极简）：
      - query / gap_context 明显指向章节 / 目录 → TREE（章节定位）
      - 否则 → HYBRID（dense + BM25 + RRF，默认，覆盖绝大多数）

    Args:
        query:       Controller 生成的检索文本
        target_gap:  指向的 requirement_id（如 "R3"，通常不含章节信息）
        gap_context: requirement.description 或 gap 描述（含"第三章"等语义）
    """
    text = f"{query} {gap_context}".strip()
    if _looks_like_section(text):
        return RetrievalTool.TREE
    return RetrievalTool.HYBRID


# ---------------------------------------------------------------------------
# 统一检索入口
# ---------------------------------------------------------------------------
def retrieve(tool: RetrievalTool, query: str, top_k: int = None, round_num: int = 0,
             reranker=None, llm=None) -> List[Evidence]:
    """统一检索入口：按 tool 路由到对应实现，返回 Evidence 列表。

    Args:
        tool:      检索路径（vector / hybrid / tree）
        query:     检索文本
        top_k:     返回数量（None=读配置 agentic.search.top_k）
        round_num: 当前循环轮次（写入 Evidence.produced_by_round）
        reranker:  Reranker 实例（tree 导航用，None=树导航降级为无 rerank）
        llm:       决策 LLM（tree 导航用，None=树导航固定策略）

    Returns:
        Evidence 列表（已做稳定 id 映射）。检索异常时返回空列表（不崩溃，
        由上层 Controller 决定下一步 REFINE / READ / ANSWER）。
    """
    if top_k is None:
        top_k = _get("search.top_k", 5)

    try:
        if tool == RetrievalTool.TREE:
            # 纯树导航：返回的已是 Evidence 列表，直接返回
            return _search_tree_nav(query, top_k, round_num, reranker, llm)
        elif tool == RetrievalTool.VECTOR:
            raw = _search_vector(query, top_k)
        elif tool == RetrievalTool.HYBRID:
            raw = _search_hybrid(query, top_k)
        else:
            raw = []
    except Exception:
        # 检索失败不崩溃，返回空证据，交给 Controller 决策
        raw = []

    return [_to_evidence(r, round_num) for r in raw]


# ---------------------------------------------------------------------------
# 三路检索实现（复用 db_service，不重写检索逻辑）
# ---------------------------------------------------------------------------
def _search_vector(query: str, top_k: int) -> list:
    from db_service import search_documents
    return search_documents(query, top_k=top_k, expand_neighbors=False, hybrid=False) or []


def _search_hybrid(query: str, top_k: int) -> list:
    from db_service import search_documents
    return search_documents(query, top_k=top_k, expand_neighbors=False, hybrid=True) or []


def _tree_dict_to_evidence(d: dict, round_num: int) -> Evidence:
    """把 tree_retrieval 的统一结果 dict 转成 Evidence。

    tree_retrieval 返回的 dict 字段（id/text/score/doc_id/parent_id/
    section_path/chunk_id/summary）与 db_service 检索结果（chunk_seq/
    section_summary）不同，故单独转换。
    """
    return Evidence(
        id=d.get("id") or "",
        text=d.get("text") or "",
        score=float(d.get("score", 0.0) or 0.0),
        source=EvidenceSource(
            doc_id=d.get("doc_id") or "",
            section_path=d.get("section_path") or "",
            parent_id=d.get("parent_id") or "",
            chunk_id=d.get("chunk_id") or d.get("id") or "",
            summary=d.get("summary") or "",
        ),
        produced_by_round=round_num,
    )


def _search_tree_nav(query: str, top_k: int, round_num: int, reranker, llm) -> List[Evidence]:
    """纯树导航检索（TREE 分支），复用根目录 tree_retrieval 三级降级。

    降级策略（tree_retrieval.tree_search 内部实现，结果不足 top_k 时逐层补齐）：
      ① 文档级路由 + 单文档树检索
      ② 章节定位检索（仍不碰向量召回）
      ③ 以文检文协同（树命中正文 + 原 query 增强，喂 hybrid 补齐）
         ——仅当树结果不足 top_k 时才触发；树结果足够时完全不碰向量召回。

    Args:
        reranker / llm: 树导航所需（None=树导航降级为固定策略/无 rerank）
    """
    import tree_retrieval
    docs = tree_retrieval.tree_search(query, top_k=top_k, reranker=reranker, llm=llm)
    return [_tree_dict_to_evidence(d, round_num) for d in (docs or [])]


# ---------------------------------------------------------------------------
# dict → Evidence 映射（稳定 id 构造）
# ---------------------------------------------------------------------------
def _build_evidence_id(doc_id: str, parent_id: str, chunk_seq: int) -> str:
    """构造稳定 Evidence id（novelty 去重基础）。

    - 结构树 chunk：doc_id 非空，chunk_seq 在同一 doc 内全局唯一 → f"{doc_id}:c{chunk_seq}"
    - 普通父块：doc_id 空，parent_id 为 UUID → f"parent:{parent_id}"
    """
    if doc_id:
        return f"{doc_id}:c{chunk_seq}"
    return f"parent:{parent_id}"


def _to_evidence(r: dict, round_num: int) -> Evidence:
    """把 db_service 检索结果 dict 映射为 Evidence（字段缺失时兜底）。"""
    doc_id = r.get("doc_id") or ""
    parent_id = r.get("parent_id") or ""
    chunk_seq = int(r.get("chunk_seq", 0) or 0)
    eid = _build_evidence_id(doc_id, parent_id, chunk_seq)

    return Evidence(
        id=eid,
        text=r.get("text") or "",
        # 注意：score 语义随检索路径不同（COSINE 距离 / RRF 分数），仅作记录；
        # quality 指标统一由 Phase 3 的 Reranker 分数给出，不可直接跨路径比较。
        score=float(r.get("score", 0.0) or 0.0),
        source=EvidenceSource(
            doc_id=doc_id,
            section_path=r.get("section_path") or "",
            parent_id=parent_id,
            chunk_id=eid,
            summary=r.get("section_summary") or "",  # 中文章节摘要，评估辅助
        ),
        produced_by_round=round_num,
    )
