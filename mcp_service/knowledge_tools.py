"""
================================================================================
MCP 知识库工具 — 把本地 RAG 知识库检索能力封装为可对外调用的工具
================================================================================

让外部模型（Claude Desktop / Cursor / 任意 MCP 客户端）通过 MCP 协议
直接检索本项目已入库的知识库，而不只是联网搜索。

提供的工具：
  1. search_knowledge_base(query, top_k, mode)  知识库检索（vector/hybrid/tree）
  2. list_knowledge_documents()                 列出已入库文档（了解知识库覆盖范围）
  3. get_knowledge_toc()                        获取知识库目录结构（章节大纲）

设计说明：
  - 本模块只封装「纯函数」，不依赖 FastMCP 装饰器，方便单独测试与复用；
    实际的 @mcp.tool 注册在 mcp_service/__main__.py 完成。
  - 所有底层依赖（db_service / reranker / llm）均「懒加载 + 带缓存」，
    避免 MCP 服务器启动时就 import torch / transformers 等重依赖。
  - tree 模式复用根目录 tree_retrieval.tree_search（三级降级：
    纯树导航 → 章节定位 → hybrid），失败不崩溃。

注意：
  - 本模块运行在 MCP 独立进程内，会读取项目根的 config/db.json 与
    config/models.json 来初始化 embedding / Milvus / reranker / LLM，
    与主程序共享同一套配置，因此「当前选中数据库 / 模型」对两边一致。
================================================================================
"""

from __future__ import annotations

import json
from typing import List, Optional

# 懒加载单例缓存
_reranker = None
_llm = None


# ---------------------------------------------------------------------------
# 结果规范化（dict / Evidence → 可 JSON 序列化的简洁 dict）
# ---------------------------------------------------------------------------
def _normalize_result(r: dict) -> dict:
    """把 db_service 检索结果 dict 精简为外部模型友好的字段。"""
    return {
        "text": (r.get("text") or "").strip(),
        "score": round(float(r.get("score", 0.0) or 0.0), 4),
        "doc_id": r.get("doc_id") or "",
        "section_path": r.get("section_path_str") or "",
        "section_summary": r.get("section_summary") or "",
        "source": r.get("source") or "",
    }


def _tree_result_to_dict(d: dict) -> dict:
    """把 tree_retrieval 的统一结果 dict 转成外部模型友好的字段。"""
    return {
        "text": (d.get("text") or "").strip(),
        "score": round(float(d.get("score", 0.0) or 0.0), 4),
        "doc_id": d.get("doc_id") or "",
        "section_path": d.get("section_path") or "",
        "section_summary": d.get("summary") or "",
        "source": "",
    }


# ---------------------------------------------------------------------------
# reranker / llm 懒加载（tree 模式专用，带缓存）
# ---------------------------------------------------------------------------
def _get_reranker():
    """懒加载 Reranker 实例（本地 HF / 远程 API，按 models.json 配置）。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    import store_config
    from reranker import Reranker

    name = store_config.get_current("reranker")
    m = store_config.get_model_by_name("reranker", name) if name else None
    if m is None:
        models = store_config.list_models("reranker")
        m = models[0] if models else None
    if m is None:
        raise RuntimeError("未配置 Reranker 模型（请在设置中添加）")

    if m.get("type") == "local":
        _reranker = Reranker(local_model_path=m.get("model_path"))
    else:
        _reranker = Reranker(
            model=m.get("model"),
            base_url=m.get("base_url"),
            api_key=m.get("api_key"),
            online=True,
        )
    return _reranker


def _get_llm():
    """懒加载决策 LLM（tool_llm 优先回退 llm）。"""
    global _llm
    if _llm is not None:
        return _llm
    from llm_factory import get_model
    _llm = get_model("tool_llm", "llm")
    return _llm


# ---------------------------------------------------------------------------
# 工具 1：知识库检索
# ---------------------------------------------------------------------------
def search_knowledge_base(query: str, top_k: int = 5, mode: str = "hybrid") -> list:
    """
    检索本地 RAG 知识库，返回与 query 最相关的文档片段。

    Args:
        query: 检索文本 / 问题
        top_k: 返回片段数量（默认 5）
        mode:  检索模式：
               vector = 纯向量语义检索；
               hybrid = 混合检索（dense 向量 + BM25 稀疏 + RRF 融合，默认）；
               tree   = 纯 LLM 树导航检索（不碰向量召回，逐层 descend/backtrack）

    Returns:
        [{"text": 片段原文, "score": 相关度分, "doc_id": 文档ID,
          "section_path": 章节路径, "section_summary": 章节摘要}, ...]
    """
    if not query or not query.strip():
        return []

    mode = (mode or "hybrid").strip().lower()
    if mode not in ("vector", "hybrid", "tree"):
        mode = "hybrid"

    if mode in ("vector", "hybrid"):
        from db_service import search_documents
        results = search_documents(
            query.strip(),
            top_k=top_k,
            expand_neighbors=True,
            hybrid=(mode == "hybrid"),
        )
        return [_normalize_result(r) for r in (results or [])]

    # tree 模式：纯树导航（统一走根目录 tree_retrieval 三级降级）
    try:
        import tree_retrieval
        docs = tree_retrieval.tree_search(
            query.strip(), top_k=top_k,
            reranker=_get_reranker(), llm=_get_llm(),
        )
        return [_tree_result_to_dict(d) for d in (docs or [])]
    except Exception:
        # 树导航依赖（reranker/LLM）不可用 → 降级 hybrid
        from db_service import search_documents
        results = search_documents(
            query.strip(), top_k=top_k, expand_neighbors=True, hybrid=True
        )
        return [_normalize_result(r) for r in (results or [])]


# ---------------------------------------------------------------------------
# 工具 2：列出已入库文档
# ---------------------------------------------------------------------------
def list_knowledge_documents() -> list:
    """
    列出知识库中已入库的文档（帮助外部模型了解知识库覆盖范围）。

    Returns:
        [{"doc_id", "title", "source", "node_count", "created_at"}, ...]
    """
    try:
        from db_service import list_documents
        docs = list_documents() or []
    except Exception:
        return []
    out = []
    for d in docs:
        out.append({
            "doc_id": d.get("doc_id") or "",
            "title": d.get("title") or d.get("source") or "",
            "source": d.get("source") or "",
            "node_count": d.get("node_count") or 0,
            "created_at": d.get("created_at") or "",
        })
    return out


# ---------------------------------------------------------------------------
# 工具 3：获取知识库目录
# ---------------------------------------------------------------------------
def get_knowledge_toc() -> str:
    """
    获取知识库中所有文档的目录结构（章节大纲），帮助外部模型判断
    「某个问题大概在哪个章节」再决定是否检索。

    Returns:
        目录文本（多文档用【文档】分隔，缩进表示层级）；知识库为空返回空串。
    """
    try:
        from db_service import get_toc_text
        return get_toc_text()
    except Exception:
        return ""


# 供 MCP 服务器注册的工具描述（与 @mcp.tool 的 docstring 保持一致）
TOOL_DEFS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "检索本地 RAG 知识库，返回与 query 最相关的文档片段。"
            "mode 可选 vector（纯向量）/ hybrid（混合检索，默认）/ tree（纯 LLM 树导航）。"
        ),
    },
    {
        "name": "list_knowledge_documents",
        "description": "列出知识库中已入库的文档，帮助了解知识库覆盖范围。",
    },
    {
        "name": "get_knowledge_toc",
        "description": "获取知识库的目录结构（章节大纲），判断问题落在哪个章节。",
    },
]
