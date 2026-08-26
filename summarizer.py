# -*- coding: utf-8 -*-
"""
================================================================================
章节摘要生成（Summarizer）—— 解析阶段的 LLM 增强
================================================================================

在「结构归位」产出文档树后，用 LLM 为每个 section 生成一句话摘要，
作为检索阶段的铺垫（廉价证据摘要）：

  - 检索时：chunk 召回后带出所属 section 摘要，快速判断章节相关性
  - 问题路由：目录型/章节定位型问题可用摘要判断落点
  - Agentic Evaluator：判断 requirement 是否被满足时，先看摘要再决定读原文

设计约束（与架构定稿一致）：
  1. summary 是「路由/评估/目录」用的元数据，**最终生成答案仍用原文 chunk**，
     绝不能用摘要替代原文（否则信息损失 + 幻觉）。
  2. 只在入库时生成，检索阶段绝不实时调 LLM。
  3. 用弱模型（tool_llm 优先，回退 llm），量大、成本可控。
  4. 失败不阻塞入库：单个 section 生成失败 → summary 留空，检索仍用原文。

用法：
    from summarizer import summarize_tree
    n = summarize_tree(root)   # 就地填充 root 上各 section 的 summary，返回生成条数
================================================================================
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from structure_resolver import TreeNode


# ---------------------------------------------------------------------------
# 模型获取（复用 tool_llm，回退 llm；与 server.get_tool_llm 逻辑一致但独立，
# 避免 db_service -> server 的循环依赖）
# ---------------------------------------------------------------------------
def _get_summary_llm():
    """获取摘要模型实例；未配置任何可用模型时返回 None（跳过摘要生成）。

    优先 summary 专用模型（用户可在设置界面配置轻量模型），回退 tool_llm、llm。
    """
    from llm_factory import get_model
    return get_model("summary", "tool_llm", "llm")


def _summarize_with(llm, system: str, user: str) -> str:
    """非流式调用 LLM，返回摘要文本；失败返回空字符串。"""
    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        text = invoke_llm(llm, [SystemMessage(content=system), HumanMessage(content=user)])
        return (text or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------
_SECTION_SYSTEM = (
    "你是文档摘要助手。请用一句话（不超过 40 字）概括给定章节的核心内容，"
    "只输出摘要本身，不要任何解释、编号或标点以外的内容。"
)


def summarize_section(title: str, text: str, llm) -> str:
    """为单个 section 生成一句话摘要。"""
    content = text.strip()
    if len(content) < 10:
        return ""
    user = f"章节标题：{title or '（无标题）'}\n\n章节内容：\n{content[:3000]}"
    return _summarize_with(llm, _SECTION_SYSTEM, user)


def _collect_direct_leaves(section: TreeNode) -> List[str]:
    """收集 section 直属叶子节点（paragraph/table/figure）的文本，不含子 section。"""
    return [
        c.text.strip()
        for c in section.children
        if c.type != "section" and c.text and c.text.strip()
    ]


def summarize_tree(root: TreeNode, llm=None, enabled: bool = True, max_workers: int = 4) -> int:
    """遍历文档树，为每个 section 就地生成一句话摘要（并发）。

    章节摘要之间**完全独立**（每个 section 只读自己的直属叶子，互不依赖），
    且 LLM 调用是网络 IO 密集，因此用线程池并发调用可显著提速：串行 N 个 section
    = N×单次耗时，并发 max_workers 个 = N/max_workers×单次耗时。结果与串行完全
    一致（纯无损：同样的输入、同样的模型、同样的 prompt，只是调度顺序不同）。

    Args:
        root:        文档树根节点（TreeNode, type=document）
        llm:         摘要模型实例；未提供时内部按配置获取
        enabled:     开关（config.summary.enabled），False 直接返回 0
        max_workers: 并发线程数（config.summary.concurrency）；<=1 时走串行（原行为）

    Returns:
        成功生成的摘要条数。
    """
    if not enabled:
        return 0
    if llm is None:
        llm = _get_summary_llm()
    if llm is None:
        return 0

    # 1. 收集所有需要摘要的 section（先遍历收集，再统一并发，避免递归中共享状态）
    targets: List[Tuple[TreeNode, str]] = []

    def collect(section: TreeNode):
        leaves = _collect_direct_leaves(section)
        if leaves:
            targets.append((section, "\n".join(leaves)))
        for c in section.children:
            if c.type == "section":
                collect(c)

    for c in root.children:
        if c.type == "section":
            collect(c)

    if not targets:
        return 0

    # 2. 生成摘要（并发或串行；结果一致）
    count = 0
    if max_workers <= 1 or len(targets) == 1:
        # 串行（原行为，兼容并发度为 1 / 单 section 场景）
        for section, text in targets:
            s = summarize_section(section.title, text, llm)
            if s:
                section.summary = s
                count += 1
        return count

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _do(section: TreeNode, text: str):
        return section, summarize_section(section.title, text, llm)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do, section, text) for section, text in targets]
        for fut in as_completed(futures):
            try:
                section, s = fut.result()
            except Exception:
                s = ""  # 单个 section 失败不阻塞整体（与串行语义一致）
            if s:
                section.summary = s
                count += 1
    return count


# ---------------------------------------------------------------------------
# 文档级主旨摘要（跨文档路由用）
# ---------------------------------------------------------------------------
_DOC_SYSTEM = (
    "你是文档索引助手。请根据给定的文档标题与章节目录，生成该文档的"
    "「主旨 + 关键词」，用于后续判断用户问题该路由到哪篇文档。"
    "严格只输出 JSON，格式："
    '{"abstract": "一句话主旨（不超过 60 字）", "keywords": ["k1", "k2", "k3"]}，'
    "不要任何解释、不要代码块标记。"
)


def summarize_document(root: TreeNode, llm=None, max_chars: int = 3000) -> str:
    """生成文档级主旨摘要（跨文档路由索引）。

    基于文档标题 + 章节标题列表（含章节摘要）生成，输出 JSON 字符串：
    {"abstract": "...", "keywords": [...]}；失败返回空串（不阻塞入库）。

    Args:
        root:      文档树根节点（type=document，summary 应已就地填充）
        llm:       摘要模型；未提供时内部按配置获取
        max_chars: 喂给 LLM 的目录文本截断上限

    Returns:
        JSON 字符串（可直接存 documents.abstract）；失败返回 ""。
    """
    if llm is None:
        llm = _get_summary_llm()
    if llm is None:
        return ""

    # 收集文档标题 + 顶层章节（含摘要）作为「文档卡片」素材
    lines = [f"文档标题：{root.title or '（无标题）'}", "章节目录："]
    for c in root.children:
        if c.type != "section":
            continue
        line = f"- {c.title}"
        if c.summary:
            line += f"（{c.summary}）"
        lines.append(line)
    toc_text = "\n".join(lines)[:max_chars]

    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        text = invoke_llm(llm, [SystemMessage(content=_DOC_SYSTEM),
                                HumanMessage(content=toc_text)])
        text = (text or "").strip()
        # 容错：剥离可能的 ```json ``` 代码块标记（统一用 common/text_utils）
        from common.text_utils import strip_code_fence
        text = strip_code_fence(text)
        return text
    except Exception:
        return ""
