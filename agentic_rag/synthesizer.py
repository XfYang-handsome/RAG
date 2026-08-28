# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Synthesis 带引用合成（Phase 8）
================================================================================

只基于「被验证的证据」生成答案，关键论断后标注引用编号，降低幻觉。

设计（与架构定稿一致）：

  1. 输入是 SUPPORTED / PARTIAL 的 evidence（按 requirement_status 关联），
     **不是全部检索结果**——MISSING 的证据不进入合成。
  2. 每个证据映射一个短编号 [1] [2] ...，答案里用 [n] 标注，
     编号可回溯到真实 evidence id + source（前端可跳转）。
  3. 无相关证据时明确回答「证据不足」，绝不编造。
  4. 模型：llm（强模型）。
  5. claim→evidence 完整矩阵 / 逐条 claim 验证留 V2，V1 只做「带引用的合成」。

用法：
    from agentic_rag.synthesizer import synthesize
    result = synthesize(state)   # {"answer": ..., "citations": [...]}
================================================================================
"""

from __future__ import annotations

import re
from typing import List, Optional

from .state import AgentState, RequirementStatus
from .settings import get as _get

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 模型获取（llm 强模型，生成答案 → answer=True 温度）
# ---------------------------------------------------------------------------
def _get_synthesis_llm():
    from llm_factory import get_model
    return get_model("llm", answer=True)


# ---------------------------------------------------------------------------
# 证据收集（只取被验证的 SUPPORTED / PARTIAL）
# ---------------------------------------------------------------------------
def _resolve_readable_source(ev) -> dict:
    """把 EvidenceSource 补成「可读来源」：文档标题 + 章节标题路径。

    前端引用列表此前只显示 `doc_id`（一串 hash）+ `section_path`（数字路径如 2/1），
    可读性极差。这里用 tree_store 补齐：
      - doc_title：documents.title（文档真实标题）
      - section_titles：沿树向上恢复的章节标题路径（如 ["第一章", "1.1 架构"]）

    兼容三种证据：
      - 树导航证据：chunk_id = "tree:{node_id}"，可直接反解 node_id 恢复标题路径
      - 普通检索证据：chunk_id = "{doc_id}:c{seq}"，无 node_id，退回 section_path + summary
      - 联网证据：origin="web"，直接用 url
    查询失败全部静默降级（不阻塞合成），宁可少字段也不崩溃。
    """
    src = ev.source.model_dump()
    doc_id = src.get("doc_id") or ""
    chunk_id = src.get("chunk_id") or ""

    # 文档标题（联网证据无 doc_id，跳过）
    if doc_id:
        try:
            import tree_store
            doc = tree_store.get_document(doc_id)
            if doc:
                src["doc_title"] = doc.get("title") or ""
        except Exception:
            pass

    # 章节标题路径：树导航证据可从 chunk_id 反解 node_id
    if doc_id:
        node_id = ""
        if chunk_id.startswith("tree:"):
            node_id = chunk_id[len("tree:"):]
        elif src.get("parent_id"):
            node_id = src["parent_id"]
        if node_id:
            try:
                import tree_store
                titles = tree_store.get_section_path_titles(doc_id, node_id)
                if titles:
                    src["section_titles"] = titles
            except Exception:
                pass

    return src


def collect_validated(state: AgentState) -> List[dict]:
    """收集被验证的证据（SUPPORTED/PARTIAL 关联的 evidence），去重并编号。

    Returns:
        [{"num": 1, "evidence_id": ..., "text": ..., "source": ...}, ...]
        source 内额外含 doc_title / section_titles（可读来源，见 _resolve_readable_source）。
    """
    valid_ids = set()
    for item in state.requirement_status.values():
        if item.status in (RequirementStatus.SUPPORTED, RequirementStatus.PARTIAL):
            valid_ids.update(item.evidence_ids)

    citations: List[dict] = []
    # 按 reranker score 降序排列：最相关的证据排最前，保证合成时优先看到高相关证据
    # （原实现用 sorted(valid_ids) 按 id 排序，最相关证据可能被排在最后）
    ordered = sorted(
        valid_ids,
        key=lambda eid: state.evidences[eid].score if eid in state.evidences else -1.0,
        reverse=True,
    )
    for i, eid in enumerate(ordered, 1):
        ev = state.evidences.get(eid)
        if ev is None:
            continue
        citations.append({
            "num": i,
            "evidence_id": eid,
            "text": ev.text,
            # 转成 dict：避免 server 侧 json.dumps(default=str) 把 pydantic 对象
            # 序列化成字符串，导致前端 source.doc_id/section_path 全为 undefined（显示"未知来源"）
            "source": _resolve_readable_source(ev),
            "score": ev.score,
        })
    return citations


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
_SYNTH_SYSTEM = (
    "你是严谨的问答助手。请只依据给定证据回答，不要编造证据之外的信息。"
    "在关键论断后标注来源编号，如 [1] 或 [1][2]；无证据支撑的内容不要写。"
    "\n\n"
    "回答格式要求（使用 Markdown 提升可读性，与检索回答保持一致）：\n"
    "1. 使用标题（##、###）划分主题，不要堆成一大段。\n"
    "2. 要点使用无序/有序列表逐条列出。\n"
    "3. 适当用 **加粗** 突出关键词，用短段落。\n"
    "4. 代码、命令、路径用行内代码（`code`）。\n"
    "5. 数学公式用 LaTeX 表示：行内公式用 \\(...\\) 包裹（如 \\(x^2\\)），"
    "块级公式用 \\[...\\] 单独成行包裹。\n"
    "6. 回答要完整但简洁：覆盖问题的各个要点，但每个要点直接给出答案，"
    "不要展开与问题无关的背景、延伸或铺垫内容；答案长度与问题复杂度匹配，"
    "简单问题用简短回答，不为了「显得详尽」而凑篇幅。\n"
    "7. 每个关键论断后用 [n] 标注其证据来源编号。"
)


def _custom_system_prompt() -> str:
    """读取用户自定义 system prompt（config.system_prompt，可为空）。

    与 rag_graph 的 _custom_system_prompt 对齐：自定义 prompt 作为角色/任务设定，
    但「只依据证据 + 标注引用」的硬约束始终保留，避免自定义 prompt 覆盖后
    回答脱离证据（这正是 agentic 模式下自定义提示词「没生效」的根因——此前
    完全没读 config）。
    """
    try:
        from config_loader import cfg
        sp = cfg("system_prompt", "") or ""
        return str(sp).strip()
    except Exception:
        return ""


def _build_system() -> str:
    """构造合成 system prompt：自定义 prompt 优先，证据约束始终保留。"""
    custom = _custom_system_prompt()
    if custom:
        # 用户自定义 prompt 作为角色/任务设定，证据约束 + 格式要求始终保留
        return (
            custom
            + "\n\n"
            +             "你正在做基于证据的问答，必须遵守：\n"
            "1. 只依据给定证据回答，不要编造证据之外的信息。\n"
            "2. 关键论断后标注来源编号 [n]，无证据支撑的内容不要写。\n"
            "3. 使用 Markdown（标题/列表/加粗/行内代码）提升可读性。\n"
            "4. 数学公式用 LaTeX 表示：行内公式用 \\(...\\) 包裹，块级公式用 \\[...\\] 单独成行包裹。\n"
            "5. 回答完整但简洁：直接给出问题各要点的答案，"
            "省略与问题无关的背景和延伸内容，长度与问题复杂度匹配。"
        )
    return _SYNTH_SYSTEM


def _build_prompt(question: str, citations: List[dict]) -> str:
    ev_chars = _get("synthesizer.evidence_chars", 1500)
    ev_lines = "\n".join(f"[{c['num']}] {c['text'][:ev_chars]}" for c in citations)
    return (
        f"问题：{question}\n\n"
        f"可用证据：\n{ev_lines or '（无）'}\n\n"
        "请综合相关证据，完整、简洁地回答上述问题：直接给出问题各要点的答案，"
        "省略与问题无关的背景和延伸内容，并在每个关键论断后用 [n] 标注证据来源编号。"
    )


def _extract_reasoning(chunk) -> str:
    """从流式 chunk 提取思考过程（deepseek-reasoner 的 reasoning_content）。"""
    if chunk is None:
        return ""
    additional = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = additional.get("reasoning_content") or additional.get("reasoning")
    if reasoning:
        if isinstance(reasoning, list):
            return "".join(
                r.get("text", "") if isinstance(r, dict) else str(r) for r in reasoning
            )
        return str(reasoning)
    return ""


def _validate_citations(answer: str, citations: List[dict]) -> str:
    """校验答案里的 [n] 引用编号，剔除越界编号（防 LLM 幻觉引用）。

    LLM 可能编造 [7]（而证据只有 5 条），导致前端引用列表对不上。
    这里扫描答案中的 [n] / [n][m] 标记，把超出可用编号范围的引用删除
    （保留文本，只去掉不存在的编号标记），避免用户点到「未知来源」。

    Args:
        answer:    合成后的答案文本
        citations: 有效证据列表（编号 1..len(citations)）

    Returns:
        清洗后的答案文本（越界引用被移除）
    """
    if not answer:
        return answer
    max_num = len(citations)  # citations 为空时 max_num=0，所有 [n] 引用都会被剔除
    # 匹配 [数字] 或连续的 [1][2][3] 引用块
    def _repl(m):
        full = m.group(0)
        # 提取内部所有编号
        nums = re.findall(r"\[(\d+)\]", full)
        if not nums:
            return full
        kept = []
        for n in nums:
            if 1 <= int(n) <= max_num:
                kept.append(f"[{n}]")
        return "".join(kept)
    return re.sub(r"(?:\[\d+\])+", _repl, answer)


def _make_stream_validator(stream_callback, max_num: int):
    """返回 (push, flush)：对流式正文做引用编号校验的缓冲推送器。

    引用块 [n] 可能被 LLM 流式切成 [、1、] 三段，故用缓冲累积：
    - 遇到 [ 且能匹配「[数字]」→ 校验编号后决定保留/丢弃；
    - 遇到 [ 但尚未闭合 → 保留在缓冲，等后续 chunk 补齐；
    - 其余普通文本直接推送。
    越界编号（> max_num）整块丢弃，防幻觉引用。
    """
    buf = ""

    def push(content: str) -> None:
        nonlocal buf
        buf += content
        out = ""
        i, n = 0, len(buf)
        while i < n:
            if buf[i] == "[":
                m = re.match(r"\[\d+\]", buf[i:])
                if m:
                    num = int(m.group(0)[1:-1])
                    if 1 <= num <= max_num:
                        out += m.group(0)
                    i += len(m.group(0))
                    continue
                # [ 尚未闭合（可能被切成 [1 / [1] 未到），暂停，保留等待
                break
            else:
                out += buf[i]
                i += 1
        buf = buf[i:]
        if out:
            stream_callback(out, False)

    def flush() -> None:
        nonlocal buf
        if buf:
            stream_callback(buf, False)
            buf = ""

    return push, flush


_NO_ANSWER_SYSTEM = (
    "你是严谨的问答助手。经过检索，知识库中没有找到足够的相关信息来回答用户的问题。"
    "请诚实、礼貌地告知用户这一结果，不要编造任何答案。回复应：\n"
    "1. 简要说明未能回答的原因（知识库中未找到相关内容）。\n"
    "2. 给出建设性建议：换一种问法、补充更多背景，或提示该问题可能超出当前知识库覆盖范围。\n"
    "3. 使用 Markdown 格式（短段落 + 列表），语气自然友好，不要机械重复。"
)


def _generate_no_answer(question: str, llm, stream_callback=None) -> str:
    """无验证证据时，用 LLM 生成一个诚实、有针对性的「无法回答」回复。

    区别于固定文案：根据具体问题生成，可建议换问法 / 补充资料，体验更好；
    且通过 stream_callback 推送给前端，避免前端因收不到任何内容而显示
    「[未获取到回答]」占位符。
    """
    messages = [
        SystemMessage(content=_NO_ANSWER_SYSTEM),
        HumanMessage(content=f"用户问题：{question}\n\n请生成一个礼貌、诚实、有建设性的「无法回答」回复。"),
    ]

    # 流式生成（与 _generate_answer 一致：正文经 stream_callback 推送，无引用校验）
    if stream_callback is not None and hasattr(llm, "stream"):
        parts = []
        try:
            for chunk in llm.stream(messages):
                reasoning = _extract_reasoning(chunk)
                if reasoning:
                    stream_callback(reasoning, True)
                    continue
                content = getattr(chunk, "content", None)
                if isinstance(content, list):
                    content = ""
                content = content or ""
                if content:
                    if bool(getattr(chunk, "is_reasoning", False)):
                        stream_callback(content, True)
                        continue
                    parts.append(content)
                    stream_callback(content, False)
            return "".join(parts).strip()
        except Exception:
            pass  # 流式失败 → 落到非流式兜底

    # 非流式兜底
    from llm import invoke_llm
    try:
        text = invoke_llm(llm, messages)
        answer = (text or "").strip()
        if stream_callback:
            stream_callback(answer, False)
        return answer
    except Exception:
        fallback = "抱歉，经过多轮检索，知识库中未找到足够的证据来回答该问题。建议您换个问法，或补充相关资料后再试。"
        if stream_callback:
            stream_callback(fallback, False)
        return fallback


def _generate_answer(question: str, citations: List[dict], llm, stream_callback=None) -> str:
    prompt = _build_prompt(question, citations)
    messages = [SystemMessage(content=_build_system()), HumanMessage(content=prompt)]
    max_num = len(citations)

    # 流式生成：正文经 stream_callback 实时推前端（fn(content, is_reasoning)）
    # 引用编号在推送前校验（越界引用丢弃），防止前端引用对不上。
    if stream_callback is not None and hasattr(llm, "stream"):
        push, flush = _make_stream_validator(stream_callback, max_num)
        parts = []
        try:
            for chunk in llm.stream(messages):
                reasoning = _extract_reasoning(chunk)
                if reasoning:
                    stream_callback(reasoning, True)
                    continue
                content = getattr(chunk, "content", None)
                if isinstance(content, list):
                    content = ""
                content = content or ""
                if content:
                    if bool(getattr(chunk, "is_reasoning", False)):
                        stream_callback(content, True)
                        continue
                    parts.append(content)
                    push(content)
            flush()
            return "".join(parts).strip()
        except Exception as e:
            return f"[合成失败: {e}]"

    # 非流式兜底
    from llm import invoke_llm
    try:
        text = invoke_llm(llm, messages)
        return _validate_citations((text or "").strip(), citations)
    except Exception as e:
        return f"[合成失败: {e}]"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def synthesize(state: AgentState, llm=None, stream_callback=None) -> dict:
    """基于已验证证据合成答案。

    Args:
        state:          循环结束后的状态（requirement_status + evidences 已填充）
        llm:            合成模型（None=内部取 llm）
        stream_callback: 流式 token 回调 fn(content, is_reasoning)，用于前端逐字输出

    Returns:
        {"answer": str, "citations": [{"num","evidence_id","text","source","score"}, ...]}
    """
    citations = collect_validated(state)

    # 无任何已验证证据 → 用 LLM 生成诚实、有针对性的「无法回答」回复（不编造）
    if not citations:
        if llm is None:
            llm = _get_synthesis_llm()
        if llm is None:
            # 无模型可用：固定文案兜底，并推送给前端避免「[未获取到回答]」占位
            fallback = "抱歉，经过多轮检索，知识库中未找到足够的证据来回答该问题。"
            if stream_callback:
                stream_callback(fallback, False)
            return {"answer": fallback, "citations": []}
        answer = _generate_no_answer(state.question, llm, stream_callback)
        return {"answer": answer, "citations": []}

    if llm is None:
        llm = _get_synthesis_llm()

    if llm is None:
        # 无模型可用：直接拼接证据原文作为兜底答案
        answer = "\n\n".join(f"[{c['num']}] {c['text']}" for c in citations)
        return {"answer": answer, "citations": citations}

    answer = _generate_answer(state.question, citations, llm, stream_callback)
    return {"answer": answer, "citations": citations}
