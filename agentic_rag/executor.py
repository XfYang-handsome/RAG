# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Action Executor + State 更新（Phase 6）
================================================================================

执行 Controller 产出的 Action，并把结果写回 State，完成：
    State → Action → Observation → State'

Action 执行语义：

  - SEARCH        : route 选路 → retrieve 检索 → 新 Evidence 加入
  - REFINE_QUERY  : 用 rewrite 模型重写 query → route → retrieve → 新 Evidence
  - READ_PARENT   : 对 target_gap 关联的 evidence 读同 section 相邻 chunk（上下文扩展）
  - READ_SECTION  : 同 READ_PARENT（V1 二者等价，均为邻近块扩展）
  - WEB_SEARCH    : 联网搜索（知识库无新信息时受控补救）→ 新 Evidence 加入
  - ANSWER        : 不执行（交由 Synthesis）

State 更新固定顺序（与 PLAN Phase 6 一致）：
  1. evidences 增量合并（add_evidence 按 id 去重）
  2. actions 追加（register_action 计 tool_calls）
  3. queries 追加（add_query）
  4. iteration += 1

no_progress_rounds 由 Phase 10 编排层计算（依赖 snapshot + evaluate 结果），
executor 不负责。
================================================================================
"""

from __future__ import annotations

import hashlib
from typing import List

from common.text_utils import (
    contains_cjk,
    has_english_entity,
    parse_json,
    translate_to_en_keywords,
)

from .state import (
    AgentState,
    Action,
    ActionType,
    Evidence,
    EvidenceSource,
    Query,
    RetrievalTool,
)
from .retriever import retrieve, route, mode_to_tool
from .settings import get as _get

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 模型获取（rewrite_llm → tool_llm → llm，与 server 的 get_rewrite_llm 一致）
# ---------------------------------------------------------------------------
def _get_rewrite_llm():
    from llm_factory import get_model
    return get_model("rewrite", "tool_llm", "llm")


# ---------------------------------------------------------------------------
# REFINE_QUERY：重写查询
# ---------------------------------------------------------------------------
def _gap_description(state: AgentState, target_gap: str) -> str:
    for g in state.gaps:
        if g.requirement_id == target_gap:
            return g.missing_what
    return ""


def _refine_query(state: AgentState, action: Action, llm=None) -> str:
    """用 rewrite 模型基于 target_gap 重写查询；失败/未配置则返回原 query。"""
    if llm is None:
        llm = _get_rewrite_llm()
    if llm is None:
        return action.query or state.question

    from llm import invoke_llm

    gap_desc = _gap_description(state, action.target_gap)
    prompt = (
        f"用户问题：{state.question}\n"
        f"当前需要补充的信息：{gap_desc}\n"
        f"上一次的查询：{action.query or '（无）'}\n\n"
        "请改写一个更精准的检索查询（用于向量+关键词检索），"
        "只输出查询本身，不要加任何解释或引号。"
    )
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是查询重写助手。"),
            HumanMessage(content=prompt),
        ]).strip()
        return text if text else (action.query or state.question)
    except Exception:
        return action.query or state.question


# ---------------------------------------------------------------------------
# SEARCH：Multi-Query 生成（一个 Gap → 2~4 个 query）
# ---------------------------------------------------------------------------
def _generate_multi_queries(state: AgentState, action: Action, llm=None) -> List[str]:
    """基于 target_gap 生成多个不同角度的检索 query（Multi-Query Retrieval）。

    动机：单一 query 对「换一种说法就搜不到」的抽象需求（如跨语言的
    「应用场景」）召回不足。一个 Gap 生成 2~4 个互补 query，合并候选池，
    显著提升召回。

    失败 / 关闭开关 / 无模型时回退为单 query（原 action.query），不阻塞检索。
    """
    base = (action.query or "").strip()
    if not base:
        return []

    if not _get("search.multi_query_enabled", True):
        return [base]

    if llm is None:
        llm = _get_rewrite_llm()
    if llm is None:
        return [base]

    mn = _get("search.multi_query_min", 2)
    mx = _get("search.multi_query_max", 4)
    gap_desc = _gap_description(state, action.target_gap)

    from llm import invoke_llm

    prompt = (
        f"用户问题：{state.question}\n"
        f"当前信息缺口（{action.target_gap}）：{gap_desc or base}\n"
        f"原始检索词：{base}\n\n"
        f"请生成 {mn}~{mx} 个不同角度的英文检索 query（直接输出英文关键词或短语，"
        f"不要中文），每个 query 覆盖该信息缺口的不同侧面，用于在英文知识库中提升召回。\n"
        f"只输出 JSON 对象，格式："
        f'{{"queries": ["query1", "query2"]}}，不要任何解释。'
    )
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是多查询生成器，为同一信息缺口生成互补的英文检索 query。"),
            HumanMessage(content=prompt),
        ])
        obj = parse_json(text)
        if obj is None:
            return [base]
        qs = obj.get("queries")
        if not isinstance(qs, list):
            return [base]

        queries: List[str] = []
        seen = set()
        for q in qs:
            q = str(q).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            queries.append(q)
            if len(queries) >= mx:
                break
        if not queries:
            return [base]
        return queries[:mx]
    except Exception:
        return [base]


# ---------------------------------------------------------------------------
# READ_PARENT / READ_SECTION：上下文扩展
# ---------------------------------------------------------------------------
def _chunk_seq_from_id(eid: str) -> int:
    """从稳定 Evidence id（{doc_id}:c{seq}）提取 chunk_seq。"""
    try:
        if ":c" in eid:
            return int(eid.rsplit(":c", 1)[-1])
    except (ValueError, IndexError):
        pass
    return -1


def _read_context(state: AgentState, action: Action, window: int = None) -> List[Evidence]:
    """对 target_gap 关联的 evidence 做邻近块扩展，返回新 Evidence。

    仅结构树 chunk（有 doc_id + section_path + 可提取 chunk_seq）可扩展；
    普通父块（doc_id 为空）在检索时已带父块原文，无需再扩展。
    """
    if window is None:
        window = _get("context.neighbor_window", 2)

    from db_service import read_neighbor_chunks

    item = state.requirement_status.get(action.target_gap)
    if item is None or not item.evidence_ids:
        return []

    new_evs: List[Evidence] = []
    for eid in item.evidence_ids:
        ev = state.evidences.get(eid)
        if ev is None or not ev.source.doc_id:
            continue  # 普通父块 / 缺失，跳过
        seq = _chunk_seq_from_id(ev.source.chunk_id)
        if seq < 0:
            continue

        neighbors = read_neighbor_chunks(
            ev.source.doc_id,
            ev.source.section_path,
            seq,
            window=window,
        )
        for nb in neighbors:
            nb_eid = f"{nb['doc_id']}:c{nb['chunk_seq']}"
            if nb_eid in state.evidences:
                continue  # 已存在，去重
            new_evs.append(Evidence(
                id=nb_eid,
                text=nb.get("text", ""),
                score=0.0,  # 邻近块分数由 evaluate 的 rerank 统一覆盖
                source=EvidenceSource(
                    doc_id=nb.get("doc_id", ""),
                    section_path=nb.get("section_path", ""),
                    parent_id=nb.get("parent_id", ""),
                    chunk_id=nb_eid,
                ),
                produced_by_round=state.iteration,
            ))
    return new_evs


# ---------------------------------------------------------------------------
# WEB_SEARCH：联网搜索（知识库无新信息时的受控补救）
# ---------------------------------------------------------------------------



def _refine_web_query(state: AgentState, action: Action, llm=None) -> str:
    """联网前提炼搜索词——保留原始问题的完整检索意图（尤其对比语义）。

    背景：Controller 拆解需求后，WEB_SEARCH 的 target_gap 指向某个单点需求，
    action.query 是针对该单点生成的搜索词，已丢失「对比 / 区别 / 差异」等
    跨需求语义。而联网搜索需要完整的检索意图，故从 state.question 出发重提炼。

    规则：
      - 保留核心实体与专有名词（模型名/人名/地名），专名保留原文不翻译
      - 保留原始问题的对比/区别/差异/优劣语义，不要拆成单点
      - 语言选择：核心实体是英文专名 → 输出英文 query（走英文源）；否则中文问题中文 / 英文问题英文
      - 去掉疑问词/语气词，2~8 个词
    失败 / 无模型 → 回退 action.query（不阻塞联网）。
    """
    question = (state.question or "").strip()
    base = (action.query or "").strip() or question

    if llm is None:
        llm = _get_rewrite_llm()
    if llm is None:
        return base

    gap_desc = _gap_description(state, action.target_gap)

    # 语言规则：核心实体为英文专名时，强制英文 query 走英文源（质量更高）
    if has_english_entity(question):
        lang_rule = (
            "输出语言：原始问题含英文专名/技术名词（如 GPT-5、Gemini、OpenAI），"
            "即使问题用中文提问，也输出英文关键词，对比语义用英文表达"
            "（例如「GPT-5 vs Gemini 2.0 multimodal comparison」）。"
        )
    else:
        lang_rule = "输出语言：中文问题输出中文关键词，英文问题输出英文关键词。"

    from llm import invoke_llm

    prompt = (
        f"原始问题：{question}\n"
        f"当前需要补充的信息：{gap_desc}\n"
        f"初步搜索词：{base}\n\n"
        "请为联网搜索提炼一个精准的搜索关键词短语：\n"
        "1. 保留原始问题里的核心实体与专有名词（模型名/人名/地名等），专名保留原文不要翻译；\n"
        "2. 若原始问题含「对比/区别/差异/优劣」等语义，务必保留该对比语义（不要拆成单点）；\n"
        f"3. {lang_rule}\n"
        "4. 去掉疑问词、语气词、礼貌用语；\n"
        "5. 2~8 个词，只输出搜索短语本身，不要解释、不要引号。"
    )
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是联网搜索关键词提炼助手，专有名词保留原文，保留对比语义。"),
            HumanMessage(content=prompt),
        ]).strip()
        # 先取首行（丢弃多余解释），再去首尾引号（顺序重要：引号可能紧跟换行）
        first = (text.split("\n")[0] if text else "").strip()
        first = first.strip('"\'“”').strip()
        return first or base
    except Exception:
        return base


def _web_search(query: str, num: int = None, round_num: int = 0) -> List[Evidence]:
    """联网搜索，把结果转成 Evidence。

    - 受 mcp.features.websearch 总开关控制（关闭则返回空，不联网）
    - 失败返回空列表（不崩溃，由上层继续走 SEARCH/ANSWER）
    - Evidence id 用 url/text 的 md5 去重，origin="web" 标记非知识库来源
    """
    if num is None:
        num = _get("web_search.num", 5)

    query = (query or "").strip()
    if not query:
        return []

    from config_loader import cfg
    if not cfg("mcp.features.websearch", True):
        return []

    from mcp_service.websearch import _web_search as do_web_search

    try:
        results, engine = do_web_search(query, num=num)
    except Exception:
        return []

    # 正文抓取配置：仅对前 max_pages 条有 url 的结果抓正文，控制延迟 + 体积
    fetch_enabled = _get("web_search.fetch_body.enabled", True)
    fetch_max_pages = int(_get("web_search.fetch_body.max_pages", 3))
    fetch_max_chars = int(_get("web_search.fetch_body.max_chars", 2000))

    new_evs: List[Evidence] = []
    fetched = 0
    for r in results or []:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        parts = [x for x in (title, snippet) if x]

        # 正文抓取：失败（超时/非200/解析空）自动降级，仅保留 title+snippet
        if fetch_enabled and url and fetched < fetch_max_pages:
            from mcp_service.websearch import fetch_page_content
            body = fetch_page_content(url, max_chars=fetch_max_chars)
            if body:
                parts.append(body)
                fetched += 1

        text = "\n".join(parts).strip()
        if not text:
            continue
        eid = "web:" + hashlib.md5((url or text).encode("utf-8")).hexdigest()[:12]
        new_evs.append(Evidence(
            id=eid,
            text=text,
            # 联网结果无 rerank 分，给中性分；evaluate 会用 reranker 统一覆盖
            score=0.5,
            source=EvidenceSource(
                origin="web",
                url=url,
                summary=f"联网搜索·{engine}",
            ),
            produced_by_round=round_num,
        ))
    return new_evs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def execute(state: AgentState, action: Action, llm=None, top_k: int = None,
            retrieval_mode: str = None, reranker=None) -> int:
    """执行 action 并更新 state，返回新增 evidence 数（去重后）。

    Args:
        state:  当前状态
        action: 待执行的 Action
        llm:    重写模型（REFINE_QUERY 用，None=内部获取）
        top_k:  检索返回数量（None=读配置 agentic.search.top_k）
        retrieval_mode: 检索模式（"vector"/"hybrid"/"tree"）。指定时**强制**该检索路径，
                        覆盖 Router 的纯规则选路；None=维持原有 route 规则。
        reranker: Reranker 实例（tree 导航用，None=树导航降级为无 rerank）

    Returns:
        本轮新增的 evidence 数（不含已存在被去重的）。
    """
    if top_k is None:
        top_k = _get("search.top_k", 5)

    if action.type == ActionType.ANSWER:
        return 0

    # 显式指定检索模式时，强制对应 tool（tree 走纯 LLM 树导航）
    force_tool = mode_to_tool(retrieval_mode) if retrieval_mode else None

    tool = action.tool

    if action.type == ActionType.WEB_SEARCH:
        # 联网搜索：知识库无新信息时的受控补救（先提炼保留对比语义的搜索词）
        query = _refine_web_query(state, action, llm)
        new_evs = _web_search(query, round_num=state.iteration)
        state.web_search_count += 1
        state.add_query(Query(text=query, tool=RetrievalTool.WEB, round=state.iteration))
        state.register_action(action)
    elif action.type in (ActionType.SEARCH, ActionType.REFINE_QUERY):
        if action.type == ActionType.REFINE_QUERY:
            # 单 query：改写后的 query 直接检索（REFINE 本身就是精修单 query）
            query = _refine_query(state, action, llm)
            tool = force_tool or route(query, action.target_gap, _gap_description(state, action.target_gap))
            queries = [query]
        else:
            # SEARCH：Multi-Query 检索（一个 Gap → 2~4 个 query，合并候选池）
            query = action.query or ""
            tool = force_tool or route(query, action.target_gap, _gap_description(state, action.target_gap))
            queries = _generate_multi_queries(state, action, llm)

        new_evs: List[Evidence] = []
        # query 级去重：跳过已检索过的 query（避免重复请求 Milvus）
        seen_texts = {q.text.strip() for q in state.queries}
        for q in queries:
            # 已英文则跳过翻译（multi-query/refine 已要求输出英文检索词）；
            # 仅当 LLM 偶发返回中文时才翻译兜底，避免对英文 query 多余走翻译逻辑。
            if contains_cjk(q):
                q_en = (translate_to_en_keywords(q, llm) or "").strip()
            else:
                q_en = q.strip()
            if not q_en or q_en in seen_texts:
                continue
            seen_texts.add(q_en)
            new_evs.extend(retrieve(tool, q_en, top_k=top_k, round_num=state.iteration,
                                    reranker=reranker, llm=llm))
            # 记录实际检索的 query + 工具（翻译/改写后的 query 与 action.query 可能不同）
            state.add_query(Query(text=q_en, tool=tool, round=state.iteration))
        state.register_action(action)
    else:
        # READ_PARENT / READ_SECTION
        new_evs = _read_context(state, action)
        state.register_action(action)

    # evidences 增量合并（去重）
    added = 0
    for ev in new_evs:
        if state.add_evidence(ev):
            added += 1

    state.iteration += 1
    return added
