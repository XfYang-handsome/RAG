"""
================================================================================
LangGraph 检索链路 — RAG 多节点状态机
================================================================================

工作流（固定链路，无模型决策）：

  模式 A（direct）：model ──→ END
                    纯 LLM 推理，不检索知识库

  模式 B（rag）：   [rewrite?] → retrieve → rerank → grade ──(相关)──→ generate → END
                                                                  │
                                                                  └─(不相关)─→ 联网搜索补救 → generate

设计说明：
  1. RAG 链路固定写死（retrieve → rerank → grade → generate），
     retrieve/rerank/grade 均不参与模型决策；
  2. rewrite（查询重写）为可选入口节点，由 config.search.rewrite 开关控制（默认关闭）；
  3. 联网搜索统一作为「grade 判定不相关（评分过低/检索为空）」时的补救手段，
     由 config.mcp.features.websearch 开关控制；
  4. 工具决策（可插拔）：开启 config.mcp.tool_calling.enabled 后，由独立的
     「工具决策模型（tool_llm，用户可选配，未配置回退到 llm）」绑定 MCP 工具
     （mcp_service.tool_bridge 动态拉取）自主决定调用哪些工具；工具执行后，
     最终答案一律由「生成模型（llm，不绑定工具）」生成，实现决策与生成解耦 + 强制收敛。

依赖：
  - langgraph：状态机编排
  - langchain_core：消息对象 + 工具绑定
  - 复用本项目已有的 embedding / milvus_store / reranker / llm 模块
================================================================================
"""

import json
import re
import concurrent.futures
from typing import List, Optional, TypedDict

from common.text_utils import parse_json

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage,
)

from config_loader import config
from dsml_read import parse_dsml_tool_calls, has_dsml as _has_dsml


def _extract_reasoning(chunk) -> str:
    """
    从流式 chunk 中提取「思考过程」内容（deepseek-reasoner 等）。

    兼容多种位置：
      1. additional_kwargs["reasoning_content"]（langchain-openai 对 deepseek 的解析）
      2. additional_kwargs["reasoning"]（部分兼容 API）
      3. chunk.is_reasoning 标记（豆包 DoubaoStreamChunk，交给上层处理，此处忽略）

    Returns:
        思考内容片段；无则返回空字符串。
    """
    if chunk is None:
        return ""
    additional = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = additional.get("reasoning_content") or additional.get("reasoning")
    if reasoning:
        # reasoning_content 可能是字符串或列表
        if isinstance(reasoning, list):
            parts = []
            for r in reasoning:
                if isinstance(r, dict):
                    parts.append(r.get("text", ""))
                else:
                    parts.append(str(r))
            return "".join(parts)
        return str(reasoning)
    return ""


# ============================================================================
# DSML 工具调用解析（DeepSeek V4 系列 agentic 模型）
# ============================================================================
#
# 背景：DeepSeek-V4-Pro 等 agentic 模型在调用工具时，可能不遵循 OpenAI 的
# function calling 协议（返回结构化 tool_calls 字段），而是把 DSML 文本标记
# 直接输出到正文（content）里，例如：
#
#   <｜DSML｜tool_calls>
#   <｜DSML｜invoke name="web_search">
#   <｜DSML｜parameter name="arguments" string="false">{"query": "...", "num": 5}</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   </｜DSML｜tool_calls>
#
# 本模块负责：检测正文中的 DSML 标记、剥离展示内容、解析出工具名与参数。

# DSML 开始标记（全角竖线 ｜ 或半角 | 两种写法都兼容）
_DSML_START_MARKERS = ("<｜DSML", "<|DSML", "<DSML")


def _find_dsml_start(text: str) -> int:
    """返回 DSML 开始标记在 text 中的最早位置；无则返回 -1。"""
    pos = -1
    for marker in _DSML_START_MARKERS:
        idx = text.find(marker)
        if idx != -1 and (pos == -1 or idx < pos):
            pos = idx
    return pos


def _strip_dsml(text: str) -> str:
    """
    剥离文本中的 DSML 标记块（含未闭合的开始标签）。

    用于：模型在最终回答里仍残留 DSML 标记时，兜底清除。
    """
    if not text:
        return text
    # 1. 匹配完整的 DSML 块：<...DSML...> ... </...DSML...>
    text = re.sub(
        r'<[^>]*DSML[^>]*>.*?</[^>]*DSML[^>]*>',
        '',
        text,
        flags=re.DOTALL,
    )
    # 2. 清理残留的未闭合 DSML 标签（如 <｜DSML｜invoke name="...">）
    text = re.sub(r'<[^>]*DSML[^>]*>', '', text)
    return text


# ============================================================================
# 状态定义
# ============================================================================

class GraphState(TypedDict, total=False):
    """
    LangGraph 全局状态（在各节点间流转）。

    字段说明：
      mode           : 运行模式（"direct" 直接对话 / "rag" 知识库问答）
      question       : 当前要回答的问题（可能被 rewrite 节点重写）
      original_question: 用户最初的问题（重写后保留，用于最终回答）
      history        : 对话历史 [{"role","content"}, ...]
      documents      : 检索到的文档（父块原文，带 source/score/parent_id）
      reranked       : rerank 后的文档列表（按相关性降序）
      filtered       : 评分后确认相关的文档列表
      relevant       : 评分结论（True=相关 / False=不相关）
      retrieval_status: 结构化评估信号（"sufficient" / "weak" / "empty"，程序化产出）
      top_score      : reranker 最高分（结构化信号）
      evidence_count : 有效文档数（结构化信号）
      kb_retried     : 是否已做过 KB 内受控重试（保证单次，防循环）
      web_results    : 联网搜索结果（generate 内工具调用或补救，供生成参考）
      generation     : 最终生成的回答
      retry_count    : 查询重写重试次数
      max_retries    : 最大重试次数（超过则强制进入生成，兜底回答）
    """
    mode:               str
    question:           str
    original_question:  str
    question_type:      str   # "toc" / "section" / "normal"（问题类型路由）
    history:            List[dict]
    documents:          List[dict]
    reranked:           List[dict]
    filtered:           List[dict]
    relevant:           bool
    retrieval_status:   str   # "sufficient" / "weak" / "empty"
    top_score:          float
    evidence_count:     int
    kb_retried:         bool
    web_results:        List[dict]
    generation:         str
    retry_count:        int
    max_retries:        int


# ============================================================================
# RAGGraph — LangGraph 检索链路
# ============================================================================

class RAGGraph:
    """
    基于 LangGraph 的 RAG 检索链路封装。

    对外暴露入口方法 run(question, history, mode="rag")，内部按状态机流转：

      mode="direct" → model 节点直接推理
      mode="rag"    → retrieve → rerank → grade → (generate | 联网补救/rewrite 重试)

    依赖注入：retriever / reranker / llm 通过构造函数传入。

    retriever：检索函数 fn(query, top_k) -> [{"text","score","parent_id"}, ...]。
      默认由 server 传入「通过 MCP 工具 search_documents 检索」的实现，
      使主程序不直接连接数据库；若未提供则回退到本地 embedder+store。
    """

    def __init__(
        self,
        reranker,          # Reranker（精排）
        llm,               # ChatOpenAI / DoubaoLLM（生成模型：最终回答，不绑定工具）
        retriever=None,    # 检索函数 fn(query, top_k) -> list[dict]（走 MCP search_documents）
        embedder=None,     # [回退] ChatOpenAIEmbeddingWrapper（query 向量化）
        store=None,        # [回退] MilvusStore（父子块检索）
        tool_llm=None,     # 工具决策模型（可选，绑定工具决定调用哪些工具；未提供回退到 llm）
        rewrite_llm=None,  # 查询重写模型（可选；未配置时 rewrite 节点直接透传原问题，避免用 reasoning 模型重写导致慢 + 改坏查询）
        retrieval_top_k: int = None,
        rerank_top_n:      int = None,
        max_retries:       int = 2,
        use_rewrite:       bool = None,   # 是否启用 rewrite 入口节点（默认读 config.search.rewrite）
        use_hybrid:        bool = None,   # 是否启用混合检索 dense+BM25+RRF（默认读 config.search.hybrid）
        retrieval_mode:    str = None,    # 检索模式：vector/hybrid/tree（默认读 config.search.retrieval_mode）
        stream_chunk_callback=None,  # 可选：生成阶段的流式 token 回调 fn(token_str, is_reasoning)
        status_callback=None,        # 可选：节点状态回调 fn(status_str)
    ):
        self.reranker  = reranker
        self.llm       = llm
        self.retriever = retriever
        # 回退组件（retriever 未提供时使用）
        self.embedder  = embedder
        self.store     = store

        # 工具决策模型：独立于生成模型，未配置时回退到生成模型 llm
        self.tool_llm = tool_llm or llm

        # 查询重写模型：独立配置（models.json 的 rewrite 字段）。未配置时为 None，
        # rewrite 节点会直接透传原问题，不调用 LLM 重写。
        self.rewrite_llm = rewrite_llm

        self._stream_cb = stream_chunk_callback
        self._status_cb = status_callback

        _search_cfg = config["search"]
        self.retrieval_top_k = retrieval_top_k or _search_cfg["retrieval_top_k"]
        self.rerank_top_n    = rerank_top_n    or _search_cfg["rerank_top_n"]
        self.max_retries     = max_retries

        # rewrite 开关：显式传入优先，否则读 config.search.rewrite（默认关闭）
        if use_rewrite is None:
            use_rewrite = bool(_search_cfg.get("rewrite", False))
        self.use_rewrite = use_rewrite

        # 混合检索开关：显式传入优先，否则读 config.search.hybrid（默认关闭）
        if use_hybrid is None:
            use_hybrid = bool(_search_cfg.get("hybrid", False))
        self.use_hybrid = use_hybrid

        # 检索模式（vector / hybrid / tree）：显式传入优先，否则读 config.search.retrieval_mode。
        # 未配置时回退由 use_hybrid 推导（兼容旧配置：hybrid=true → "hybrid"，否则 "vector"）。
        if retrieval_mode is None:
            retrieval_mode = _search_cfg.get("retrieval_mode") or (
                "hybrid" if use_hybrid else "vector"
            )
        retrieval_mode = (retrieval_mode or "").strip().lower()
        if retrieval_mode not in ("vector", "hybrid", "tree"):
            retrieval_mode = "hybrid" if use_hybrid else "vector"
        self.retrieval_mode = retrieval_mode

        # KB 内受控重试（grade 判定检索不足时，先换 tree 重试一次，再兜底联网）。
        # 定位：确定性的架构增强，不是自由 Agent 循环——单次重试、强制收敛。
        _kb_retry_cfg = _search_cfg.get("kb_retry", {}) or {}
        self._kb_retry_enabled = bool(_kb_retry_cfg.get("enabled", True))
        self._kb_retry_mode = (_kb_retry_cfg.get("retry_mode") or "tree").strip().lower()
        if self._kb_retry_mode not in ("vector", "hybrid", "tree"):
            self._kb_retry_mode = "tree"

        # 联网搜索功能开关：关闭时彻底禁用联网补救
        self._websearch_enabled = bool(
            config.get("mcp", {}).get("features", {}).get("websearch", True)
        )

        # 工具调用（可插拔工具决策）：默认开启，由「每个工具是否启用」决定实际
        # 可调用哪些工具（单个工具的启停在 MCP 管理页独立管理）。
        # 若决策模型不支持 function calling（如 DeepSeek V4 以 DSML 输出），
        # _decide_and_run_tools 会自然回退，不会产生空答案。
        self._tool_calling_enabled = bool(
            config.get("mcp", {}).get("tool_calling", {}).get("enabled", True)
        )

        # 构建并编译状态图
        self._graph = self._build_graph()

    def _emit_status(self, msg: str):
        """向前端发送节点状态（若有回调）。"""
        if self._status_cb:
            try:
                self._status_cb(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 构建状态图
    # ------------------------------------------------------------------
    def _build_graph(self):
        """固定构建 LangGraph 状态图（direct / rag 两种模式）。"""
        graph = StateGraph(GraphState)

        # 注册节点
        graph.add_node("model",       self.model_node)      # direct 模式：纯 LLM
        graph.add_node("rewrite",     self.rewrite_node)    # 可选：查询重写
        graph.add_node("retrieve",    self.retrieve_node)
        graph.add_node("rerank",      self.rerank_node)
        graph.add_node("grade",       self.grade_node)
        graph.add_node("kb_retry",    self.kb_retry_node)   # 受控 KB 内重试（单次）
        graph.add_node("generate",    self.generate_node)

        # 入口：按 mode 路由（参数路由，非模型决策）
        graph.set_entry_point("entry")
        graph.add_node("entry", self._entry_router)
        graph.add_conditional_edges(
            "entry",
            self._route_entry,
            {"model": "model", "rag": "rewrite" if self.use_rewrite else "retrieve"},
        )

        # direct：model → END
        graph.add_edge("model", END)

        # rag 主流转（固定）
        if self.use_rewrite:
            graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "grade")

        # grade 后根据结构化评估决定：sufficient→generate；不足→受控 KB 重试（单次）
        graph.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {
                "generate":  "generate",
                "kb_retry":  "kb_retry",   # 受控 KB 内重试（tree 重试 / web 决策）
                "rewrite":   "rewrite",    # 兜底：KB 重试关闭且开启 rewrite 时走重写
            },
        )

        # KB 重试后强制收敛到 generate（generate 内部做最终 Web 兜底）
        graph.add_edge("kb_retry", "generate")

        # 终止节点
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # 路由函数
    # ------------------------------------------------------------------
    def _entry_router(self, state: GraphState) -> dict:
        """入口节点：直接透传 state（仅作为路由起点）。"""
        return {}

    def _route_entry(self, state: GraphState) -> str:
        """按 mode 路由：direct→model，rag→retrieve（或 rewrite）。"""
        return "model" if state.get("mode") == "direct" else "rag"

    def _route_after_grade(self, state: GraphState) -> str:
        """
        评分后的决策路由（规则路由，非模型决策）。

          - sufficient（检索足够好）→ generate
          - 不足（weak / empty）→
              * 若 KB 内受控重试开启且尚未重试 → kb_retry（换 tree 重试一次）
              * 否则 → rewrite 重试（未耗尽，且开启 rewrite）或 generate（Web 兜底）

        关键：KB 重试是「单次、受控」的（kb_retry 节点结束后固定进 generate，
        不会再回到这里），因此不会演变成自由循环。
        """
        status = state.get("retrieval_status") or (
            "sufficient" if state.get("relevant") else "empty"
        )

        if status == "sufficient":
            return "generate"

        # 不足：优先受控 KB 重试（单次）
        if self._kb_retry_enabled and not state.get("kb_retried", False):
            return "kb_retry"

        # KB 重试关闭 / 已重试：兜底
        retry_exhausted = state.get("retry_count", 0) >= state.get("max_retries", self.max_retries)
        if self.use_rewrite and not retry_exhausted:
            return "rewrite"
        return "generate"

    # ------------------------------------------------------------------
    # system prompt 构造（用户自定义优先）
    # ------------------------------------------------------------------
    @staticmethod
    def _custom_system_prompt() -> str:
        """读取用户自定义 system prompt（config.system_prompt，可为空）。"""
        try:
            sp = config.get("system_prompt", "") or ""
            return str(sp).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 节点：direct 模式纯 LLM 推理
    # ------------------------------------------------------------------
    def model_node(self, state: GraphState) -> dict:
        """direct 模式：直接 LLM 推理，不检索知识库（纯对话，不做工具决策）。"""
        question = state["question"]
        self._emit_status("AI 正在思考...")

        custom = self._custom_system_prompt()
        direct_format = (
            "回答格式要求（使用 Markdown 提升可读性）：\n"
            "1. 使用标题（##、###）划分主题，不要堆成一大段。\n"
            "2. 要点使用无序/有序列表逐条列出。\n"
            "3. 适当用**加粗**突出关键词，用短段落。\n"
            "4. 代码、命令、路径用行内代码（`code`）。\n"
            "5. 数学公式用 LaTeX 表示：行内公式用 \\(...\\) 包裹（如 \\(x^2\\)），"
            "块级公式用 \\[...\\] 单独成行包裹。"
        )
        if custom:
            system = custom + "\n\n" + direct_format
        else:
            system = "你是一个智能助手。请直接回答用户问题。\n\n" + direct_format
        messages = [SystemMessage(content=system)]
        for msg in state.get("history", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        generation = self._generate_stream(messages)
        return {"generation": generation}

    # ------------------------------------------------------------------
    # 节点 1：检索（子块 → 回溯父块）
    # ------------------------------------------------------------------
    def retrieve_node(self, state: GraphState) -> dict:
        """
        检索节点（含问题类型路由）：

          - toc     章节/目录类问题 → Tree 独立入口（返回文档目录结构）
          - section 章节定位类问题 → Tree 入口（match_sections + section_path 过滤）
          - normal  普通问题       → Vector/BM25 召回（Tree 作解释器/恢复器）

        Tree 的定位：不是第三个召回器，而是：
          1. 普通类：作为 Vector/BM25 召回结果的「结构解释器 + 上下文恢复器」
          2. 章节/目录类：作为独立检索入口
        """
        question = state["question"]

        qtype = self._classify_question(question)
        label = {"toc": "目录", "section": "章节定位", "normal": ""}.get(qtype, "")
        self._emit_status(("正在检索..." + (f"[{label}]" if label else "")))

        if qtype == "toc":
            docs = self._retrieve_toc()
            if not docs:  # 无树数据时兜底普通检索
                docs = self._retrieve_normal(question)
        elif qtype == "section":
            docs = self._retrieve_section(question)
            if not docs:  # 章节定位失败兜底普通检索（不让 Tree 入口成为单点）
                docs = self._retrieve_normal(question)
        else:
            docs = self._retrieve_normal(question)

        return {"documents": docs, "question_type": qtype}

    # ------------------------------------------------------------------
    # 问题类型分类（规则 + LLM 双保险）
    # ------------------------------------------------------------------
    def _classify_question(self, question: str) -> str:
        """把问题分类为 toc / section / normal。

        规则优先；规则不确定（含模糊结构词）时用 LLM 精判。
        """
        q = (question or "").strip()

        # 1. 规则：目录型（问文档有哪些章节/结构）
        if re.search(r"(目录|大纲|章节列表|有哪些章节|分为几[章节部分]|几[章节]|章节结构|章节目录|组织结构|文档结构)", q):
            return "toc"

        # 2. 规则：章节定位型（问第几章/某章节内容）
        if re.search(r"第[一二三四五六七八九十百\d]+[章节部分]|哪一[章节部分]|哪个章节|第几[章节]|在哪[章节]|属于哪[章节]", q):
            return "section"

        # 3. 规则不确定：含模糊结构词时用 LLM 精判
        if re.search(r"(章节|结构|部分|内容|小节)", q):
            try:
                return self._llm_classify(question)
            except Exception:
                pass

        return "normal"

    def _llm_classify(self, question: str) -> str:
        """LLM 精判问题类型（仅规则不确定时调用）。"""
        from llm import invoke_llm
        prompt = (
            "请判断用户问题属于哪一类，只输出一个词：\n"
            "1. 如果用户想了解文档的目录/章节结构/大纲，输出 toc\n"
            "2. 如果用户明确询问某个章节/小节的内容，输出 section\n"
            "3. 否则输出 normal\n\n"
            f"用户问题：{question}\n\n只输出 toc、section 或 normal 其中之一。"
        )
        text = invoke_llm(self.llm, [
            SystemMessage(content="你是问题分类助手，只输出 toc / section / normal 之一。"),
            HumanMessage(content=prompt),
        ]).strip()
        text = text.lower().strip('"\'').strip()
        for key in ("toc", "section", "normal"):
            if key in text:
                return key
        return "normal"

    # ------------------------------------------------------------------
    # 三类检索实现
    # ------------------------------------------------------------------
    def _retrieve_normal(self, question: str) -> List[dict]:
        """普通检索：Vector/BM25 召回（Tree 作解释器/恢复器，见 db_service 补全）。"""
        return self._retrieve_with_mode(question, self.retrieval_mode)

    def _retrieve_with_mode(self, question: str, mode: str) -> List[dict]:
        """用指定检索模式召回，并把结果统一成 docs 格式。

        供「默认检索」与「KB 内受控重试」复用（重试用 tree 模式时 mode="tree"）。
        """
        docs = []
        try:
            if self.retriever is not None:
                raw_results = self._call_retriever(question, mode) or []
            else:
                # 无 retriever 回退 store（tree 模式 store 不支持 → 空结果触发兜底）
                query_vector = self.embedder.embed_text(question)
                if mode == "tree":
                    raw_results = []
                elif mode == "hybrid" and hasattr(self.store, "search_hybrid"):
                    raw_results = self.store.search_hybrid(
                        query_vector, question, top_k=self.retrieval_top_k
                    )
                else:
                    raw_results = self.store.search(
                        query_vector, top_k=self.retrieval_top_k
                    )

            for r in raw_results:
                docs.append({
                    "text":      r.get("text", ""),
                    "score":     r.get("score", 0.0),
                    "parent_id": r.get("parent_id", ""),
                    "doc_id":    r.get("doc_id", ""),
                    "section_path": r.get("section_path_str", r.get("section_path", "")),
                    "is_neighbor": r.get("is_neighbor", False),
                })
        except Exception as e:
            docs.append({
                "text":  f"[检索失败: {e}]",
                "score": 0.0,
                "parent_id": "",
            })
        return docs

    def _call_retriever(self, question: str, mode: str = None):
        """调用 retriever，透传检索模式（兼容两参数/三参数签名）。"""
        m = mode or self.retrieval_mode
        try:
            return self.retriever(question, self.retrieval_top_k, m)
        except TypeError:
            # 旧版 retriever 仅接受 (query, top_k)：忽略 mode 参数再试一次
            return self.retriever(question, self.retrieval_top_k)

    def _retrieve_toc(self) -> List[dict]:
        """目录型检索（Tree 独立入口）：返回所有文档的目录结构。"""
        try:
            from db_service import get_toc_text
            toc = get_toc_text()
            if not toc:
                return []
            return [{
                "text":         toc,
                "score":        1.0,
                "parent_id":    "",
                "doc_id":       "",
                "section_path": "",
                "is_toc":       True,
            }]
        except Exception:
            return []

    def _retrieve_section(self, question: str) -> List[dict]:
        """章节定位型检索（Tree 独立入口）：match_sections + section_path 过滤。"""
        try:
            from db_service import retrieve_by_section_entry
            result = retrieve_by_section_entry(question, top_k=self.retrieval_top_k)
            docs = result.get("docs") or []
            if not docs:
                return []
            return [{
                "text":         d.get("text", ""),
                "score":        d.get("score", 0.0),
                "parent_id":    d.get("parent_id", ""),
                "doc_id":       d.get("doc_id", ""),
                "section_path": d.get("section_path_str", ""),
            } for d in docs]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 节点 2：Rerank 精排
    # ------------------------------------------------------------------
    def rerank_node(self, state: GraphState) -> dict:
        """
        Rerank 节点：用 BGE-Reranker 对候选文档重新排序。
        输入 state["documents"]，输出 state["reranked"]。
        """
        question = state["question"]
        docs     = state.get("documents", [])

        if not docs:
            return {"reranked": []}

        self._emit_status(f"正在 Rerank 精排（{len(docs)} 条候选）...")
        reranked = self._rerank_docs(question, docs)
        return {"reranked": reranked}

    def _rerank_docs(self, question: str, docs: List[dict]) -> List[dict]:
        """对候选文档做 Rerank 精排（KB 重试合并结果时复用）。"""
        if not docs:
            return []
        texts = [d["text"] for d in docs]
        try:
            ranked = self.reranker.rerank(question, texts, top_n=self.rerank_top_n)
            reranked = []
            for item in ranked:
                idx = item["index"]
                if idx >= len(docs):
                    continue
                reranked.append({
                    "text":      item["text"],
                    "score":     item["score"],
                    "parent_id": docs[idx].get("parent_id", ""),
                    "doc_id":    docs[idx].get("doc_id", ""),
                    "section_path": docs[idx].get("section_path", ""),
                    "is_neighbor": docs[idx].get("is_neighbor", False),
                })
            return reranked
        except Exception:
            # rerank 失败降级：直接用原顺序，截断到 top_n
            return docs[:self.rerank_top_n]

    # ------------------------------------------------------------------
    # 节点 3：相关性评分
    # ------------------------------------------------------------------
    def grade_node(self, state: GraphState) -> dict:
        """
        评分节点：判断检索/重排后的文档是否与问题相关，并产出「结构化评估信号」。

        输出三态信号（程序化，非 LLM 直接定义「结果好不好」）：
          - retrieval_status: "sufficient"（足够）/ "weak"（有结果但相关性不足）/ "empty"（无结果）
          - top_score:       reranker 最高分
          - evidence_count:  有效文档数

        判定规则：
          - 无结果 → empty
          - reranker 最高分 ≥ 阈值 → sufficient
          - 否则 LLM 精判：相关 → sufficient，不相关 → weak
        """
        question = state["question"]
        docs     = state.get("reranked", []) or state.get("documents", [])

        if not docs:
            return {
                "relevant":        False,
                "filtered":        [],
                "retrieval_status": "empty",
                "top_score":       0.0,
                "evidence_count":  0,
            }

        _grade_threshold = float(
            config.get("search", {}).get("grade_relevance_threshold", 0.25)
        )
        scores = [d.get("score", 0.0) for d in docs]
        best = max(scores) if scores else 0.0
        if best >= _grade_threshold:
            self._emit_status(f"文档相关（最高相关度 {best:.3f} ≥ {_grade_threshold}），跳过 LLM 评分")
            return {
                "relevant":        True,
                "filtered":        docs,
                "retrieval_status": "sufficient",
                "top_score":       best,
                "evidence_count":  len(docs),
            }

        self._emit_status("正在评估文档相关性...")

        doc_list = "\n\n".join(
            f"[文档 {i}] {d['text'][:800]}" for i, d in enumerate(docs)
        )
        prompt = (
            "你是一个检索结果相关性评估器。请判断下列检索到的文档是否与用户问题相关。\n\n"
            f"用户问题：{question}\n\n"
            f"检索到的文档：\n{doc_list}\n\n"
            "请严格只输出 JSON 对象，格式如下（不要输出任何其他文字）：\n"
            '{"relevant": true 或 false, "reason": "简要说明"}'
        )

        relevant = False
        try:
            from llm import invoke_llm
            # 用工具决策模型（温度 0、确定性）做相关性评估，而非主 LLM（reasoning 模型）。
            # 主 LLM 默认 thinking 开启，评估这类「要快」的场景会先跑一大段思考再输出，
            # 表现为卡住十几秒甚至更久。
            eval_llm = self.tool_llm
            def _do_grade():
                return invoke_llm(eval_llm, [
                    SystemMessage(content="你是相关性评估器，只输出 JSON。"),
                    HumanMessage(content=prompt),
                ])
            # 加超时保护：评估非关键路径，超时（如模型卡住/限流）按「相关」处理，
            # 不阻塞整条链路，保证对话能继续。
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _fut = _ex.submit(_do_grade)
            try:
                text = _fut.result(timeout=20)
            except Exception:
                _ex.shutdown(wait=False)
                relevant = True
            else:
                _ex.shutdown(wait=False)
                obj = parse_json(text)
                if obj is not None:
                    relevant = bool(obj.get("relevant", False))
        except Exception:
            relevant = True

        filtered = docs if relevant else []
        status = "sufficient" if relevant else "weak"
        return {
            "relevant":        relevant,
            "filtered":        filtered,
            "retrieval_status": status,
            "top_score":       best,
            "evidence_count":  len(filtered),
        }

    # ------------------------------------------------------------------
    # 节点 3.5：KB 内受控重试（单次，强制收敛到 generate）
    # ------------------------------------------------------------------
    def kb_retry_node(self, state: GraphState) -> dict:
        """
        受控 KB 内重试节点：当 grade 判定检索「不足」（weak / empty）时，
        在「换 tree 重试知识库」与「直接联网」之间做**一次**决策，重试后
        无论结果如何都进入 generate（generate 内部做最终 Web 兜底）。

        定位：确定性的架构增强，不是自由 Agent 循环。约束：
          1. 单次执行（路由层保证，且 kb_retried 标志兜底）；
          2. 决策空间固定为 retry_tree / web_search 二选一；
          3. 结构化信号（retrieval_status/top_score/evidence_count）由
             grade 程序化产出，LLM 只做「换 tree 还是联网」的受限判断。
        """
        question = state["question"]
        status = state.get("retrieval_status") or "weak"
        top_score = float(state.get("top_score", 0.0) or 0.0)
        evidence_count = int(state.get("evidence_count", 0) or 0)
        retry_mode = self._kb_retry_mode

        base = {"kb_retried": True}

        # 规则短路 1：知识库为空 → 重试无意义，直接 Web 兜底
        if self._knowledge_base_empty():
            self._emit_status("知识库为空，跳过 KB 重试，走联网兜底")
            return {**base, "retrieval_status": "empty"}

        # 规则短路 2：默认模式已等于重试模式（如已是 tree）→ 换模式无增益，
        #            直接 Web 兜底（避免「tree 不足 → 再 tree」的空转）
        if self.retrieval_mode == retry_mode:
            self._emit_status(f"默认已是 {retry_mode} 模式且结果不足，跳过重复重试，走联网兜底")
            return base

        # 受控 LLM 决策：retry_tree / web_search（TOC 作为「知识库范围」上下文输入）
        docs_brief = self._docs_brief(state.get("reranked", []))
        toc_brief = self._get_toc_brief()
        action = self._decide_kb_retry(
            question, status, top_score, evidence_count, docs_brief, toc_brief
        )

        if action != "retry_tree":
            self._emit_status("决策：知识库内无相关内容，走联网兜底")
            return base

        # 执行受控重试：换 retry_mode（默认 tree）重新检索
        self._emit_status(f"检索结果不足，尝试用 {retry_mode} 模式重试知识库...")
        retry_docs = self._retrieve_with_mode(question, retry_mode)
        if not retry_docs:
            self._emit_status(f"{retry_mode} 重试无结果，走联网兜底")
            return {**base, "retrieval_status": "empty"}

        # 合并旧结果 + 重试结果 → rerank → 程序化判定是否 sufficient
        merged = self._rerank_docs(question, retry_docs)
        threshold = float(
            config.get("search", {}).get("grade_relevance_threshold", 0.25)
        )
        best = max([d.get("score", 0.0) for d in merged], default=0.0)
        if best >= threshold:
            self._emit_status(f"KB 重试命中（最高相关度 {best:.3f} ≥ {threshold}）")
            return {
                **base,
                "reranked":         merged,
                "filtered":         merged,
                "retrieval_status": "sufficient",
                "top_score":        best,
                "evidence_count":   len(merged),
                "relevant":         True,
            }
        else:
            self._emit_status(f"KB 重试仍不足（最高相关度 {best:.3f}），走联网兜底")
            return {**base, "retrieval_status": "weak", "top_score": best}

    def _knowledge_base_empty(self) -> bool:
        """判断知识库是否为空（是否有已入库文档）。"""
        try:
            from db_service import list_documents
            return not list_documents()
        except Exception:
            return False  # 查询失败保守假定非空，交给后续决策

    def _get_toc_brief(self, max_chars: int = 1200) -> str:
        """获取知识库目录概要（作为「问题是否在知识库范围」的判断上下文）。"""
        try:
            from db_service import get_toc_text
            toc = get_toc_text()
            return (toc or "")[:max_chars]
        except Exception:
            return ""

    @staticmethod
    def _docs_brief(docs: List[dict], limit: int = 3, chars: int = 160) -> str:
        """把已检索文档摘要成简短文本（供决策参考，控制 prompt 体积）。"""
        if not docs:
            return "（无）"
        lines = []
        for d in docs[:limit]:
            t = (d.get("text", "") or "").replace("\n", " ")[:chars]
            lines.append(f"- [score={float(d.get('score', 0.0) or 0.0):.3f}] {t}")
        return "\n".join(lines)

    def _decide_kb_retry(self, question: str, status: str, top_score: float,
                         evidence_count: int, docs_brief: str, toc_brief: str) -> str:
        """受控决策：返回 "retry_tree" 或 "web_search"（单次、严格约束）。

        决策模型只允许在「换 tree 重试」与「联网」之间二选一，禁止自由调用
        其它工具、禁止多次决策。TOC 作为「知识库覆盖范围」的上下文输入，
        帮助模型判断「问题是否可能在知识库内」。
        """
        status_label = {"empty": "无检索结果", "weak": "有结果但相关性不足"}.get(
            status, status
        )
        prompt = (
            "你是知识库检索的补救决策器。用户问题在默认检索模式下没有检索到足够相关的内容，"
            "现在需要你在两个受控选项里做**一次**决策（不要调用任何工具、不要输出其它内容）：\n\n"
            f"用户问题：{question}\n"
            f"当前检索状态：{status_label}\n"
            f"检索信号：最高相关度={top_score:.3f}，有效文档数={evidence_count}\n\n"
            f"已检索内容摘要：\n{docs_brief}\n\n"
            f"知识库目录（章节大纲）：\n{toc_brief or '（知识库为空或无目录）'}\n\n"
            "请判断：用户问题**是否可能**在知识库范围内有答案？\n"
            " - 若目录/已有内容表明「可能相关、值得换纯树导航重试」，输出 {\"action\": \"retry_tree\"}\n"
            " - 若知识库明显没有相关内容（或知识库为空），输出 {\"action\": \"web_search\"}\n\n"
            "只输出 JSON：{\"action\": \"retry_tree\"} 或 {\"action\": \"web_search\"}，不要任何解释。"
        )
        try:
            from llm import invoke_llm
            text = invoke_llm(self.tool_llm, [
                SystemMessage(content="你是检索补救决策器，只输出 retry_tree 或 web_search 之一。"),
                HumanMessage(content=prompt),
            ]).strip()
            obj = parse_json(text)
            if obj is not None:
                action = str(obj.get("action", "")).strip().lower()
                if action in ("retry_tree", "web_search"):
                    return action
        except Exception:
            pass
        # 兜底规则（无 LLM）：知识库有目录 → 重试 tree；否则联网
        return "retry_tree" if toc_brief else "web_search"

    # ------------------------------------------------------------------
    # 节点 4：查询重写（可选入口 + 补救重试）
    # ------------------------------------------------------------------
    def rewrite_node(self, state: GraphState) -> dict:
        """
        查询重写节点（可选）：入口优化 + 补救重试（直接用本地 LLM）。
        """
        question = state["question"]
        already_searched = bool(state.get("documents"))

        if already_searched:
            self._emit_status("检索结果不相关，正在重写查询...")
        else:
            self._emit_status("正在优化查询...")

        rewritten = self._rewrite_local(question)

        retry_count = state.get("retry_count", 0)
        if already_searched:
            retry_count += 1

        return {
            "question":     rewritten,
            "retry_count":  retry_count,
        }

    def _rewrite_local(self, question: str) -> str:
        """重写查询（用独立 rewrite_llm；未配置时直接透传原问题）。

        说明：DeepSeek V4 Pro 等 reasoning/agentic 模型做「查询重写」这类简单
        任务会触发深度思考（单次实测约 11.5s），且重写质量不稳定、可能把好查询
        改坏导致检索变差。因此 rewrite 默认走独立轻量模型（models.json 的
        rewrite 字段）；未配置独立模型时直接返回原问题，避免慢 + 改坏查询。
        """
        if self.rewrite_llm is None:
            # 未配置独立重写模型：透传原问题（等价于跳过重写）
            self._emit_status("未配置独立重写模型，跳过查询重写")
            return question

        prompt = (
            "你是一个查询重写助手。用户的问题检索效果不佳，"
            "请换个角度、更清晰地重新表述这个问题，以提高检索召回率。\n\n"
            f"原问题：{question}\n\n"
            "只输出重写后的问题本身，不要加任何解释或引号。"
        )
        try:
            from llm import invoke_llm
            text = invoke_llm(self.rewrite_llm, [
                SystemMessage(content="你是查询重写助手。"),
                HumanMessage(content=prompt),
            ]).strip()
            return text if text else question
        except Exception:
            return question

    # ------------------------------------------------------------------
    # 节点 5：生成回答（确定性联网：grade 判定不相关时自动搜索）
    # ------------------------------------------------------------------
    def generate_node(self, state: GraphState) -> dict:
        """
        生成节点：基于（相关）文档回答用户问题。

        工具决策（可插拔）：
          - 开启 mcp.tool_calling.enabled 且决策模型支持 function calling 时，
            由「决策模型（tool_llm）」绑定工具自主决定要调用哪些工具
            （如 web_search），执行结果并入上下文；
          - 未开启时回退「确定性联网」：grade 判定不相关才自动搜索一次。

        收敛设计：无论工具决策与否，最终答案一律由「生成模型（self.llm，
        不绑定任何工具）」直接生成，避免 agentic 模型在生成阶段反复输出
        DSML 工具标记、生成不出答案的问题。
        """
        original = state.get("original_question") or state["question"]
        docs     = state.get("filtered") or state.get("reranked") or state.get("documents", [])

        # 组装知识库上下文（含结构树章节路径补全 + 邻近块标记 + 来源可读化）
        if docs:
            # 来源可读化：doc_id → 文档标题缓存（一次性查全，避免逐条查询）
            _title_cache = {}
            try:
                import tree_store
                for _d in tree_store.list_documents():
                    _title_cache[_d.get("doc_id") or ""] = (
                        _d.get("title") or _d.get("source") or ""
                    )
            except Exception:
                pass

            def _readable_source(d):
                """把检索结果补成「可读来源」：文档标题 + 章节标题路径。"""
                doc_id = d.get("doc_id") or ""
                title = _title_cache.get(doc_id, "")
                path = ""
                # 优先用已有的标题路径（hybrid 检索已带 section_path_titles）
                st = d.get("section_path_titles")
                if st:
                    path = " > ".join(str(x) for x in st)
                else:
                    raw = d.get("section_path_str") or ""
                    # 数字路径（如 "0/1"）不友好，丢弃，下面用树库重查标题路径
                    if raw and not re.fullmatch(r"[0-9\s/>]+", raw):
                        path = raw
                if not path and doc_id:
                    try:
                        import tree_store
                        titles = tree_store.get_section_path_titles(
                            doc_id, d.get("parent_id", "")
                        )
                        if titles:
                            path = " > ".join(titles)
                    except Exception:
                        pass
                return title, path

            context_parts = []
            for i, d in enumerate(docs):
                title, path = _readable_source(d)
                if d.get("is_neighbor"):
                    # 邻近块：补充上下文，标注其来源关系
                    header = f"[相邻上下文 {i} | 章节: {path}]" if path else f"[相邻上下文 {i}]"
                else:
                    parts = [f"来源 {i}"]
                    if title:
                        parts.append(f"文档: {title}")
                    if path:
                        parts.append(f"章节: {path}")
                    parts.append(f"相关度 {d.get('score', 0):.4f}")
                    header = "[" + " | ".join(parts) + "]"
                context_parts.append(f"{header}\n{d['text']}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "（未检索到相关文档）"

        # 联网/工具补充：优先可插拔工具决策，否则回退确定性联网。
        # 传入历史上下文，使多轮追问（如「那菲尔兹奖颁给谁了」）的
        # 搜索/决策能理解指代关系。
        #
        # relevant 由结构化信号 retrieval_status 推导（grade / kb_retry 程序化
        # 产出），保证「重试命中 → 不联网」「重试仍不足 → 联网兜底」稳定一致。
        status = state.get("retrieval_status")
        if status == "sufficient":
            relevant = True
        elif status in ("weak", "empty"):
            relevant = False
        else:
            relevant = state.get("relevant", True)
        history = state.get("history", [])
        tool_text = ""
        if self._tool_calling_enabled and self._supports_function_calling(self.tool_llm):
            tool_text = self._decide_and_run_tools(original, context, relevant, history)
        elif not relevant and self._websearch_enabled:
            # 确定性联网兜底：grade 判定不相关时自动搜索一次
            tool_text = self._deterministic_web_search(original, history)

        # 通用输出约束：无论是否使用自定义提示词，都强制保留 Markdown 排版
        # 与「注明信息依据」要求，避免自定义 prompt 覆盖后输出退化为纯文本。
        format_guide = (
            "回答格式要求（使用 Markdown 提升可读性）：\n"
            "1. 使用标题（##、###）划分主题，不要堆成一大段。\n"
            "2. 要点使用无序/有序列表逐条列出。\n"
            "3. 适当用**加粗**突出关键词，用短段落。\n"
            "4. 代码、命令、路径用行内代码（`code`）。\n"
            "5. 数学公式用 LaTeX 表示：行内公式用 \\(...\\) 包裹（如 \\(x^2\\)），"
            "块级公式用 \\[...\\] 单独成行包裹。\n"
            "6. 每个关键论断后必须标注来源编号，如 [来源1] 或 [来源1][来源2]，"
            "编号对应上方「检索内容」里每个条目的 [来源 N] 标题。\n"
            "7. 回答结尾必须单独列出一节「信息来源」，逐条写明："
            "[来源N] 文档名 + 章节路径。没有依据支撑的内容不要写。\n"
            "8. 回答完整但简洁：直接给出问题各要点的答案，省略与问题无关的"
            "背景和延伸内容，长度与问题复杂度匹配，不为了「显得详尽」而凑篇幅。\n"
            "9. 严格按上面系统提示词的角色设定与风格要求回答，忽略对话历史中"
            "与当前设定不一致的语气或风格（历史仅用于理解上下文，不得模仿其措辞风格）。"
        )

        # 工具调用/联网结果独立成段，并明确告知生成模型：这是可信信息，
        # 可直接用于回答。此前把它混在「检索内容」里，被"只依据检索内容"
        # 的约束压制，导致明明算了结果却仍答"无法回答"。
        tool_section = ""
        if tool_text:
            tool_section = (
                f"\n\n=== 工具调用结果 ===\n{tool_text}\n"
                "（以上是工具调用/联网搜索得到的可信结果，可直接用于回答用户问题，"
                "无需标注来源编号。）\n=== 结束 ==="
            )

        custom = self._custom_system_prompt()
        if custom:
            # 用户自定义 prompt 作为角色/任务设定，通用格式约束始终保留
            system = (
                custom
                + "\n\n"
                + format_guide
                + "\n\n请优先基于以下检索内容回答：\n"
                + f"=== 检索内容 ===\n{context}\n=== 结束 ==="
                + tool_section
            )
        else:
            system = (
                "你是 RAG 知识助手。请基于下面提供的检索内容和工具调用结果回答用户问题。"
                "若两者都不足请明确告知。\n\n"
                + format_guide
                + f"\n\n=== 检索内容 ===\n{context}\n=== 结束 ==="
                + tool_section
            )

        messages = [SystemMessage(content=system)]
        for msg in state.get("history", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=original))

        self._emit_status("AI 正在思考...")

        generation = self._generate_stream(messages)
        return {"generation": generation}

    # ------------------------------------------------------------------
    # 可插拔工具决策（决策模型 bind 工具 → 执行 → 收敛给生成模型）
    # ------------------------------------------------------------------
    @staticmethod
    def _supports_function_calling(llm) -> bool:
        """
        判断模型是否支持 function calling（bind_tools）。

        ChatOpenAI（OpenAI 兼容）提供 bind_tools → True；
        DoubaoLLM（豆包 Responses API）不提供 → False。

        说明：即使模型不遵循 OpenAI function calling 协议（如 DeepSeek V4 以
        DSML 文本表达工具调用），只要它通过 ChatOpenAI 实例化（有 bind_tools），
        工具决策仍会尝试；具体调用时由 dsml_read 解析兜底处理
        （见 _decide_and_run_tools）。
        """
        return hasattr(llm, "bind_tools")

    def _decide_and_run_tools(self, question: str, context: str, relevant: bool, history: list = None) -> str:
        """
        可插拔工具决策：让「决策模型（self.tool_llm）」绑定工具，自主决定
        要调用哪些工具，执行后返回结果文本（供 generate_node 并入上下文）。

        history 用于让决策模型理解多轮指代（如「再搜一下刚才那个」）。

        收敛设计：本方法只负责「决策 + 执行工具」，不产出最终答案；
        最终答案由生成模型（self.llm，不绑定工具）在 generate_node 中生成，
        从而避免 agentic 模型反复输出 DSML 工具标记、生成不出答案的问题。
        """
        from mcp_service.tool_bridge import get_mcp_tools_as_langchain
        enabled_tools = get_mcp_tools_as_langchain()
        if not enabled_tools:
            return ""

        tool_map = {t.name: t for t in enabled_tools}
        try:
            llm_with_tools = self.tool_llm.bind_tools(enabled_tools)
        except Exception:
            return ""

        self._emit_status("正在决策工具调用...")

        system = (
            "你是工具调用决策器。请根据用户问题自主判断：回答该问题是否需要调用外部工具。"
            "可用的工具列表见下方。你需要什么信息就调用对应工具；"
            "不需要任何工具就不调用。只做工具决策，不要直接回答问题。"
        )
        # 把知识库检索情况如实告知决策模型，但不做"该不该调工具"的引导——
        # 调不调工具完全由模型根据问题性质自主判断。
        kb_hint = (
            "【知识库检索情况】未检索到与用户问题相关的内容。"
            if not relevant else
            "【知识库检索情况】已检索到部分内容（可能相关也可能不相关，请自行判断）。"
        )
        hist_brief = self._history_brief(history or [])
        if hist_brief:
            user_prompt = (
                f"对话历史：\n{hist_brief}\n\n"
                f"用户最新问题：{question}\n\n"
                f"{kb_hint}\n\n"
                f"知识库检索上下文：\n{context}\n\n"
                "请判断是否需要调用工具补充信息。"
            )
        else:
            user_prompt = (
                f"用户问题：{question}\n\n"
                f"{kb_hint}\n\n"
                f"知识库检索上下文：\n{context}\n\n"
                "请判断是否需要调用工具补充信息。"
            )
        try:
            resp = llm_with_tools.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user_prompt),
            ])
        except Exception as e:
            self._emit_status(f"工具决策失败，跳过: {e}")
            return ""

        tool_calls = getattr(resp, "tool_calls", None) or []

        # DSML 兜底：DeepSeek V4 等 agentic 模型不返回结构化 tool_calls，
        # 而是把工具调用以 DSML 文本输出到正文里。此处解析正文中的 DSML，
        # 转为结构化 tool_calls，使这类模型也能成功调用工具。
        if not tool_calls:
            content = getattr(resp, "content", "") or ""
            if content and _has_dsml(content):
                _, dsml_calls = parse_dsml_tool_calls(content)
                if dsml_calls:
                    self._emit_status("检测到 DSML 工具调用，正在解析...")
                    tool_calls = dsml_calls
        if not tool_calls:
            return ""

        results = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            # args 可能是字符串（部分模型），尝试解析为 dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            if not isinstance(args, dict):
                args = {}
            tool = tool_map.get(name)
            if tool is None:
                continue
            # 工具决策模型（tool_llm）已通过 function calling 按 system prompt
            # 提炼好 query（保留专名原文、中文用中文，实测正确）。此处**不再二次
            # 提炼**——二次提炼会引入额外 LLM 调用，且可能把已正确的中文专名
            # 改坏（如改回日文原名「紙の上の魔法使い」导致搜不到）。
            # 最终清洗由 _web_search 入口的 _sanitize_query 兜底。
            if name == "web_search" and isinstance(args.get("query"), str):
                args["query"] = args["query"].strip()
            self._emit_status(f"调用工具: {name}")
            try:
                output = tool.invoke(args)
                results.append(f"[工具 {name}]\n{output}")
            except Exception as e:
                results.append(f"[工具 {name} 调用失败: {e}]")

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # 流式收集：剥离 DSML 标记与思考过程
    # ------------------------------------------------------------------
    def _stream_collect(self, chunks) -> dict:
        """
        流式遍历 chunks：实时剥离 DSML 标记与思考过程，累积结构化 tool_calls。

        返回：
          {
            "generation": 剥离 DSML 后的正文（已推前端）,
            "full":       完整正文（含 DSML，用于解析工具调用）,
            "tool_calls": {index: {id,name,args}} 结构化工具调用,
          }
        """
        generation = ""
        full = ""
        tool_calls_by_index = {}
        dsml_active = False
        pending = ""
        keep = max(len(m) for m in _DSML_START_MARKERS)

        for chunk in chunks:
            # 思考过程（deepseek-reasoner 的 reasoning_content）
            reasoning = _extract_reasoning(chunk)
            if reasoning:
                if self._stream_cb is not None:
                    self._stream_cb(reasoning, True)
                continue

            content = getattr(chunk, "content", None)
            if isinstance(content, list):
                content = ""
            content = content or ""

            if content:
                # 豆包 reasoning 标记（chunk.is_reasoning=True 时 content 为思考过程）
                if bool(getattr(chunk, "is_reasoning", False)):
                    if self._stream_cb is not None:
                        self._stream_cb(content, True)
                    continue

                full += content
                if dsml_active:
                    # 已进入 DSML：只累积（用于解析），不推前端
                    pass
                else:
                    pending += content
                    idx = _find_dsml_start(pending)
                    if idx != -1:
                        if idx > 0:
                            generation += pending[:idx]
                            if self._stream_cb is not None:
                                self._stream_cb(pending[:idx], False)
                        dsml_active = True
                        pending = ""
                    else:
                        if len(pending) > keep:
                            flush = pending[:-keep]
                            generation += flush
                            if self._stream_cb is not None:
                                self._stream_cb(flush, False)
                            pending = pending[-keep:]

            # 累积结构化 tool_calls
            for tc in (getattr(chunk, "tool_call_chunks", None) or []):
                idx = tc.get("index", 0)
                entry = tool_calls_by_index.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    entry["id"] += tc["id"]
                if tc.get("name"):
                    entry["name"] += tc["name"]
                if tc.get("args"):
                    entry["args"] += tc["args"]

        # 流结束：flush 剩余正常缓冲
        if not dsml_active and pending:
            generation += pending
            if self._stream_cb is not None:
                self._stream_cb(pending, False)

        return {"generation": generation, "full": full, "tool_calls": tool_calls_by_index}

    @staticmethod
    def _history_brief(history: list, limit: int = 6) -> str:
        """
        把最近若干条对话历史格式化成简短文本，供工具决策 / 搜索提炼参考。

        多轮追问（如「那菲尔兹奖颁给谁了」）需要结合上文才能理解指代，
        因此在决策/搜索阶段也带上历史上下文。只取最近 limit 条、每条截断
        200 字，避免 token 过长。
        """
        if not history:
            return ""
        recent = history[-limit:] if len(history) > limit else history
        lines = []
        for m in recent:
            role = "用户" if m.get("role") == "user" else "助手"
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _refine_search_query(self, question: str, history: list = None) -> str:
        """
        用 LLM 把用户问题提炼成适合搜索引擎的简洁关键词短语。

        关键：**不强制翻译成英文**。专有名词（书名号《》内容、作品名、人名、
        地名、日文/中文专名、英文术语）必须保留原文，否则像《纸上的魔法使》
        这类日文 Galgame 游戏名会被意译成英文（paper magician 之类）导致搜不到。

        中文问题输出中文关键词，英文问题输出英文关键词，只去掉疑问词/语气词。

        提供 history 时会结合对话历史理解多轮指代（如「再搜一下刚才那个」）。
        """
        hist_brief = self._history_brief(history or [])
        base_rules = (
            "1. 去掉疑问词、语气词、礼貌用语；\n"
            "2. 保留核心实体、概念、时间限定；\n"
            "3. 专有名词（书名号《》内的内容、作品名、人名、地名、日文/中文专名、英文术语）必须保留原文，不要翻译；\n"
            "4. 中文问题输出中文关键词，英文问题输出英文关键词；\n"
            "5. 尽量 2~6 个词；\n"
            "6. 同一个专有名词若有中文、日文、英文多种写法，只保留一种（优先中文，搜索引擎对中文召回更好）。"
        )
        if hist_brief:
            prompt = (
                "请结合对话历史，把用户的最新问题提炼成适合搜索引擎的关键词短语：\n\n"
                f"{base_rules}\n\n"
                f"对话历史：\n{hist_brief}\n\n"
                f"最新问题：{question}\n\n只输出关键词短语本身，不要解释、不要加引号。"
            )
        else:
            prompt = (
                "请把下面的用户问题提炼成适合搜索引擎的关键词短语：\n\n"
                f"{base_rules}\n\n"
                f"{question}\n\n只输出关键词短语本身，不要解释、不要加引号。"
            )
        try:
            from llm import invoke_llm
            text = invoke_llm(self.llm, [
                SystemMessage(content="你是搜索关键词提炼助手，只输出关键词短语，专有名词保留原文不翻译。"),
                HumanMessage(content=prompt),
            ]).strip()
            text = text.strip('"\'').strip()
            if not text:
                return question
            # 防止模型输出多行/附带解释，取第一行
            return text.split("\n")[0].strip() or question
        except Exception:
            return question

    @staticmethod
    def _extract_fallback_query(question: str) -> str:
        """
        兜底查询：优先提取书名号《》/引号内的专有名词，否则返回去掉疑问词的原文。

        当 LLM 提炼出的关键词搜不到时，用原始专有名词直接搜一次，
        避免「翻译/提炼失败」导致彻底搜不到（如日文/中文作品名）。
        """
        q = (question or "").strip()
        if not q:
            return ""
        # 1. 优先书名号《》内容
        m = re.search(r"《([^》]+)》", q)
        if m:
            return m.group(1).strip()
        # 2. 英文双引号 / 中文引号内容
        m = re.search(r'[“"\']([^”"\']+)[”"\']', q)
        if m:
            return m.group(1).strip()
        # 3. 去掉疑问词后的原文（保留主体实体）
        cleaned = re.sub(r"(是什么|什么是|什么意思|怎么样|如何|介绍|定义|含义|简介|概述|为什么|是谁|哪[个些]|请问|帮我|搜一下|查一下)", "", q)
        cleaned = re.sub(r"[？?。！!，,：:]+$", "", cleaned).strip()
        return cleaned or q

    def _deterministic_web_search(self, question: str, history: list = None) -> str:
        """
        确定性联网搜索：grade 判定不相关时触发一次。

        搜索前先结合历史提炼简洁关键词，再调用 web_search；若提炼词搜不到，
        用原始专有名词（书名号内容/去疑问词原文）兜底再搜一次。失败返回空串。
        """
        from mcp_service.tool_bridge import call_tool_by_name
        self._emit_status("正在提炼搜索关键词...")
        query = self._refine_search_query(question, history)
        self._emit_status(f"正在联网搜索: {query}")

        def _do_search(q: str) -> str:
            try:
                r = call_tool_by_name("web_search", {"query": q, "num": 5})
                return str(r) if r else ""
            except Exception:
                return ""

        result = _do_search(query)

        # 兜底重搜：提炼关键词无结果时，用原始专有名词再搜一次
        if not result or result == "[]":
            fallback = self._extract_fallback_query(question)
            if fallback and fallback != query:
                self._emit_status(f"搜索无结果，改用原文重搜: {fallback}")
                result = _do_search(fallback)

        return result

    def _generate_stream(self, messages: list) -> str:
        """普通流式/非流式生成（含 DSML 剥离兜底）。"""
        generation = ""
        try:
            if hasattr(self.llm, "stream"):
                r = self._stream_collect(self.llm.stream(messages))
                generation = r["generation"]
                # 若生成内容为空（可能纯 DSML 被剥离），用剥离后的 full 兜底
                if not generation.strip():
                    generation = _strip_dsml(r["full"]).strip() or r["full"]
            else:
                from llm import invoke_llm
                generation = invoke_llm(self.llm, messages)
                generation = _strip_dsml(generation) if generation else generation
                if self._stream_cb is not None:
                    self._stream_cb(generation, False)
        except Exception as e:
            generation = f"[生成失败: {e}]"
            if self._stream_cb is not None:
                self._stream_cb(generation, False)
        return generation

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run(self, question: str, history: List[dict] = None, mode: str = "rag") -> GraphState:
        """
        执行完整链路。

        Args:
            question: 用户问题
            history:  对话历史 [{"role","content"}, ...]
            mode:     "direct"（直接对话）或 "rag"（知识库问答），默认 "rag"

        Returns:
            最终 GraphState（含 generation 字段为最终回答）
        """
        initial_state: GraphState = {
            "mode":              mode,
            "question":          question,
            "original_question": question,
            "history":           history or [],
            "documents":         [],
            "reranked":          [],
            "filtered":          [],
            "relevant":          False,
            "retrieval_status":  "empty",
            "top_score":         0.0,
            "evidence_count":    0,
            "kb_retried":        False,
            "web_results":       [],
            "generation":        "",
            "retry_count":       0,
            "max_retries":       self.max_retries,
        }
        return self._graph.invoke(initial_state)


# ============================================================================
# 工厂函数：从 server 现有组件构建 RAGGraph
# ============================================================================

def build_rag_graph(reranker, llm, retriever=None, embedder=None, store=None, **kwargs) -> RAGGraph:
    """便捷工厂：用现有组件构建 RAGGraph 实例。"""
    return RAGGraph(reranker, llm, retriever=retriever, embedder=embedder, store=store, **kwargs)
