# -*- coding: utf-8 -*-
"""
================================================================================
Agentic RAG — Planner 需求拆解（Phase 9）
================================================================================

把 COMPLEX 问题拆解为多个信息需求（Requirement），遵循「最小充分拆解」：
  能一条检索解决的，就不要拆成多个。

输出 requirements（带 importance），驱动后续 Requirement–Evidence–Gap 循环。

用法：
    from agentic_rag.planner import plan
    reqs = plan(question)   # [Requirement(id="R1", description=..., importance=1.0), ...]
================================================================================
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .state import Requirement, RequirementStatus
from .settings import get as _get

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 模型获取（tool_llm 优先，回退 llm）
# ---------------------------------------------------------------------------
def _get_planner_llm():
    from llm_factory import get_model
    return get_model("tool_llm", "llm")


# ---------------------------------------------------------------------------
# 结构化输出
# ---------------------------------------------------------------------------
class _ReqItem(BaseModel):
    id: str
    description: str = ""
    importance: float = 1.0
    synthetic: bool = False
    synthesize_from: List[str] = Field(default_factory=list)

    @field_validator("importance", mode="before")
    @classmethod
    def _norm_importance(cls, v):
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 1.0

    @field_validator("synthesize_from", mode="before")
    @classmethod
    def _norm_deps(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [x.strip() for x in v.split(",") if x.strip()] if v else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


class _PlanResult(BaseModel):
    requirements: List[_ReqItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 对比型需求拆分（方案 A：跨章节对比 → 拆成单点需求，根治「永远 PARTIAL」）
# ---------------------------------------------------------------------------
# 对比型需求（如「A、B、C 的应用场景的区别」）的问题：它要求三类内容**同时**
# 出现在证据里才能判 SUPPORTED，而三类内容通常分散在不同章节，导致永远 PARTIAL。
# 解法：把「A、B、C 的 X 的区别」拆成「A 的 X」「B 的 X」「C 的 X」三个单点需求，
# 每个都能被单点检索满足；对比归纳的职责交给最终 Synthesis。

_COMPARE_PAT = re.compile(r"(区别|差异|不同|对比|比较|异同|优缺点|差别|哪个更好|分别|各自)")
_ENTITY_SPLIT_PAT = re.compile(r"[、,，和及与以及]")
_TRAIL_CONN = re.compile(r"[、,，和及与以及]+$")


def _split_compare_req(desc: str) -> List[str]:
    """把「A、B、C 的 X 的区别」拆成「A 的 X」「B 的 X」「C 的 X」单点需求。

    仅当能可靠提取「实体列表 + 共同主题」时才拆分，否则原样返回（不做破坏性拆解）。
    这是 LLM 拆解后的**代码兜底**：LLM 若仍输出笼统对比需求，这里强制拆开。
    """
    d = (desc or "").strip()
    if not d or not _COMPARE_PAT.search(d):
        return [d]

    # 去掉结尾对比词（含可选前缀「的/有/有什么/有何」）
    core = re.sub(r"(的|有|有什么|有何)?\s*(区别|差异|不同|差别|对比|异同|优缺点)$", "", d)

    # 提取「实体列表 的 主题」（非贪婪匹配第一个「的」）
    m = re.match(r"^(.+?)\s*的\s*(.+)$", core)
    if m:
        entity_part, topic = m.group(1).strip(), m.group(2).strip()
    else:
        entity_part, topic = core.strip(), ""

    # 清洗主题：去掉残留对比词 + 结尾连接词
    topic = _COMPARE_PAT.sub("", topic)
    topic = _TRAIL_CONN.sub("", topic).strip()

    entities = [e.strip() for e in _ENTITY_SPLIT_PAT.split(entity_part) if e.strip()]
    if len(entities) < 2:
        return [d]  # 无法可靠提取多个实体 → 保持原样

    if not topic:
        return entities  # 无独立主题 → 直接用实体本身作为单点需求
    return [f"{e} 的 {topic}" for e in entities]


def _split_compare_requirements(reqs: List[Requirement]) -> List[Requirement]:
    """把 requirements 中的对比型需求拆成单点需求，并重新编号 R1..Rn。

    重建 Requirement 时保留 synthetic/synthesize_from 字段（关系型需求不被
    _split_compare_req 拆分，会原样保留，但其合成属性不能丢，否则关系类
    需求会重新沦为独立检索目标导致空转）。synthesize_from 里的旧 id 会在
    _mark_synthetic_requirements 兜底阶段被校验/重算，故此处仅原样携带。
    """
    out: List[Requirement] = []
    for r in reqs:
        for s in _split_compare_req(r.description):
            out.append(Requirement(
                id="",
                description=s,
                importance=r.importance,
                synthetic=r.synthetic,
                synthesize_from=list(r.synthesize_from),
            ))
    for i, r in enumerate(out, 1):
        r.id = f"R{i}"
    return out


# ---------------------------------------------------------------------------
# 枚举-展开（「有哪些 X + 它们的区别」类问题的两阶段拆解）
# ---------------------------------------------------------------------------
# 这类问题的难点：对比对象在问题里【未列出】，LLM 若强行拆对比就会臆造分类
# （如「工具型/陪伴型/专家型」），导致检索无法命中。正确做法是两阶段：
#   阶段1：先检索「列举需求」，拿到真实对象清单；
#   阶段2：从证据抽取真实清单，动态生成逐项对比子需求，再检索。
# 这里把「泛化对比需求」标记为 deferred，由 Agent 循环在依赖需求 SUPPORTED 后展开。

_LIST_SIGNAL = re.compile(r"有哪些|有什么|哪些|何种|列举|多少种|哪几种|几种|分类|类型|种类|清单|列表")
_COMPARE_SIGNAL = re.compile(r"区别|差异|不同|对比|比较|异同|优缺点|分别|各自")


def _is_enum_compare_question(question: str) -> bool:
    """判断问题是否为「枚举 + 对比」模式（有哪些 X + 它们的区别）。"""
    q = question or ""
    return bool(_LIST_SIGNAL.search(q)) and bool(_COMPARE_SIGNAL.search(q))


def _mark_deferred_requirements(reqs: List[Requirement], question: str) -> List[Requirement]:
    """代码兜底：把「枚举-对比」问题里的泛化对比需求标记为 deferred。

    识别基于「requirements 本身」而非仅看原始 question（更健壮：用户措辞多变，
    如「有什么人格」而非「有哪些类型」，仅靠 question 正则易漏判）。只要 reqs 里
    同时存在「列举需求」和「含对比词的泛化对比需求」，就触发 deferred 展开。

    已明确列出对象的对比需求（如「A、B、C 的区别」）会被 _split_compare_req 提前
    拆成单点，其 description 已不含对比词，不会误入 compare_reqs，天然安全。

    识别：
      - 列举需求：description 含列举信号、不含对比词（作为枚举源 depends_on）
      - 泛化对比需求：description 含对比词（对象未列出，故仍是笼统的）
    把泛化对比需求标记 deferred=True, depends_on=列举需求.id。
    """
    list_req: Optional[Requirement] = None
    compare_reqs: List[Requirement] = []
    for r in reqs:
        if _COMPARE_SIGNAL.search(r.description):
            compare_reqs.append(r)
        elif _LIST_SIGNAL.search(r.description) and list_req is None:
            list_req = r

    if list_req is None or not compare_reqs:
        return reqs

    for r in compare_reqs:
        r.deferred = True
        r.depends_on = list_req.id
    return reqs


# ---------------------------------------------------------------------------
# 关系-综合型需求识别（方案 A：关系/联系类需求 → synthetic，不做独立检索）
# ---------------------------------------------------------------------------
# 关系型需求（如「A 与 B 的关系/联系/如何结合」）的问题：它要求两类内容**同时**
# 出现在证据里才能判 SUPPORTED，但知识库通常只有 A、B 各自独立的章节，没有
# 直接论述二者关系的段落，直接检索该需求必然 MISSING 空转（Controller 反复
# SEARCH/READ/WEB_SEARCH 都无果）。
# 解法：标记 synthetic=True + synthesize_from=[依赖需求]。它不进入 gap 循环，
# 待依赖需求都 SUPPORTED 后由最终 Synthesis 归纳二者关系，与「对比型拆单点」
# 的思路一致：归纳职责交给 Synthesis，检索职责交给单点需求。

_RELATION_SIGNAL = re.compile(
    r"关系|联系|关联|协同|结合|配合|融合|相互作用|相互影响"
)

# 关系型需求必须同时出现「实体连接/指代」词：避免误伤「关联规则」「数据融合」
# 「多模态融合」等单实体技术名词（它们含关系信号词，但本身是可检索的单点概念，
# 不应被标记为综合归纳需求）。
_RELATION_LINK = re.compile(r"与|和|及|以及|之间|二者|两者|它们|彼此")


def _mark_synthetic_requirements(reqs: List[Requirement]) -> List[Requirement]:
    """代码兜底：把「关系/联系/关联」类综合需求标记为 synthetic。

    这是 LLM 拆解后的**代码兜底**：LLM 若仍输出笼统关系需求（或未正确标记
    synthetic），这里强制识别并标记，防止关系类需求沦为独立检索目标导致空转。

    依赖（synthesize_from）的确定：
      - 优先采用 LLM 已给出的且仍有效的依赖 id；
      - 否则兜底为「所有其他非综合型需求」（关系归纳通常需要其它需求各自到位）。

    边界：若关系需求是唯一需求（无任何其它非综合需求可归纳），则取消 synthetic，
    退化为正常检索（此时没有归纳基础，至少尝试检索一次，而非直接零检索）。
    """
    synthetic_ids: set = set()
    for r in reqs:
        # 需求描述同时含关系信号 + 实体连接/指代 → 综合型（关系归纳本身没有独立文档论述）
        if _RELATION_SIGNAL.search(r.description) and _RELATION_LINK.search(r.description):
            synthetic_ids.add(r.id)

    if not synthetic_ids:
        return reqs

    valid_ids = {r.id for r in reqs}
    non_synthetic_ids = [r.id for r in reqs if r.id not in synthetic_ids]
    for r in reqs:
        if r.id not in synthetic_ids:
            continue
        # LLM 给出的依赖若仍有效则保留，否则兜底为所有其它非综合需求
        deps = [d for d in r.synthesize_from if d in valid_ids and d != r.id]
        if not deps:
            deps = non_synthetic_ids
        if deps:
            r.synthetic = True
            r.synthesize_from = deps
        else:
            # 无归纳基础（关系需求是唯一需求）→ 取消 synthetic，正常检索
            r.synthetic = False
            r.synthesize_from = []
    return reqs


def _extract_sub_requirements(state, dep_id: str, question: str, llm) -> List[str]:
    """从枚举需求（dep_id）的已验证证据中，抽取真实对象清单并生成对比子需求描述。

    Returns:
        List[str]：每个是「单个真实对象 + 对比维度」的可检索信息点描述，
        如 "工具型角色agent的人设特点"。这些将作为新 Requirement.description。
    """
    item = state.requirement_status.get(dep_id)
    if item is None or not item.evidence_ids:
        return []

    texts: List[str] = []
    for eid in item.evidence_ids[:8]:  # 最多 8 条证据，控制 prompt 体积
        ev = state.evidences.get(eid)
        if ev is not None and ev.text:
            texts.append(ev.text.strip())
    if not texts:
        return []

    evidence = "\n---\n".join(t[:400] for t in texts)[:4000]

    prompt = (
        f"原始问题：{question}\n\n"
        f"以下是已确认包含「问题所问的列举对象」的证据：\n{evidence}\n\n"
        f"任务：从证据中提取「列举对象」的真实清单（严格以证据为准，不要臆造证据中没有的对象），"
        f"并为每个对象生成一个可检索的对比信息点描述（结合问题所问的对比维度，如特点/应用/优劣）。\n"
        f"只输出描述列表，一行一个，不要编号、不要解释、不要引号。"
    )
    from llm import invoke_llm
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是证据抽取器。只输出干净的信息点列表，一行一个。"),
            HumanMessage(content=prompt),
        ]).strip()
    except Exception:
        return []

    out: List[str] = []
    seen = set()
    for ln in text.splitlines():
        ln = ln.strip().lstrip("0123456789.、-•*· ").strip()
        if ln and len(ln) <= 80 and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def expand_deferred(state, llm=None) -> List[Requirement]:
    """展开 deferred 需求：当 depends_on 已 SUPPORTED 时，从证据抽取真实清单，
    动态生成逐项对比子需求替换 deferred 需求。

    Args:
        state: 当前 AgentState（就地修改 requirements / requirement_status / gaps）
        llm:   抽取模型（None=内部取 tool_llm）

    Returns:
        新生成的子需求列表（空=本轮未发生展开）。
    """
    if llm is None:
        llm = _get_planner_llm()
    if llm is None:
        return []

    new_reqs: List[Requirement] = []
    removed: List[str] = []
    expanded: List[Requirement] = []

    for r in state.requirements:
        if not r.deferred or r.id in state.expanded_requirements:
            new_reqs.append(r)
            continue

        # 依赖需求是否已解决？
        dep_item = state.requirement_status.get(r.depends_on)
        if dep_item is None or dep_item.status != RequirementStatus.SUPPORTED:
            new_reqs.append(r)  # 依赖未满足 → 保留 deferred，等下一轮
            continue

        # 从证据抽取真实清单
        subs = _extract_sub_requirements(state, r.depends_on, state.question, llm)
        if len(subs) < 2:
            # 抽不到 ≥2 个对象 → 保留 deferred（由 no-progress 停止兜底，避免臆造）
            new_reqs.append(r)
            continue

        # 展开：替换 deferred 需求为逐项对比子需求
        removed.append(r.id)
        for i, desc in enumerate(subs, 1):
            sub = Requirement(id=f"{r.id}.{i}", description=desc, importance=r.importance)
            new_reqs.append(sub)
            expanded.append(sub)

    if removed:
        state.requirements = new_reqs
        for rid in removed:
            state.requirement_status.pop(rid, None)
            state.expanded_requirements.append(rid)
        state.derive_gaps()

    return expanded


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_PLAN_SYSTEM = (
    "你是问题拆解器。把复杂问题拆解成若干信息需求，遵循最小充分拆解原则"
    "（能一次检索解决的不要拆）。只输出 JSON，不要解释。"
    "绝不臆造问题中未出现的具体实体、分类、人名或术语作为检索目标。"
)


def _build_prompt(question: str) -> str:
    mn = _get("planner.min_requirements", 1)
    mx = _get("planner.max_requirements", 5)
    return (
        f"复杂问题：{question}\n\n"
        "请拆解为信息需求，输出 JSON：\n"
        '{"requirements": [{"id": "R1", "description": "OpenAI 的技术路线", "importance": 1.0}]}\n\n'
        "规则：\n"
        "1. id 从 R1 开始递增；description 简洁明确，是可检索的具体信息点。\n"
        "2. importance 在 0~1 之间：核心需求 1.0，次要/补充 0.8，边缘 0.5。\n"
        f"3. 最小充分拆解：{mn}~{mx} 个即可，不要过度拆解。\n"
        "4. 问题若包含多个不同疑问（如「是什么」+「为什么」），每个疑问至少拆成 1 个独立需求，"
        "不要合并成一个笼统需求。\n"
        "5. 对比型需求分两种情况处理：\n"
        "   a. 问题【已明确列出】被对比对象（如「A、B、C 的区别/差异/不同」）→ 必须拆成每个对象的独立单点需求，不要输出笼统对比需求。例如：\n"
        "      「人口统计人设、角色人设、个体化人设的应用场景有什么不同」应拆成：\n"
        '      {"id":"R1","description":"人口统计人设的应用场景","importance":1.0}\n'
        '      {"id":"R2","description":"角色人设的应用场景","importance":1.0}\n'
        '      {"id":"R3","description":"个体化人设的应用场景","importance":1.0}\n'
        "      这样每个需求都能被单点检索满足，对比归纳留给最终答案生成阶段。\n"
        "   b. 问题用「有哪些 X」+「它们的区别」这种形式，被对比对象【未列出】→ 禁止臆造具体对象！\n"
        "      此时拆成【两条】需求：一条纯列举、一条泛化对比（对比需求用泛指词，不写具体类型名）。例如：\n"
        "      「角色型agent有哪些人设？它们的区别是什么？」应拆成：\n"
        '      {"id":"R1","description":"角色型agent的人设类型","importance":1.0}\n'
        '      {"id":"R2","description":"角色型agent各人设类型的区别","importance":1.0}\n'
        "      系统会先检索 R1 得到真实类型清单，再自动把 R2 展开成逐项对比，所以 R2 千万不要写死具体类型名。\n"
        "6. 关系/联系/关联类综合需求（如「A 与 B 的关系/联系/如何结合」）不是独立检索点：\n"
        "   知识库通常没有直接论述二者关系的独立段落，直接检索该需求必然空转。正确拆法：\n"
        "   a. 先把 A、B 各自拆成独立单点需求（如 R1=A 的定义、R2=B 的定义/流程）；\n"
        "   b. 再输出一个综合需求 R3，描述二者的关系，并设 synthetic=true、\n"
        "      synthesize_from=[\"R1\",\"R2\"]（依赖前面的单点需求）。\n"
        "   系统不会单独检索 R3，而是等 R1、R2 都有证据后，由答案生成阶段自动归纳二者关系。\n"
        "   例如「角色型agent与RAG的关系」应拆成：\n"
        '      {"id":"R1","description":"角色型agent的定义与特点","importance":1.0}\n'
        '      {"id":"R2","description":"RAG 的定义与工作流程","importance":1.0}\n'
        '      {"id":"R3","description":"角色型agent与RAG的关系","importance":1.0,"synthetic":true,"synthesize_from":["R1","R2"]}\n'
        "7. 绝不臆造问题中未出现的具体实体、分类或术语作为检索目标。\n"
        "   当问题只要求「列举/有哪些 X」而没给出具体清单时，就用「列举 X 类型」本身作为需求，\n"
        "   不要自行脑补「工具型/陪伴型/专家型」这类具体分类。"
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def plan(question: str, llm=None) -> List[Requirement]:
    """拆解复杂问题为 requirements。

    Args:
        question: 复杂问题
        llm:      拆解模型（None=内部取 tool_llm）

    Returns:
        List[Requirement]。LLM 失败/无模型时返回单条兜底 requirement。
    """
    if llm is None:
        llm = _get_planner_llm()
    if llm is None:
        return [Requirement(id="R1", description=question, importance=1.0)]

    from llm import invoke_llm
    from common.text_utils import parse_json

    try:
        text = invoke_llm(llm, [
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(content=_build_prompt(question)),
        ])
        obj = parse_json(text)
        if obj is None:
            raise ValueError("no json")
        result = _PlanResult.model_validate(obj)

        reqs = [
            Requirement(
                id=item.id,
                description=item.description or item.id,
                importance=item.importance,
                synthetic=item.synthetic,
                synthesize_from=list(item.synthesize_from),
            )
            for item in result.requirements
        ]
        if reqs:
            # 方案 A 兜底：把 LLM 仍输出的对比型需求拆成单点需求
            reqs = _split_compare_requirements(reqs)
            # 枚举-展开：把「有哪些 X + 区别」的泛化对比需求标记为 deferred
            reqs = _mark_deferred_requirements(reqs, question)
            # 关系-综合型：把「A 与 B 的关系/联系」标记为 synthetic（不做独立检索）
            return _mark_synthetic_requirements(reqs)
    except Exception:
        pass

    # 兜底：拆解失败 → 单条 requirement（整个问题作为一个需求）
    return [Requirement(id="R1", description=question, importance=1.0)]
