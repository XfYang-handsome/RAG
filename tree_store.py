# -*- coding: utf-8 -*-
"""结构树持久化（SQLite）—— 文档树存储与查询。

架构定位（与定稿一致）：
  - SQLite 存 Structure Tree（文档 → 章节 → 段落/表格/图片）
  - Milvus 只存 Retrieval Chunk（向量检索）
  - JSON 只作为中间产物 / 导出格式

提供能力：
  1. 保存 / 加载文档树
  2. 章节路径恢复（命中 chunk 的 parent_node_id → 沿树向上还原章节标题路径）
  3. 结构匹配（query 关键词 → section title 匹配）
  4. 文档管理（列出 / 删除文档树）

说明：SQLite 为 Python 内置模块，零额外依赖。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional

from structure_resolver import TreeNode

# 数据库文件位置：项目根目录 data/doc_tree.db
_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "doc_tree.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    source      TEXT,
    title       TEXT,
    abstract    TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id        TEXT PRIMARY KEY,
    doc_id         TEXT,
    type           TEXT,
    title          TEXT,
    text           TEXT,
    summary        TEXT,
    level          INTEGER,
    page           INTEGER,
    parent_node_id TEXT,
    ord            INTEGER,
    section_path   TEXT,
    source_type    TEXT,
    source_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_doc    ON nodes(doc_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(type);
"""


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    # WAL + busy_timeout：多进程（多 Celery worker）并发写树库时避免「database is
    # locked」。WAL 让读写不互斥；busy_timeout 让写锁竞争时等待而非立即报错。
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection):
    """轻量迁移：为旧库补充新增列（summary / abstract），避免「表已存在」导致列缺失。"""
    node_cols = [r["name"] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()]
    if "summary" not in node_cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN summary TEXT DEFAULT ''")
        conn.commit()

    doc_cols = [r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    if "abstract" not in doc_cols:
        conn.execute("ALTER TABLE documents ADD COLUMN abstract TEXT DEFAULT ''")
        conn.commit()


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------

def save_tree(root: TreeNode, source: str = "", abstract: str = "") -> str:
    """保存整棵文档树到 SQLite，返回 doc_id。

    Args:
        root:     文档树根节点
        source:   来源文件名（用于展示 / 删除）
        abstract: 文档级主旨摘要（LLM 生成，供跨文档路由；可空）

    幂等：同一 doc_id 重复保存会先删除旧数据再插入。
    """
    doc_id = root.doc_id
    conn = _get_conn()
    try:
        with conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM nodes WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "INSERT INTO documents (doc_id, source, title, abstract) VALUES (?, ?, ?, ?)",
                (doc_id, source, root.title or "", abstract or ""),
            )
            _insert_node_recursive(conn, root)
    finally:
        conn.close()
    return doc_id


def _insert_node_recursive(conn: sqlite3.Connection, node: TreeNode):
    conn.execute(
        """INSERT INTO nodes
           (node_id, doc_id, type, title, text, summary, level, page,
            parent_node_id, ord, section_path, source_type, source_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            node.node_id,
            node.doc_id,
            node.type,
            node.title,
            node.text,
            node.summary,
            node.level,
            node.page,
            node.parent_node_id,
            node.order,
            json.dumps(node.section_path),
            node.source_type,
            node.source_id,
        ),
    )
    for child in node.children:
        _insert_node_recursive(conn, child)


def load_tree(doc_id: str) -> Optional[TreeNode]:
    """从 SQLite 加载文档树，返回根节点（未找到返回 None）。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? ORDER BY ord, rowid", (doc_id,)
        ).fetchall()
        if not rows:
            return None
        nodes = {r["node_id"]: _row_to_node(r) for r in rows}
        root = None
        for n in nodes.values():
            if n.parent_node_id and n.parent_node_id in nodes:
                nodes[n.parent_node_id].children.append(n)
            else:
                root = n
        # 根节点 type 应为 document
        if root is None:
            # 兜底：找 type=document
            for n in nodes.values():
                if n.type == "document":
                    root = n
                    break
        return root
    finally:
        conn.close()


def _row_to_node(row) -> TreeNode:
    return TreeNode(
        node_id=row["node_id"],
        type=row["type"],
        doc_id=row["doc_id"],
        title=row["title"] or "",
        text=row["text"] or "",
        summary=row["summary"] or "",
        level=row["level"],
        page=row["page"] or 0,
        parent_node_id=row["parent_node_id"],
        order=row["ord"] or 0,
        section_path=json.loads(row["section_path"]) if row["section_path"] else [],
        source_type=row["source_type"] or "",
        source_id=row["source_id"] or "",
    )


# ---------------------------------------------------------------------------
# 章节路径恢复（上下文补全核心）
# ---------------------------------------------------------------------------

def get_section_path_titles(doc_id: str, node_id: str) -> List[str]:
    """沿树向上恢复 node_id 的章节标题路径（不含 document 根）。

    例：命中 chunk 的 parent_node_id = "doc_xx:line:3"（段落节点）→
        返回 ["第一章 系统设计", "1.1 架构"]。
    """
    conn = _get_conn()
    try:
        path = []
        current = node_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (current,)
            ).fetchone()
            if row is None:
                break
            if row["type"] == "section":
                path.append(row["title"])
            current = row["parent_node_id"]
        path.reverse()
        return path
    finally:
        conn.close()


def get_section_summary(doc_id: str, node_id: str) -> str:
    """沿树向上返回最近的 section 节点的摘要（检索铺垫用）。

    命中 chunk 的 parent_node_id 可能是叶子节点（段落），需向上找到
    所属 section 的 summary；找不到返回空字符串。
    """
    conn = _get_conn()
    try:
        current = node_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            row = conn.execute(
                "SELECT type, summary, parent_node_id FROM nodes WHERE node_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            if row["type"] == "section" and row["summary"]:
                return row["summary"]
            current = row["parent_node_id"]
        return ""
    finally:
        conn.close()


def get_section_meta_batch(doc_id: str, node_ids: List[str]) -> Dict[str, dict]:
    """批量恢复多个节点的章节标题路径 + 所属 section 摘要（消除 N+1）。

    一次拉取该文档全部节点建索引，再对每个 node_id 沿 parent 链向上爬，
    结果与逐条调用 get_section_path_titles + get_section_summary 完全一致。

    Returns:
        {node_id: {"path_titles": [...], "section_summary": str}}
    """
    result: Dict[str, dict] = {
        nid: {"path_titles": [], "section_summary": ""} for nid in node_ids
    }
    if not node_ids:
        return result

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, type, title, summary, parent_node_id FROM nodes WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    parent_of = {r["node_id"]: r["parent_node_id"] for r in rows}
    type_of = {r["node_id"]: r["type"] for r in rows}
    title_of = {r["node_id"]: r["title"] for r in rows}
    summary_of = {r["node_id"]: r["summary"] for r in rows}

    for nid in node_ids:
        path: List[str] = []
        summary = ""
        current = nid
        seen = set()
        while current and current not in seen:
            seen.add(current)
            if current not in type_of:
                break
            if type_of[current] == "section":
                path.append(title_of[current])
                if not summary and summary_of[current]:
                    summary = summary_of[current]
            current = parent_of.get(current)
        path.reverse()
        result[nid] = {"path_titles": path, "section_summary": summary}
    return result


def get_section_path_by_node_ids(node_ids: List[str]) -> List[str]:
    """批量恢复多个 chunk 的章节路径（取最具体的那个，去重）。

    多个 chunk 可能属于同一 section，这里返回去重后的 section 列表，
    避免上下文重复膨胀。
    """
    # 通过 doc_id 前缀推断（node_id 形如 doc_xxx:...）
    all_paths = []
    seen_docs = set()
    for nid in node_ids:
        doc_id = _doc_id_from_node_id(nid)
        if doc_id and doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        all_paths.extend(get_section_path_titles(doc_id, nid))
    # 去重且保序
    seen = set()
    result = []
    for t in all_paths:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _doc_id_from_node_id(node_id: str) -> str:
    """从 node_id 提取 doc_id（node_id 形如 doc_xxx:source_id）。"""
    if not node_id:
        return ""
    return node_id.split(":", 1)[0]


# ---------------------------------------------------------------------------
# 结构匹配（query → section）
# ---------------------------------------------------------------------------

def match_sections(query: str, limit: int = 5) -> List[dict]:
    """结构匹配：query 关键词命中 section 标题。

    采用确定性关键词匹配（非 embedding）：
      - 对 query 与 section title 做分词（中英文字符 + 数字 + 字母）
      - 按关键词命中数打分，命中越多越靠前

    Returns:
        [{"doc_id", "node_id", "title", "level", "section_path", "score"}, ...]
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE type = 'section'"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored = []
    for r in rows:
        title = r["title"] or ""
        t_tokens = _tokenize(title)
        if not t_tokens:
            continue
        # 2-gram 交集数（中文词重叠度）
        hits = len(q_tokens & t_tokens)
        score = hits * 2
        if score > 0:
            scored.append({
                "doc_id": r["doc_id"],
                "node_id": r["node_id"],
                "title": title,
                "level": r["level"],
                "section_path": json.loads(r["section_path"]) if r["section_path"] else [],
                "score": score,
            })

    scored.sort(key=lambda x: (-x["score"], x["level"]))
    return scored[:limit]


def _tokenize(text: str) -> set:
    """分词：中文用字符 2-gram（单字保留），英文/数字用完整单词 + 词形归一化。

    这样「系统架构设计」与「系统设计」能通过 2-gram 重叠（系统/设计）命中；
    英文「personalized」与「individualized」能通过同义词归一化命中。
    """
    import re
    tokens = set()
    # 中文 2-gram（连续片段）
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.add(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.add(seg[i:i + 2])
    # 英文/数字单词（词形归一化：小写 + 同义词 + 词干化）
    for m in re.findall(r"[a-zA-Z0-9]+", text):
        tokens.add(_normalize_token(m))
    return tokens


# ---------------------------------------------------------------------------
# 英文词形归一化（修复 match_sections 词形不匹配导致的漏召回）
# ---------------------------------------------------------------------------
# 同义词归一：把近义/变体词统一映射到代表词，解决「personalized ↔ individualized」
# 「scenario ↔ application」这类词形不同但语义相同导致的精确匹配漏召回。
_SYNONYM_MAP = {
    # personalized / individualized 家族 → personalized
    "individualized": "personalized",
    "individualised": "personalized",
    "individualization": "personalized",
    "personalization": "personalized",
    "personalisation": "personalized",
    "personalised": "personalized",
    "customized": "personalized",
    "customised": "personalized",
    "customization": "personalized",
    "tailored": "personalized",
    "tailoring": "personalized",
    # application / scenario 家族 → application
    "scenario": "application",
    "scenarios": "application",
    "usecase": "application",
    "usecases": "application",
    "usage": "application",
}


def _stem_en(word: str) -> str:
    """轻量英文词干化：只处理复数（最安全、最常用），不做 ed/ing 剥离（易破坏）。"""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"          # personalities -> personality
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]                # personas -> persona, applications -> application
    return w


def _normalize_token(word: str) -> str:
    """英文 token 词形归一化：小写 → 同义词映射 → 词干化。"""
    w = (word or "").lower()
    return _SYNONYM_MAP.get(w, _stem_en(w))


# ---------------------------------------------------------------------------
# 树导航查询（纯树检索 T1：结构原语，全走 SQLite，不碰 Milvus）
# ---------------------------------------------------------------------------

def get_top_ancestor(node_id: str) -> Optional[str]:
    """向上遍历 parent，返回顶层 section 祖先（root/document 的直接子节点）的 node_id。

    纯树导航「优先起点」用：match_sections 命中的可能是叶子级 section（无 section
    子节点、无 descend 空间），取它的顶层祖先作为起点，让 LLM 从顶层逐层 descend
    到命中处，途中也能探索兄弟分支（而非直接 push 命中节点导致 LLM 决策无空间）。

    若 node_id 本身就是顶层 section（parent 是 document），返回它自己。
    节点不存在或 parent 链异常时返回 None。
    """
    conn = _get_conn()
    try:
        cur = node_id
        while cur:
            row = conn.execute(
                "SELECT node_id, parent_node_id, type FROM nodes WHERE node_id = ?",
                (cur,),
            ).fetchone()
            if row is None:
                return None
            parent = row["parent_node_id"]
            # 判断父节点类型：parent 是 document → cur 即顶层 section
            if parent:
                prow = conn.execute(
                    "SELECT type FROM nodes WHERE node_id = ?", (parent,)
                ).fetchone()
                if prow is not None and prow["type"] == "document":
                    return cur
                cur = parent
                continue
            # 无 parent：cur 就是根（document 或孤儿 section）
            return cur if row["type"] == "section" else None
        return None
    finally:
        conn.close()


def get_document_root_id(doc_id: str) -> Optional[str]:
    """返回文档根节点的 node_id（纯树导航初始化的起点定位）。

    根节点特征：type='document'；兜底用 parent_node_id 为空/不存在的节点。
    注意：根节点的 node_id 通常 ≠ doc_id（doc_id 是文档哈希，node_id 是
    树内节点标识），所以不能直接用 doc_id 当作根节点 id 去查 children。
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT node_id FROM nodes WHERE doc_id = ? AND type = 'document' "
            "ORDER BY rowid LIMIT 1",
            (doc_id,),
        ).fetchone()
        if row:
            return row["node_id"]
        # 兜底：parent_node_id 为空（或空串）的节点即根
        row = conn.execute(
            "SELECT node_id FROM nodes WHERE doc_id = ? AND "
            "(parent_node_id IS NULL OR parent_node_id = '') "
            "ORDER BY rowid LIMIT 1",
            (doc_id,),
        ).fetchone()
        return row["node_id"] if row else None
    finally:
        conn.close()


def get_children(node_id: str) -> List[dict]:
    """返回某节点的直接子节点（纯树导航的 expand 原语）。

    按 ord 排序保证阅读顺序（ord 相同时按 rowid 稳定排序）。返回：
    [{"node_id", "doc_id", "type", "title", "summary", "level", "parent_node_id"}, ...]

    仅返回「结构信息」，不含原文 text（expand 只看路牌，不读正文）。
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, doc_id, type, title, summary, level, parent_node_id "
            "FROM nodes WHERE parent_node_id = ? ORDER BY ord, rowid",
            (node_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_subtree_stats(node_id: str) -> dict:
    """返回某节点子树的体量统计，供 LLM 决策理解章节层级深度。

    动机：LLM 决策（descend/backtrack）若只看到 title+summary+score，看不到
    「这个章节下面还有几层子章节、有多少叶子」，只能盲选。本函数补上体量，
    让 LLM 判断 descend 深度（是否值得深入、往下还有没有内容）。

    Returns:
        {"section_count": int, "leaf_count": int, "max_depth": int}
        section_count: 子树内 section 节点总数（不含自身）
        leaf_count:    子树内叶子节点总数（paragraph/table/figure 等非 section）
        max_depth:     子树最大 section 层级深度（自身=0，直接子 section=1，...）

    节点不存在时返回全 0。
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT doc_id FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return {"section_count": 0, "leaf_count": 0, "max_depth": 0}
        doc_id = row["doc_id"]
        rows = conn.execute(
            "SELECT node_id, type, parent_node_id FROM nodes WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    children_map: Dict[str, List] = {}
    for r in rows:
        children_map.setdefault(r["parent_node_id"] or "", []).append(r)

    section_count = 0
    leaf_count = 0
    max_depth = 0

    def walk(nid: str, depth: int) -> None:
        nonlocal section_count, leaf_count, max_depth
        if depth > max_depth:
            max_depth = depth
        for c in children_map.get(nid, []):
            if c["type"] == "section":
                section_count += 1
                walk(c["node_id"], depth + 1)
            else:
                leaf_count += 1

    walk(node_id, 0)
    return {"section_count": section_count, "leaf_count": leaf_count, "max_depth": max_depth}


def get_subtree_stats_all(doc_id: str) -> Dict[str, dict]:
    """一次性返回文档内所有 section 节点的子树体量统计。

    动机：`get_subtree_stats` 每次调用都「全量拉取该文档所有节点再 Python 统计」，
    而纯树导航的 LLM 决策会逐轮对每个候选子章节调用它，重复拉取 + 重复遍历。
    本函数一次拉全文档，用 memo 化后序遍历 O(n) 算完所有 section 体量，供
    树导航初始化时缓存一次、后续 O(1) 查询。

    Returns:
        {node_id: {"section_count": int, "leaf_count": int, "max_depth": int}}
        只包含 section 节点；空文档返回空 dict。
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, type, parent_node_id FROM nodes WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    children_map: Dict[str, List] = {}
    for r in rows:
        children_map.setdefault(r["parent_node_id"] or "", []).append(r)

    memo: Dict[str, dict] = {}

    def dfs(nid: str) -> dict:
        if nid in memo:
            return memo[nid]
        section_count = 0
        leaf_count = 0
        max_depth = 0
        for c in children_map.get(nid, []):
            if c["type"] == "section":
                section_count += 1
                sub = dfs(c["node_id"])
                section_count += sub["section_count"]
                leaf_count += sub["leaf_count"]
                if sub["max_depth"] + 1 > max_depth:
                    max_depth = sub["max_depth"] + 1
            else:
                leaf_count += 1
        memo[nid] = {"section_count": section_count, "leaf_count": leaf_count, "max_depth": max_depth}
        return memo[nid]

    for r in rows:
        if r["type"] == "section":
            dfs(r["node_id"])

    return memo


def get_representative_texts_all(doc_id: str, max_chars: int = 200) -> Dict[str, str]:
    """一次性返回文档内所有 section 节点的「代表性叶子文本」。

    动机：纯树导航的 Node Rerank 对每个候选 section 子节点都要递归
    `get_children` + `get_node` 找「第一个叶子文本」——N 个候选 = N 次递归
    N+1 查询。本函数一次拉全文档节点（含 text），O(n) 建树 + memo 化
    后序遍历算出每个 section 的第一个叶子文本（前 max_chars），供树导航
    按 doc 缓存一次、后续 O(1) 查询。

    Returns:
        {node_id: 代表性叶子文本}，只含 section 节点；空文档返回空 dict。
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, type, text, parent_node_id FROM nodes WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    children_map: Dict[str, List] = {}
    for r in rows:
        children_map.setdefault(r["parent_node_id"] or "", []).append(r)

    memo: Dict[str, str] = {}

    def first_leaf_text(nid: str) -> str:
        if nid in memo:
            return memo[nid]
        result = ""
        # 1) 先找直接叶子子节点里的第一个非空文本
        for c in children_map.get(nid, []):
            if c["type"] != "section":
                t = (c["text"] or "").strip()
                if t:
                    result = t[:max_chars]
                    break
        # 2) 无直接叶子文本 → 递归子 section 找第一个叶子
        if not result:
            for c in children_map.get(nid, []):
                if c["type"] == "section":
                    t = first_leaf_text(c["node_id"])
                    if t:
                        result = t
                        break
        memo[nid] = result
        return result

    out: Dict[str, str] = {}
    for r in rows:
        if r["type"] == "section":
            out[r["node_id"]] = first_leaf_text(r["node_id"])
    return out


def get_node(node_id: str) -> Optional[dict]:
    """返回单节点完整详情（含原文 text），纯树导航的 read 原语。

    返回 None 表示节点不存在。section_path 反序列化为 List[int]。
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["section_path"] = json.loads(d["section_path"]) if d.get("section_path") else []
        return d
    finally:
        conn.close()


def list_top_sections(doc_id: str = None) -> List[dict]:
    """返回顶层 section 列表（document 的直接 section 子节点）。

    供 LLM 语义入口选择（match_sections 关键词无命中时的兜底）：LLM 基于
    「顶层 section 标题 + 摘要」判断问题该从哪个章节进入，比关键词硬匹配更可靠。

    Args:
        doc_id: 指定文档则只返回该文档顶层 section；None=所有文档。

    Returns:
        [{"node_id", "doc_id", "title", "summary"}, ...]
    """
    if doc_id:
        doc_ids = [doc_id]
    else:
        doc_ids = [d["doc_id"] for d in list_documents()]

    result: List[dict] = []
    for did in doc_ids:
        root_id = get_document_root_id(did)
        if not root_id:
            continue
        for c in get_children(root_id):
            if c.get("type") == "section":
                result.append({
                    "node_id": c.get("node_id", ""),
                    "doc_id": did,
                    "title": c.get("title", ""),
                    "summary": c.get("summary", ""),
                })
    return result


def get_doc_toc(doc_id: str) -> str:
    """返回单文档的目录树文本（缩进标题 + 摘要），喂 LLM 的「局部地图」。

    按 section_path 排序，用 level 决定缩进与 # 号层级（level 相对最小
    值归一化，兼容 level 起始值不为 0 的解析器）。摘要为空时省略。
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, title, summary, level, section_path "
            "FROM nodes WHERE doc_id = ? AND type = 'section'",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return ""

    sections = []
    for r in rows:
        sections.append({
            "node_id": r["node_id"],
            "title": r["title"] or "",
            "summary": r["summary"] or "",
            "level": r["level"] if r["level"] is not None else 0,
            "section_path": json.loads(r["section_path"]) if r["section_path"] else [],
        })
    sections.sort(key=lambda x: x["section_path"])

    # level 归一化：以最小 level 为基准（兼容 level 起始值不为 0 的解析器）
    min_level = min(s["level"] for s in sections)
    parts = []
    for s in sections:
        depth = max(0, s["level"] - min_level)
        indent = "  " * depth
        mark = "#" * min(depth + 1, 6)  # 最多 6 级标题
        line = f"{indent}{mark} {s['title']}"
        if s["summary"]:
            line += f" — {s['summary']}"
        parts.append(line)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 文档管理
# ---------------------------------------------------------------------------

def list_documents() -> List[dict]:
    """列出所有已保存的文档树（含 abstract 文档主旨，供跨文档路由）。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT d.doc_id, d.source, d.title, d.abstract, d.created_at, "
            "COUNT(n.node_id) AS node_count "
            "FROM documents d LEFT JOIN nodes n ON d.doc_id = n.doc_id "
            "GROUP BY d.doc_id ORDER BY d.created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """删除指定文档树。"""
    conn = _get_conn()
    try:
        with conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM nodes WHERE doc_id = ?", (doc_id,))
        return True
    finally:
        conn.close()


def get_document(doc_id: str) -> Optional[dict]:
    """获取单个文档元信息（含 abstract）。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_document_abstract(doc_id: str, abstract: str) -> bool:
    """更新文档的 abstract（供已入库文档补生成主旨摘要）。"""
    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                "UPDATE documents SET abstract = ? WHERE doc_id = ?",
                (abstract or "", doc_id),
            )
        return True
    finally:
        conn.close()


def rename_document_source(old_source: str, new_source: str) -> bool:
    """重命名文档的 source（只改文件名，不动文档内容/结构/摘要）。

    Returns:
        是否有文档被更新。
    """
    conn = _get_conn()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE documents SET source = ? WHERE source = ?",
                (new_source, old_source),
            )
        return cur.rowcount > 0
    finally:
        conn.close()


def list_sections(doc_id: str) -> List[dict]:
    """返回文档的章节树（目录型问题用）。

    按 section_path 排序，保证层级与阅读顺序正确。
    Returns:
        [{"node_id", "title", "level", "section_path", "page"}, ...]
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT node_id, title, level, section_path, page "
            "FROM nodes WHERE doc_id = ? AND type = 'section'",
            (doc_id,),
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "node_id": r["node_id"],
                "title": r["title"],
                "level": r["level"],
                "section_path": json.loads(r["section_path"]) if r["section_path"] else [],
                "page": r["page"],
            })
        # 按 section_path（List[int]）真实顺序排序，保证目录层级与阅读顺序正确
        result.sort(key=lambda x: x["section_path"])
        return result
    finally:
        conn.close()


def list_document_structure() -> List[dict]:
    """列出所有文档的章节结构（跨文档目录，目录型问题用）。"""
    conn = _get_conn()
    try:
        docs = conn.execute("SELECT doc_id, title, source FROM documents").fetchall()
        result = []
        for d in docs:
            sections = list_sections(d["doc_id"])
            result.append({
                "doc_id": d["doc_id"],
                "doc_title": d["title"],
                "source": d["source"],
                "sections": sections,
            })
        return result
    finally:
        conn.close()


def export_tree(doc_id: str) -> Optional[dict]:
    """导出文档树为 dict（JSON 中间产物/导出格式）。"""
    root = load_tree(doc_id)
    return root.to_dict() if root else None
