# -*- coding: utf-8 -*-
"""
评测集生成：从 doc_tree.db 抽取文档内容，用 LLM 生成 20 题 QA 对。

  20 题 = README.md 5 题 + rag.pdf 5 题 + 2404.18231v2.pdf 5 题 + 跨文档 5 题

输出：data/dataset.json
  [{"id": "q01", "question": ..., "ground_truth": ..., "doc_id": ..., "type": ...}, ...]

运行（主 poetry 环境）：
  poetry run python eval/generate_dataset.py
"""

import json
import os
import sqlite3
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.dirname(EVAL_DIR))

import config


DB_PATH = os.path.join(config.PROJECT_ROOT, "data", "doc_tree.db")


# ============================================================================
# 文档内容抽取
# ============================================================================
def extract_doc(doc_id: str, max_paragraphs: int = 30, para_chars: int = 500):
    """抽取文档的章节大纲 + 段落样本，返回结构化的内容文本。"""
    c = sqlite3.connect(DB_PATH)
    cur = c.cursor()

    sections = cur.execute(
        "SELECT title, level, summary FROM nodes "
        "WHERE doc_id=? AND type='section' ORDER BY level, ord",
        (doc_id,),
    ).fetchall()

    paragraphs = cur.execute(
        "SELECT text, summary FROM nodes "
        "WHERE doc_id=? AND type='paragraph' AND text IS NOT NULL AND text != '' "
        "ORDER BY ord LIMIT ?",
        (doc_id, max_paragraphs),
    ).fetchall()

    c.close()

    lines = []
    lines.append("【章节大纲】")
    for title, level, summary in sections:
        indent = "  " * (level if level else 0)
        summary = (summary or "").strip()
        title = (title or "").strip()
        seg = f"{indent}- {title}" if title else f"{indent}- (未命名章节)"
        if summary:
            seg += f"：{summary}"
        lines.append(seg)

    lines.append("\n【段落样本】")
    for i, (text, summary) in enumerate(paragraphs, 1):
        t = (text or "").strip().replace("\n", " ")
        s = (summary or "").strip()
        seg = f"[{i}] {s}" if s else f"[{i}] {t[:para_chars]}"
        if len(seg) > para_chars + 120:
            seg = seg[: para_chars + 120]
        lines.append(seg)

    return "\n".join(lines)


def load_document_titles():
    """返回 {doc_id: title}。"""
    c = sqlite3.connect(DB_PATH)
    cur = c.cursor()
    rows = cur.execute("SELECT doc_id, title FROM documents").fetchall()
    c.close()
    return {doc_id: title for doc_id, title in rows}


# ============================================================================
# LLM 调用
# ============================================================================
_SYSTEM = (
    "你是 RAG（检索增强生成）系统评测集构建专家。"
    "你的任务是：基于给定文档内容，生成用于评测 RAG 系统效果的问答对。"
    "输出必须是合法的 JSON，不要输出任何 JSON 之外的文字。"
)

_SINGLE_PROMPT = """下面是一份文档的内容（章节大纲 + 段落样本）。

要求：基于该文档内容，生成 {n} 个问答对。规则如下：
1. 每个问题的答案必须能在给定文档内容中找到依据（不要问文档没提到的内容）。
2. 问题类型要多样化，尽量覆盖：
   - 单点事实：具体概念、术语、数字、名称的定义或说明
   - 概念解释：某个方法/技术的原理、作用、流程
   - 分类/列举：文档中提到的多个方法、类别、阶段及其区别
3. 问题用中文提问（即使文档是英文）。
4. ground_truth 是准确、完整、信息充分的标准答案，包含关键信息点，
   不要只给一句话结论。ground_truth 用中文写。
5. 问题之间不要重复、不要高度雷同。

输出 JSON 数组，每个元素格式：
{{"question": "问题", "ground_truth": "标准答案", "type": "类型"}}

文档内容：
{content}
"""

_CROSS_PROMPT = """下面是三份文档的标题与主题概览。

要求：生成 {n} 个「需要综合多份文档才能完整回答」的问答对（跨文档问题）。
规则：
1. 每个问题的答案需要跨越至少两份文档的内容才能完整作答（例如对比不同文档对同一主题的论述、
   关联不同文档里的概念、或综合多文档信息回答一个整体性问题）。
2. ground_truth 是准确、完整、信息充分的标准答案，包含关键信息点，用中文写。
3. 问题用中文提问。

输出 JSON 数组，每个元素格式：
{{"question": "问题", "ground_truth": "标准答案", "type": "类型"}}

文档概览：
{overview}
"""


def call_llm(prompt: str) -> str:
    from llm_factory import get_model
    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_model("llm", answer=True)
    if llm is None:
        raise RuntimeError("无可用 LLM 模型，无法生成评测集")
    return invoke_llm(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])


def parse_qa(text: str):
    """健壮地解析 LLM 输出的 JSON 数组（兼容代码围栏 / 前后缀文字）。"""
    from common.text_utils import strip_code_fence
    text = strip_code_fence(text or "").strip()

    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        # 尝试截取第一个 [ 到最后一个 ] 之间的内容
        s = text.find("[")
        e = text.rfind("]")
        if s >= 0 and e > s:
            try:
                obj = json.loads(text[s:e + 1])
            except Exception:
                return None
        else:
            return None

    if isinstance(obj, dict):
        # 某些模型会包一层 {"questions": [...]}
        for k in ("questions", "qa", "items", "data"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return None
    return obj if isinstance(obj, list) else None


# ============================================================================
# 主流程
# ============================================================================
def main():
    config.ensure_dirs()
    titles = load_document_titles()
    docs = list(titles.items())  # [(doc_id, title), ...]

    if len(docs) < 3:
        raise RuntimeError(f"知识库文档数不足（当前 {len(docs)}），无法生成评测集。请先入库文档。")

    dataset = []
    qid = 0

    # --- 单文档：每篇 5 题 ---
    for doc_id, title in docs:
        print(f"\n[生成] {title} ...")
        content = extract_doc(doc_id)
        prompt = _SINGLE_PROMPT.format(n=5, content=content)
        text = call_llm(prompt)
        items = parse_qa(text)
        if not items:
            print(f"  [警告] {title} 解析失败，原始输出前 200 字：{text[:200]}")
            continue
        for it in items[:5]:
            q = (it.get("question") or "").strip()
            gt = (it.get("ground_truth") or "").strip()
            if not q or not gt:
                continue
            qid += 1
            dataset.append({
                "id": f"q{qid:02d}",
                "question": q,
                "ground_truth": gt,
                "doc_id": doc_id,
                "doc_title": title,
                "type": it.get("type", ""),
                "cross_doc": False,
            })
        print(f"  → 已生成 {len([d for d in dataset if d['doc_id'] == doc_id])} 题")

    # --- 跨文档：5 题 ---
    print("\n[生成] 跨文档 5 题 ...")
    overview = "\n".join(
        f"- 文档《{title}》：{extract_overview(doc_id)}" for doc_id, title in docs
    )
    prompt = _CROSS_PROMPT.format(n=5, overview=overview)
    text = call_llm(prompt)
    items = parse_qa(text)
    if items:
        for it in items[:5]:
            q = (it.get("question") or "").strip()
            gt = (it.get("ground_truth") or "").strip()
            if not q or not gt:
                continue
            qid += 1
            dataset.append({
                "id": f"q{qid:02d}",
                "question": q,
                "ground_truth": gt,
                "doc_id": "",
                "doc_title": "跨文档",
                "type": it.get("type", ""),
                "cross_doc": True,
            })

    with open(config.DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 共 {len(dataset)} 题，已写入 {config.DATASET_PATH}")
    for d in dataset:
        print(f"  {d['id']} [{d['doc_title']}] {d['question'][:50]}...")


def extract_overview(doc_id: str) -> str:
    """文档一句话概览（用于跨文档 prompt）。"""
    c = sqlite3.connect(DB_PATH)
    cur = c.cursor()
    row = cur.execute("SELECT abstract FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    c.close()
    if not row or not row[0]:
        return "（无概览）"
    try:
        obj = json.loads(row[0])
        return obj.get("abstract") or obj.get("title") or "（无概览）"
    except Exception:
        return str(row[0])[:200]


if __name__ == "__main__":
    main()
