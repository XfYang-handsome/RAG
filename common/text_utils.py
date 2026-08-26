# -*- coding: utf-8 -*-
"""
================================================================================
通用文本 / JSON 工具（tree_retrieval 与 agentic_rag 共用）
================================================================================

抽出几处「等价」的重复实现，统一维护：
  - extract_json / parse_json : 从 LLM 输出中健壮提取 JSON（平衡括号扫描，
    能正确处理嵌套对象、字符串值内的花括号、转义引号、代码围栏前缀等），
    替代脆弱的 ``text.find("{") + text.rfind("}")``。
  - contains_cjk / HAS_CJK    : 判断文本是否含中文字符。
  - has_english_entity        : 判断文本是否含英文专名/技术名词。
  - translate_to_en_keywords  : 中文 query 英化为英文检索关键词（跨语言检索）。

注意：本模块顶层不 import 任何项目内重依赖（llm / llm_factory / config_loader），
保持 common 可被 deepdoc 等场景独立 import；需要时在函数内惰性 import。
================================================================================
"""

from __future__ import annotations

import json
import re
from typing import Optional

# ---------------------------------------------------------------------------
# 中文字符检测
# ---------------------------------------------------------------------------
HAS_CJK = re.compile(r"[\u4e00-\u9fff]")


def contains_cjk(text: str) -> bool:
    """判断文本是否含中文字符。"""
    return bool(HAS_CJK.search(text or ""))


# ---------------------------------------------------------------------------
# 英文专名 / 技术名词检测
# ---------------------------------------------------------------------------
_EN_ENTITY_RE = re.compile(r"[A-Z][A-Za-z0-9\-\.]{1,}")


def has_english_entity(text: str) -> bool:
    """检测文本是否含英文专名/技术名词（大写字母开头、长度≥2 的 token）。"""
    for tok in _EN_ENTITY_RE.findall(text or ""):
        if len(tok) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# 健壮 JSON 提取（平衡括号扫描）
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[str]:
    """从文本中提取首个完整 JSON 对象/数组子串，失败返回 None。

    与 ``find("{") + rfind("}")`` 的区别：用平衡括号 + 字符串状态机扫描，
    能正确处理嵌套对象、字符串值内的花括号、转义引号、代码围栏前缀等，
    不会因为解释性前缀/后缀或值内花括号而截断错误。
    """
    if not text:
        return None

    # 定位第一个 { 或 [
    start = -1
    open_ch = close_ch = ""
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text: str) -> Optional[dict]:
    """提取并解析 JSON 对象；提取失败或结果非对象返回 None。"""
    sub = extract_json(text)
    if sub is None:
        return None
    try:
        obj = json.loads(sub)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def strip_code_fence(text: str) -> str:
    """剥离 LLM 输出首尾的 Markdown 代码围栏（如 ```json ... ``` 或 ``` ... ```）。

    仅处理首尾的围栏标记，不误伤正文中的反引号；无围栏时原样返回。

    比「``text.strip("`")``」更精确：后者会把正文首尾的反引号也一并去掉，
    且无法区分 `` ```json `` 的语言标识。
    """
    text = (text or "").strip()
    # 首部围栏：``` 及可选的语言标识（json/python/javascript/yaml/... 到换行为止）
    if text.startswith("```"):
        text = text[3:]
        text = re.sub(r"^[A-Za-z0-9_+#.-]*[ \t]*\r?\n?", "", text, count=1)
        text = text.lstrip()
    # 尾部围栏：```
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


# ---------------------------------------------------------------------------
# 中文 query 英化（跨语言检索）
# ---------------------------------------------------------------------------
def translate_to_en_keywords(query: str, llm=None) -> str:
    """把中文 query 翻译成英文检索关键词（用于检索英文知识库）。

    仅当 query 含中文字符才翻译；翻译失败返回原 query（不阻塞检索）。
    """
    if not contains_cjk(query):
        return query

    if llm is None:
        from llm_factory import get_model
        llm = get_model("rewrite", "tool_llm", "llm")
    if llm is None:
        return query

    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    from config_loader import cfg as _cfg

    mn = _cfg("agentic.search.translate_keywords_min", 3)
    mx = _cfg("agentic.search.translate_keywords_max", 6)
    prompt = (
        f"把下面的中文检索词翻译成英文检索关键词，"
        f"只输出 {mn}~{mx} 个英文关键词（空格分隔，小写），不要句子、不要解释：\n\n{query}"
    )
    try:
        text = invoke_llm(llm, [
            SystemMessage(content="你是检索关键词翻译助手，知识库为英文文档。"),
            HumanMessage(content=prompt),
        ]).strip()
        return text if text else query
    except Exception:
        return query
