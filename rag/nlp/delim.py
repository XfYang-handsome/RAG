"""RAGFlow rag.nlp.delim 的轻量替代实现。"""

import re

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def parse_delimiter_field(delimiter):
    """把分隔符字符串解析为分隔符列表。

    支持普通字符、反斜杠转义（\\n \\t \\r）以及反引号包裹的多字符 token。
    """
    if not delimiter:
        return []
    parts = []
    i = 0
    n = len(delimiter)
    while i < n:
        ch = delimiter[i]
        if ch == "\\" and i + 1 < n and delimiter[i + 1] in _ESCAPES:
            parts.append(_ESCAPES[delimiter[i + 1]])
            i += 2
            continue
        if ch == "`":
            j = delimiter.find("`", i + 1)
            if j != -1:
                parts.append(delimiter[i + 1:j])
                i = j + 1
                continue
        parts.append(ch)
        i += 1
    return parts


def compile_delimiter_pattern(dels):
    """编译分隔符列表为可用于 re.split / re.match 的正则片段。"""
    return "|".join(re.escape(d) for d in dels)


def normalize_text_newlines(txt):
    return txt.replace("\r\n", "\n").replace("\r", "\n")
