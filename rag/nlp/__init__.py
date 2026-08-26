"""RAGFlow rag.nlp 的轻量替代实现。

deepdoc 只用到一小部分能力：分词（tokenize/tag/is_chinese）、
编码检测（find_codec）以及段落合并（MergeStrategy/merge_paragraphs）。
这里重新实现以让拷贝过来的 deepdoc 包能够脱离完整 RAGFlow 工程独立运行。
"""

import re
from enum import Enum

# ---------------------------------------------------------------------------
# CJK 字符判定与分词
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)


class _RagTokenizer:
    """极小化的 rag_tokenizer 替代实现。

    deepdoc 仅调用以下方法：
      - tokenize(text) -> str（空格分隔的 token）
      - is_chinese(ch) -> bool
      - tag(word)     -> str（词性；"nr" 表示人名）
    """

    def tokenize(self, text: str) -> str:
        if not text:
            return ""
        # CJK 逐字切分，英文/数字保持整词
        s = _CJK_RE.sub(lambda m: " " + m.group(0) + " ", text)
        tokens = [t for t in re.split(r"\s+", s) if t]
        return " ".join(tokens)

    def is_chinese(self, ch: str) -> bool:
        return bool(ch) and _CJK_RE.fullmatch(ch) is not None

    def tag(self, word: str) -> str:
        # 简化词性标注：始终返回 "x"，从不返回 "nr"，
        # 这样 deepdoc 中基于人名的启发式逻辑不会误触发（安全且无害）。
        return "x"


rag_tokenizer = _RagTokenizer()


# ---------------------------------------------------------------------------
# 编码检测
# ---------------------------------------------------------------------------
_CODEC_ALIASES = {
    "ascii": "utf-8",
    "utf-8": "utf-8",
    "utf8": "utf-8",
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "gb18030": "gb18030",
    "big5": "big5",
    "iso-8859-1": "latin-1",
    "latin-1": "latin-1",
    "windows-1252": "latin-1",
}


def find_codec(binary) -> str:
    if isinstance(binary, str):
        return "utf-8"
    try:
        import chardet
        enc = (chardet.detect(binary).get("encoding") or "utf-8").lower()
    except Exception:
        enc = "utf-8"
    return _CODEC_ALIASES.get(enc, enc or "utf-8")


# ---------------------------------------------------------------------------
# 段落合并（txt parser 使用）
# ---------------------------------------------------------------------------
class MergeStrategy(str, Enum):
    STD = "std"
    OVER_CAP = "over_cap"
    MAX_CAP = "max_cap"
    CONTEXT_AWARE = "context_aware"
    TABLE = "table"


def merge_paragraphs(paragraphs, chunk_token_num, strategy=MergeStrategy.STD):
    """把段落按 token 数合并成若干组，每组不超过 chunk_token_num。"""
    groups = []
    current = []
    current_tokens = 0
    for p in paragraphs:
        if not p:
            continue
        t = len(rag_tokenizer.tokenize(p).split())
        if t <= 0:
            continue
        if current and current_tokens + t > chunk_token_num:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(p)
        current_tokens += t
    if current:
        groups.append(current)
    return groups
