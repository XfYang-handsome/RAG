"""
================================================================================
deepdoc 增强解析封装
================================================================================
将 RAGFlow 的 deepdoc 解析器接入主程序的文档上传流程。

deepdoc 是从 RAGFlow 拷贝过来的解析库，本身依赖 RAGFlow 的 common / rag 模块。
本项目在根目录提供了这两个模块的轻量实现（见 common/ 与 rag/），使 deepdoc
可以在脱离完整 RAGFlow 工程的情况下独立运行。

各解析器返回结构：
  - DocxParser(fnm)                  -> (secs, tbls)，secs 为 [(text, style), ...]
  - ExcelParser(binary)              -> [row_str, ...]
  - PptParser(fnm, from, to)         -> [slide_str, ...]
  - EpubParser(fnm)                  -> [section_str, ...]
  - HtmlParser(fnm)                  -> [section_str, ...]
  - MarkdownElementExtractor(text)   -> extract_elements() -> [section_str, ...]
  - JsonParser(binary)               -> [json_chunk_str, ...]
  - TxtParser(fnm, binary)           -> [[chunk, ""], ...]
  - PdfParser(fnm)                   -> 需 OCR/版面模型，失败时回退 pypdf
================================================================================
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# deepdoc 增强解析支持的扩展名
DEEPOC_EXTENSIONS = {
    ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".pdf",
    ".epub", ".html", ".htm", ".md", ".json", ".txt",
}


def _flatten_strings(items):
    """把 parser 返回的嵌套字符串结构统一摊平为一维字符串列表。"""
    out = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, (list, tuple)):
            out.extend(_flatten_strings(it))
        else:
            out.append(str(it))
    return out


# ---------------------------------------------------------------------------
# 各格式解析
# ---------------------------------------------------------------------------
def _extract_docx(path):
    from deepdoc.parser import DocxParser
    secs, tbls = DocxParser()(path)
    parts = []
    for text, style in secs:
        text = (text or "").strip()
        if not text:
            continue
        # 标题样式加 "#" 前缀，尽量保留文档层级
        if style and "heading" in (style or "").lower():
            parts.append("# " + text)
        else:
            parts.append(text)
    for tbl in tbls:
        parts.extend(_flatten_strings(tbl))
    return "\n\n".join(p for p in parts if p.strip())


def _extract_excel(path, binary=None):
    from deepdoc.parser import ExcelParser
    if binary is not None:
        data = binary
    else:
        with open(path, "rb") as f:
            data = f.read()
    rows = ExcelParser()(data)
    return "\n".join(r for r in rows if r and str(r).strip())


def _extract_pptx(path):
    from deepdoc.parser import PptParser
    slides = PptParser()(path, 0, 100000)
    return "\n\n".join(s for s in slides if s and s.strip())


def _extract_html(path):
    from deepdoc.parser import HtmlParser
    return "\n\n".join(HtmlParser()(path))


def _extract_epub(path):
    from deepdoc.parser import EpubParser
    return "\n\n".join(EpubParser()(path))


def _extract_markdown(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    from deepdoc.parser.markdown_parser import MarkdownElementExtractor
    sections = MarkdownElementExtractor(text).extract_elements()
    return "\n\n".join(s for s in sections if s and str(s).strip())


def _extract_json(path):
    with open(path, "rb") as f:
        binary = f.read()
    from deepdoc.parser import JsonParser
    chunks = JsonParser()(binary)
    return "\n".join(c for c in chunks if c and c.strip())


def _extract_txt(path):
    with open(path, "rb") as f:
        binary = f.read()
    from deepdoc.parser import TxtParser
    chunks = TxtParser()(path, binary=binary)
    return "\n\n".join(c[0] for c in chunks if c and c[0])


def _extract_pdf_text(path):
    """纯文本 PDF 提取（pypdf），作为完整 OCR 解析的回退。"""
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


_POS_TAG_RE = re.compile(r"@@[0-9\-]+\t[0-9.\-]+\t[0-9.\-]+\t[0-9.\-]+\t[0-9.\-]+##")


def _clean_position_tags(text):
    """去掉 deepdoc 输出的版面位置标签（@@页码\tx0\tx1\ttop\tbottom##）。"""
    text = _POS_TAG_RE.sub("", text)
    # 清理残留的多余空行
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_pdf(path):
    """优先使用 deepdoc 完整 PDF 解析，失败时回退 pypdf 文本提取。"""
    try:
        from deepdoc.parser import PdfParser
        if PdfParser is None:
            raise ImportError("PdfParser unavailable")
        parser = PdfParser()
        texts, tbls = parser(path, need_image=False)
        if isinstance(texts, str):
            text = texts
        else:
            text = "\n\n".join(str(t) for t in texts)
        text = _clean_position_tags(text)

        # 合并表格识别的结构化文本（"表头：值; ..." 格式）
        table_parts = []
        for tbl in tbls or []:
            rows = tbl[1] if isinstance(tbl, (tuple, list)) and len(tbl) == 2 else tbl
            if isinstance(rows, (list, tuple)):
                table_parts.extend(str(r) for r in rows if r)
            elif rows is not None:
                table_parts.append(str(rows))
        if table_parts:
            text = (text + "\n\n" if text else "") + "\n".join(table_parts)
        return text
    except Exception as e:
        logger.warning("deepdoc 完整 PDF 解析不可用(%s)，回退 pypdf 文本提取", e)
        return _extract_pdf_text(path)


_HANDLERS = {
    ".docx": _extract_docx,
    ".xlsx": _extract_excel,
    ".xls": _extract_excel,
    ".csv": _extract_excel,
    ".pptx": _extract_pptx,
    ".epub": _extract_epub,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".md": _extract_markdown,
    ".json": _extract_json,
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
}


def extract_deepdoc_text(filepath):
    """按扩展名调用对应的 deepdoc 解析器，返回纯文本。"""
    ext = os.path.splitext(filepath)[1].lower()
    handler = _HANDLERS.get(ext)
    if handler is None:
        # 未覆盖的文本/代码格式：直接按文本读取
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return handler(filepath)
