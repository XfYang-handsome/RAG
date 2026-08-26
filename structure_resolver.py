# -*- coding: utf-8 -*-
"""结构归位（Structure Resolver）—— 步骤 1。

把 deepdoc 从文档中提取的「扁平元素」还原成文档树（Document Tree）：

    Document
    └── Section (标题)
        ├── Paragraph
        ├── Table
        └── Figure

支持的格式与标题层级来源：

| 格式   | 标题层级来源                          |
|--------|--------------------------------------|
| .pdf   | 书签大纲（outlines）优先，title 元素 + 编号/字号兜底 |
| .docx  | Word 样式（Heading 1/2/3 或 标题 1/2/3） |
| .md    | Markdown 标题（#/##/###）             |
| .html  | h1~h6 标签                            |
| .txt   | 无层级（退化为扁平 Document → Paragraph） |

说明：本模块只负责「结构归位」，不涉及 chunk 切分 / embedding（那是后续步骤）。
"""

from __future__ import annotations

import hashlib
import logging
import os

from common.text_utils import parse_json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 标题编号正则 -> 层级（0-based：0=章，1=节，2=小节）
_HEADING_PATTERNS: List[tuple] = [
    (re.compile(r"^第[零一二三四五六七八九十百千]+章"), 0),
    (re.compile(r"^第[零一二三四五六七八九十百千]+节"), 1),
    (re.compile(r"^\d+\.\d+\.\d+"), 2),
    (re.compile(r"^\d+\.\d+"), 1),
    (re.compile(r"^\d+[\.、)）]"), 0),
    (re.compile(r"^[A-Z]\.\d+"), 1),
    (re.compile(r"^[A-Z]\.\s"), 0),
]

# 这些 layout_type 不进入树（页眉页脚 / 空类型）
_IGNORED_LAYOUTS = {"header", "footer", ""}

# 结构归位支持的文档扩展名（其余格式在增强解析时回退到普通父子块入库）
STRUCTURED_EXTENSIONS = {
    ".pdf", ".docx", ".md", ".markdown", ".html", ".htm", ".txt", ".text",
}

# deepdoc layout_type -> 树节点 type
_LAYOUT_TO_TYPE = {
    "text": "paragraph",
    "reference": "paragraph",
    "figure caption": "paragraph",
    "table caption": "paragraph",
    "table": "table",
    "figure": "figure",
    "equation": "figure",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """文档树节点。"""
    node_id: str
    type: str                    # document / section / paragraph / table / figure
    doc_id: str = ""             # 所属文档稳定 ID（内容哈希）
    title: str = ""              # 仅 section 有
    text: str = ""               # paragraph / table / figure 内容
    summary: str = ""            # 章节摘要（LLM 生成，检索铺垫用；空=未生成）
    level: int = -1              # section 层级（-1=根，0=章，1=节…）
    page: int = 0
    bbox: Optional[List[float]] = None   # [x0, top, x1, bottom]（PDF 有，其余 None）
    parent_node_id: Optional[str] = None
    order: int = 0               # 同层顺序
    section_path: List[int] = field(default_factory=list)  # 章节路径，如 [1, 2]
    source_type: str = ""        # deepdoc 原始 layout_type / 格式来源
    source_id: str = ""          # 溯源键：page:bbox 或 行号
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        # bbox 可能含 numpy.float32（deepdoc 返回），必须转原生 float，
        # 否则 Phase 3 步骤化落盘时 json.dump 会报「float32 is not JSON serializable」。
        bbox = self.bbox
        if bbox is not None:
            bbox = [float(v) for v in bbox]
        return {
            "node_id": self.node_id,
            "type": self.type,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "summary": self.summary,
            "level": self.level,
            "page": self.page,
            "bbox": bbox,
            "parent_node_id": self.parent_node_id,
            "order": self.order,
            "section_path": self.section_path,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        """从 to_dict() 的输出反序列化回 TreeNode（递归重建 children）。

        用于 Phase 3 步骤级重跑：结构树 parse 产物落盘后，chunk/index 步骤
        需从 JSON 读回完整树，与内存对象语义完全一致（往返无损）。
        """
        children = [cls.from_dict(c) for c in (d.get("children") or [])]
        return cls(
            node_id=d.get("node_id", ""),
            type=d.get("type", "document"),
            doc_id=d.get("doc_id", ""),
            title=d.get("title", ""),
            text=d.get("text", ""),
            summary=d.get("summary", ""),
            level=d.get("level", -1),
            page=d.get("page", 0),
            bbox=d.get("bbox"),
            parent_node_id=d.get("parent_node_id"),
            order=d.get("order", 0),
            section_path=list(d.get("section_path") or []),
            source_type=d.get("source_type", ""),
            source_id=d.get("source_id", ""),
            children=children,
        )


# ---------------------------------------------------------------------------
# 标题工具函数
# ---------------------------------------------------------------------------

def _strip_number_prefix(text: str) -> str:
    """剥离标题开头的编号前缀（如 ``7.5`` / ``A.1`` / ``第1章``）。"""
    t = text.strip()
    for pat, _level in _HEADING_PATTERNS:
        m = pat.match(t)
        if m:
            return t[m.end():].strip()
    return t


def _normalize_title(text: str) -> str:
    """标题归一化（去编号 + 去空白 + 小写），用于去重比对。"""
    return re.sub(r"\s+", "", _strip_number_prefix(text)).lower()


def _heading_level(text: str) -> Optional[int]:
    """用编号正则推断标题层级，匹配不到返回 None。"""
    t = text.strip()
    for pat, level in _HEADING_PATTERNS:
        if pat.match(t):
            return level
    return None


def _title_matches(a: str, b: str) -> bool:
    """两个标题是否指向同一个（归一化后相等或互为子串，容忍 OCR 粘连）。"""
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if shorter in longer and len(shorter) / max(len(longer), 1) >= 0.6:
        return True
    return False


def _dedup_headings(headings: List[dict]) -> List[dict]:
    """标题去重：优先保留层级来源更可靠的（outline > style > markdown）。"""
    seen = []
    for h in headings:
        if any(_title_matches(h["title"], s["title"]) for s in seen):
            continue
        seen.append(h)
    return seen


# ---------------------------------------------------------------------------
# Structure Resolver
# ---------------------------------------------------------------------------

class StructureResolver:
    """把 deepdoc 扁平元素归位成文档树。"""

    def __init__(self):
        self._seq = 0
        self._used_ids: set = set()

    # -- 入口 ---------------------------------------------------------------
    def resolve(self, filepath: str, on_progress=None) -> TreeNode:
        ext = os.path.splitext(filepath)[1].lower()
        doc_id = compute_doc_id(filepath)
        doc_title = ""
        if ext == ".pdf":
            headings, content, doc_title = self._parse_pdf(filepath, on_progress=on_progress)
        elif ext == ".docx":
            headings, content = self._parse_docx(filepath)
        elif ext in (".md", ".markdown"):
            headings, content = self._parse_markdown(filepath)
        elif ext in (".html", ".htm"):
            headings, content = self._parse_html(filepath)
        elif ext in (".txt", ".text"):
            headings, content = self._parse_txt(filepath)
        else:
            raise NotImplementedError(f"暂不支持该格式的结构归位: {ext}")

        root = self._build_tree(headings, content, doc_id, doc_title)

        # 结构残缺（无标题 / 大量内容挂在 root）→ LLM 从正文重建目录树。
        # 失败静默回退扁平树（不阻塞入库）。
        if self._should_reconstruct(root, content):
            new_headings, new_content = self._reconstruct_headings(content, doc_id)
            if new_headings:
                logger.info(
                    "结构残缺，LLM 重建目录树成功：%d 个章节标题（原文块 %d → %d）",
                    len(new_headings), len(content), len(new_content),
                )
                # 重建前清空已用 id：第一次建树的结果被丢弃，其 node_id 不应
                # 占用唯一性空间，否则重建后的 root 等节点会被追加 #1 后缀。
                self._used_ids.clear()
                self._seq = 0
                root = self._build_tree(new_headings, new_content, doc_id, doc_title)
            else:
                logger.info("结构残缺，LLM 重建失败，保留扁平树")

        return root

    # -- 残缺检测 + LLM 重建 -------------------------------------------------
    def _should_reconstruct(self, root: TreeNode, content: List[dict]) -> bool:
        """判断文档树是否「结构残缺」，需要 LLM 重建目录树。

        残缺判据（纯代码，不靠 LLM）：
          1. section 数为 0（如 .txt、无大纲且无 bbox 标题的 PDF）→ 残缺
          2. 挂载在 root 下的内容占比过高（大量正文没归到任何 section）→ 残缺
        """
        from config_loader import cfg
        if not cfg("structure.reconstruct.enabled", True):
            return False
        if not content:
            return False

        stats = count_nodes(root)
        if stats["section"] == 0:
            return True

        # root 直属内容（非 section）占全部内容节点的比例
        root_leaves = sum(1 for c in root.children if c.type != "section")
        total_leaves = stats["paragraph"] + stats["table"] + stats["figure"]
        ratio_threshold = float(cfg("structure.reconstruct.root_content_ratio", 0.6))
        if total_leaves > 0 and root_leaves / total_leaves > ratio_threshold:
            return True
        return False

    def _reconstruct_headings(self, content: List[dict], doc_id: str
                              ) -> Tuple[List[dict], List[dict]]:
        """用 LLM 从扁平 content 重建章节标题。

        分批喂 LLM，每批识别「哪些块是章节标题 + 标题文本 + 层级」。
        返回 (headings, filtered_content)：
          - headings: 与 _build_tree 兼容的标题列表（source="llm_reconstruct"）
          - filtered_content: 移除标题块后的正文（其余块原样保留，不做改写）
        重建失败返回 ([], content)（保持扁平树）。
        """
        from config_loader import cfg
        batch_size = int(cfg("structure.reconstruct.batch_size", 20))
        max_calls = int(cfg("structure.reconstruct.max_llm_calls", 30))

        llm = _get_reconstruct_llm()
        if llm is None:
            return [], content

        section_items: List[dict] = []  # [{block_index, title, level}]
        calls = 0
        for start in range(0, len(content), batch_size):
            if calls >= max_calls:
                logger.warning("LLM 重建达到最大调用次数 %d，停止识别", max_calls)
                break
            batch = content[start:start + batch_size]
            items = self._llm_extract_sections(llm, batch, start)
            calls += 1
            section_items.extend(items)

        if not section_items:
            return [], content

        # 标题块从 content 移除（它们现在是 section 节点，不再是正文节点）
        title_indices = {it["block_index"] for it in section_items}
        filtered = [c for i, c in enumerate(content) if i not in title_indices]

        headings: List[dict] = []
        for it in section_items:
            blk = content[it["block_index"]]
            headings.append({
                "title": it["title"],
                "level": max(0, int(it.get("level", 0) or 0)),
                "page": blk.get("page", 0),
                "order": blk["order"],
                "source": "llm_reconstruct",
                "source_id": f"llm_reconstruct:{it['block_index']}",
            })
        return headings, filtered

    def _llm_extract_sections(self, llm, batch: List[dict], offset: int) -> List[dict]:
        """LLM 从一批块里识别章节标题。

        Args:
            llm:    重建 LLM
            batch:  一批 content 块
            offset: 该批在全局 content 中的起始索引（用于把批次内 index 转全局）

        Returns:
            [{block_index(全局), title, level}]，失败返回空列表。
        """
        from llm import invoke_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        # 构造块列表（批次内编号 0..N-1）
        lines = []
        for i, blk in enumerate(batch):
            text = (blk.get("text", "") or "").strip().replace("\n", " ")
            lines.append(f"[{i}] {text[:80]}")
        block_text = "\n".join(lines)

        system = (
            "你是文档结构分析助手。请识别给定文本块列表中哪些是「章节标题」"
            "（章/节/小节标题、带编号的标题等），其余是正文。"
            "严格只输出 JSON："
            '{"sections": [{"index": 3, "title": "1.1 背景", "level": 1}]}。'
            "index 是块编号（方括号里的数字，批次内从 0 开始）；"
            "level 是层级（0=章，1=节，2=小节）。"
            "只输出 JSON，不要任何解释、不要代码块标记。"
        )
        prompt = (
            f"文本块列表（编号 / 内容开头）：\n{block_text}\n\n"
            f"请识别哪些块是章节标题。"
        )

        try:
            text = invoke_llm(llm, [SystemMessage(content=system),
                                    HumanMessage(content=prompt)])
            text = (text or "").strip()
            # 容错：剥离可能的 ```json ``` 代码块标记（统一用 common/text_utils）
            from common.text_utils import strip_code_fence
            text = strip_code_fence(text)
            obj = parse_json(text)
            if obj is None:
                return []
            sections = obj.get("sections") or []
        except Exception as e:
            logger.warning("LLM 章节识别失败: %s", e)
            return []

        items: List[dict] = []
        for s in sections:
            try:
                idx = int(s.get("index", -1))
                title = str(s.get("title", "")).strip()
                if idx < 0 or idx >= len(batch) or not title:
                    continue
                items.append({
                    "block_index": offset + idx,
                    "title": title,
                    "level": int(s.get("level", 0) or 0),
                })
            except (ValueError, TypeError):
                continue
        return items

    # -- 建树（通用） -------------------------------------------------------
    def _build_tree(self, headings: List[dict], content: List[dict], doc_id: str, doc_title: str = "") -> TreeNode:
        root = self._new_node("document", doc_id=doc_id, level=-1)
        root.title = doc_title
        root.section_path = []

        # 去重 + 排序
        headings = _dedup_headings(headings)
        headings.sort(key=lambda h: h["order"])
        content.sort(key=lambda c: c["order"])

        # 建 section 节点（含层级 + section_path）
        stack = [root]
        heading_nodes = []  # (order, node)
        for h in headings:
            node = self._new_node(
                "section",
                doc_id=doc_id,
                title=h["title"],
                level=h["level"],
                page=h.get("page", 0),
                source_type=h.get("source", ""),
                source_id=h.get("source_id", ""),
            )
            while len(stack) > 1 and stack[-1].level >= node.level:
                stack.pop()
            node.parent_node_id = stack[-1].node_id
            stack[-1].children.append(node)
            node.order = len(stack[-1].children) - 1
            node.section_path = stack[-1].section_path + [node.order]
            stack.append(node)
            heading_nodes.append((h["order"], node))

        # 内容归位：归到「顺序在其之前的最近标题」
        for c in content:
            target = root
            for order, node in heading_nodes:
                if order <= c["order"]:
                    target = node
                else:
                    break
            child = self._new_node(
                c["type"],
                doc_id=doc_id,
                text=c["text"],
                page=c.get("page", 0),
                bbox=c.get("bbox"),
                source_type=c.get("source_type", ""),
                source_id=c.get("source_id", ""),
            )
            child.parent_node_id = target.node_id
            child.section_path = target.section_path
            target.children.append(child)
            child.order = len(target.children) - 1

        return root

    # -- PDF ----------------------------------------------------------------
    def _parse_pdf(self, filepath: str, on_progress=None):
        from deepdoc.parser.pdf_parser import RAGFlowPdfParser
        from deepdoc.parser.utils import extract_pdf_outlines

        # zoomin 从配置读取（config.deepdoc.zoomin，默认 3）：控制 PDF 渲染分辨率
        # 与 OCR/版面识别精度。3 为最高精度（识别小字/细节），降低可显著提速但
        # 可能漏检小字号内容，属「质量/速度」权衡，故配置化而非硬编码。
        try:
            from config_loader import cfg
            zoomin = int(cfg("deepdoc.zoomin", 3))
        except Exception:
            zoomin = 3

        # deepdoc 的 callback 签名不固定：OCR 阶段传 1 个参数 (progress)，
        # 版面/表格/合并阶段传 2 个参数 (progress, msg)。统一适配成
        # on_progress("PARSING", int(progress*100))，让前端「解析中」显示真实进度。
        def _cb(progress, msg=None):
            if on_progress:
                try:
                    on_progress("PARSING", int(float(progress) * 100))
                except Exception:
                    pass

        outlines = extract_pdf_outlines(filepath)
        parser = RAGFlowPdfParser()
        boxes = parser.parse_into_bboxes(filepath, callback=_cb, zoomin=zoomin)

        # 排序：按阅读顺序 (page, top)
        boxes = sorted(boxes, key=lambda b: (b.get("page_number", 0), b.get("top", 0)))

        headings = []
        # outline 标题（depth -> level 0-based）
        outline_norm = set()
        for idx, (title, depth, page) in enumerate(outlines):
            title = title.strip()
            if not title:
                continue
            headings.append({
                "title": title,
                "level": max(0, depth),
                "page": page,
                "order": (page, -1.0),
                "source": "outline",
                "source_id": f"outline:{idx}",
            })
            outline_norm.add(_normalize_title(title))

        # 文档主标题（独立大标题，如论文题目）：识别后作为 document 的 title，
        # 不生成 section 节点，避免误判成「章」。
        doc_title = self._detect_document_title(boxes)
        doc_title_norm = _normalize_title(doc_title) if doc_title else ""

        # bbox title 元素：校准 outline 标题的真实坐标（修复内容归位错位）。
        #
        # 关键：outline 只提供 page、无页内 top，若直接用其 (page, -1.0) 参与
        # 内容归位（order <= 内容 order），会导致「上一章节标题之后、本章标题
        # 之前的正文」（top > -1.0）被错误归到本章，进而使章节摘要整体错位。
        # 这里用 bbox title 的真实 (page_number, top) 覆盖 outline 的占位坐标。
        for b in boxes:
            if b.get("layout_type") != "title":
                continue
            text = str(b.get("text", "")).strip()
            if not text:
                continue
            norm = _normalize_title(text)
            # 跳过文档主标题
            if doc_title_norm and norm == doc_title_norm:
                continue
            # 尝试校准已有 outline 标题（top 仍为 -1.0 占位、且标题匹配）
            calibrated = False
            if norm in outline_norm:
                for h in headings:
                    if (h.get("source") == "outline"
                            and h.get("order") is not None
                            and h["order"][1] < 0
                            and _title_matches(text, h["title"])):
                        h["order"] = (b.get("page_number", 0), b.get("top", 0.0))
                        h["page"] = b.get("page_number", 0)
                        calibrated = True
                        break
            if calibrated:
                continue
            # 否则作为新标题（outline 之外的标题元素）
            level = _heading_level(text)
            if level is None:
                level = self._infer_level_by_height(text, boxes)
            headings.append({
                "title": text,
                "level": level,
                "page": b.get("page_number", 0),
                "order": (b.get("page_number", 0), b.get("top", 0.0)),
                "source": "bbox",
                "source_id": self._make_source_id(b),
            })

        # 内容元素
        content = []
        for b in boxes:
            lt = b.get("layout_type", "text")
            if lt in _IGNORED_LAYOUTS or lt == "title":
                continue
            text = str(b.get("text", "")).strip()
            if not text:
                continue
            content.append({
                "type": _LAYOUT_TO_TYPE.get(lt, "paragraph"),
                "text": text,
                "page": b.get("page_number", 0),
                "order": (b.get("page_number", 0), b.get("top", 0.0)),
                "bbox": [b.get("x0", 0), b.get("top", 0), b.get("x1", 0), b.get("bottom", 0)],
                "source_type": lt,
                "source_id": self._make_source_id(b),
            })

        return headings, content, doc_title

    def _detect_document_title(self, boxes) -> str:
        """识别文档主标题（如论文题目）。

        条件（同时满足才判定，避免把章标题误判成文档标题）：
        1. 无编号（``_heading_level`` 返回 None）
        2. 是阅读顺序第一个 title 元素
        3. 字号显著大于其余 title 元素的最大字号（>= 1.3 倍）
        """
        title_boxes = [b for b in boxes
                       if b.get("layout_type") == "title" and str(b.get("text", "")).strip()]
        if len(title_boxes) < 2:
            return ""

        def height(b):
            return b.get("bottom", 0) - b.get("top", 0)

        first = min(title_boxes, key=lambda b: (b.get("page_number", 0), b.get("top", 0)))
        text = str(first.get("text", "")).strip()
        if _heading_level(text) is not None:
            return ""
        # 字号必须显著大于其余 title
        hs = sorted([height(b) for b in title_boxes], reverse=True)
        if hs[0] >= 1.3 * hs[1] and height(first) == hs[0]:
            return text
        return ""

    def _infer_level_by_height(self, text: str, boxes) -> int:
        """无编号匹配时，用字号（box 高度）推断层级：高度显著大于标题中位数 -> 章。"""
        heights = [b["bottom"] - b["top"] for b in boxes
                   if b.get("layout_type") == "title" and str(b.get("text", "")).strip()]
        cur = 0.0
        for b in boxes:
            if b.get("layout_type") == "title" and str(b.get("text", "")).strip() == text:
                cur = b["bottom"] - b["top"]
                break
        if not heights or cur <= 0:
            return 1
        median = sorted(heights)[len(heights) // 2]
        if median > 0 and cur >= 1.4 * median:
            return 0
        return 1

    # -- DOCX ---------------------------------------------------------------
    def _parse_docx(self, filepath: str):
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        from docx.oxml.ns import qn

        doc = Document(filepath)
        headings = []
        content = []
        seq = 0

        def heading_level(style_name: str) -> Optional[int]:
            m = re.search(r"(?:heading|标题)\s*([1-9])", style_name or "", re.I)
            return int(m.group(1)) - 1 if m else None

        for child in doc.element.body.iterchildren():
            if child.tag == qn("w:p"):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue
                style = p.style.name if p.style is not None else ""
                lvl = heading_level(style)
                if lvl is not None:
                    headings.append({
                        "title": text,
                        "level": lvl,
                        "page": 0,
                        "order": (0, seq),
                        "source": "docx_style",
                        "source_id": f"para:{seq}",
                    })
                else:
                    content.append({
                        "type": "paragraph",
                        "text": text,
                        "page": 0,
                        "order": (0, seq),
                        "bbox": None,
                        "source_type": "paragraph",
                        "source_id": f"para:{seq}",
                    })
                seq += 1
            elif child.tag == qn("w:tbl"):
                t = Table(child, doc)
                rows = []
                for row in t.rows:
                    cells = [c.text.strip() for c in row.cells]
                    rows.append(" | ".join(cells))
                content.append({
                    "type": "table",
                    "text": "\n".join(rows),
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "table",
                    "source_id": f"table:{seq}",
                })
                seq += 1

        return headings, content

    # -- Markdown -----------------------------------------------------------
    def _parse_markdown(self, filepath: str):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        headings = []
        content = []
        lines = text.split("\n")
        i = 0
        seq = 0
        code_fence = None

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 代码块保护：``` 或 ~~~ 之间不识别标题
            if code_fence is None and stripped.startswith(("```", "~~~")):
                code_fence = stripped[:3]
                buf = [line]
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if lines[i].strip().startswith(code_fence):
                        i += 1
                        break
                    i += 1
                content.append({
                    "type": "paragraph",
                    "text": "\n".join(buf),
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "code_block",
                    "source_id": f"line:{seq}",
                })
                seq += 1
                code_fence = None
                continue

            # 标题
            m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if m:
                headings.append({
                    "title": m.group(2).strip(),
                    "level": len(m.group(1)) - 1,
                    "page": 0,
                    "order": (0, seq),
                    "source": "markdown",
                    "source_id": f"line:{seq}",
                })
                seq += 1
                i += 1
                continue

            # 表格行（含 | 的行聚合为一个 table 块）
            if "|" in stripped and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|", lines[i + 1]):
                buf = [line]
                i += 2
                while i < len(lines) and "|" in lines[i].strip():
                    buf.append(lines[i])
                    i += 1
                content.append({
                    "type": "table",
                    "text": "\n".join(buf),
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "table",
                    "source_id": f"line:{seq}",
                })
                seq += 1
                continue

            # 普通文本块：连续非空行聚合
            if stripped:
                buf = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6})\s+", lines[i].strip()):
                    buf.append(lines[i])
                    i += 1
                content.append({
                    "type": "paragraph",
                    "text": "\n".join(buf),
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "paragraph",
                    "source_id": f"line:{seq}",
                })
                seq += 1
                continue

            i += 1

        return headings, content

    # -- HTML ---------------------------------------------------------------
    def _parse_html(self, filepath: str):
        from bs4 import BeautifulSoup

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all(["style", "script"]):
            tag.decompose()

        headings = []
        content = []
        seq = 0
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li"]):
            name = el.name.lower()
            txt = el.get_text(" ", strip=True)
            if not txt:
                continue
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                headings.append({
                    "title": txt,
                    "level": int(name[1]) - 1,
                    "page": 0,
                    "order": (0, seq),
                    "source": "html",
                    "source_id": f"tag:{seq}",
                })
            elif name == "table":
                rows = []
                for tr in el.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                    rows.append(" | ".join(cells))
                content.append({
                    "type": "table",
                    "text": "\n".join(rows),
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "table",
                    "source_id": f"tag:{seq}",
                })
            else:
                content.append({
                    "type": "paragraph",
                    "text": txt,
                    "page": 0,
                    "order": (0, seq),
                    "bbox": None,
                    "source_type": "paragraph",
                    "source_id": f"tag:{seq}",
                })
            seq += 1

        return headings, content

    # -- TXT ----------------------------------------------------------------
    def _parse_txt(self, filepath: str):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        content = []
        seq = 0
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            content.append({
                "type": "paragraph",
                "text": block,
                "page": 0,
                "order": (0, seq),
                "bbox": None,
                "source_type": "paragraph",
                "source_id": f"block:{seq}",
            })
            seq += 1

        return [], content

    # -- 工具 ---------------------------------------------------------------
    def _make_source_id(self, box) -> str:
        """基于版面坐标生成溯源键。

        用 page + 四角坐标（x0/top/x1/bottom，各保留 2 位小数）。相比旧版只取
        ``page:x0:top`` 且 round(..., 1)，这里加入右下角坐标并提高精度，避免
        同一页面内坐标接近的不同 box（如同一行多个元素、密集表格单元格）在
        四舍五入后碰撞出相同 source_id，进而产生重复 node_id 触发入库时的
        UNIQUE constraint failed。
        """
        page = box.get("page_number", 0)
        x0 = round(float(box.get("x0", 0)), 2)
        top = round(float(box.get("top", 0)), 2)
        x1 = round(float(box.get("x1", 0)), 2)
        bottom = round(float(box.get("bottom", 0)), 2)
        return f"{page}:{x0}:{top}:{x1}:{bottom}"

    def _new_node(self, type_: str, doc_id: str = "", **kwargs) -> TreeNode:
        """创建节点。node_id 基于 doc_id + source_id 确定性生成（跨解析稳定）。"""
        source_id = kwargs.get("source_id", "")
        if doc_id:
            if source_id:
                node_id = f"{doc_id}:{source_id}"
            elif type_ == "document":
                node_id = f"{doc_id}:root"
            else:
                self._seq += 1
                node_id = f"{doc_id}:n{self._seq:04d}"
        else:
            self._seq += 1
            node_id = f"n{self._seq:05d}"

        # 唯一性兜底：即便 source_id 仍碰撞（极端情况下坐标完全相同），也保证
        # node_id 全局唯一，杜绝 save_tree 时的 UNIQUE constraint failed。
        base = node_id
        n = 1
        while node_id in self._used_ids:
            node_id = f"{base}#{n}"
            n += 1
        self._used_ids.add(node_id)

        return TreeNode(node_id=node_id, type=type_, doc_id=doc_id, **kwargs)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def _get_reconstruct_llm():
    """获取目录树重建用的 LLM 实例；未配置任何可用模型时返回 None（跳过重建）。

    summary 模型优先，回退 tool_llm/llm（统一走 llm_factory）。
    """
    from llm_factory import get_model
    return get_model("summary", "tool_llm", "llm")


def compute_doc_id(filepath: str) -> str:
    """基于文件内容计算稳定文档 ID（SHA256 前 16 位，形如 ``doc_xxxx``）。

    同一文件（即使重命名 / 重传）得到同一 doc_id，可用于：
    - 识别「文档已存在」
    - 作为树节点 / chunk ID 的稳定前缀
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "doc_" + h.hexdigest()[:16]


def build_document_tree(filepath: str, on_progress=None) -> TreeNode:
    """结构归位入口：返回 document 根节点。

    on_progress: 可选进度回调 on_progress(stage, progress)，用于解析阶段的
    前端进度展示（deepdoc 版面解析耗时最长且此前无进度）。
    """
    return StructureResolver().resolve(filepath, on_progress=on_progress)


def print_tree(node: TreeNode, indent: int = 0) -> None:
    """打印树结构（供调试 / 验证）。"""
    for child in node.children:
        if child.type == "section":
            mark = "#" * (child.level + 1)
            print(f"{'  ' * indent}{mark} [{child.page}页] {child.title}")
        else:
            preview = child.text[:50].replace("\n", " ")
            print(f"{'  ' * indent}  · [{child.type}] {preview}")
        print_tree(child, indent + 1)


def count_nodes(node: TreeNode) -> dict:
    """统计各类节点数量。"""
    result = {"section": 0, "paragraph": 0, "table": 0, "figure": 0}
    for c in node.children:
        if c.type == "section":
            result["section"] += 1
            sub = count_nodes(c)
            for k in result:
                result[k] += sub[k]
        elif c.type in result:
            result[c.type] += 1
    return result
