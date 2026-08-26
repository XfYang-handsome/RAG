# -*- coding: utf-8 -*-
"""步骤 3：结构树 → Retrieval Chunk 切分与对齐。

把结构树（Structure Resolver 的产出）的叶子节点（paragraph / table / figure）
切分为检索单元（Retrieval Chunk）。

设计要点（与架构定稿一致）：

1. 两层彻底分离
   - Structure Tree 的节点只负责结构 / 溯源，不生成 embedding
   - Retrieval Chunk 独立，通过 ``parent_node_id`` 挂回树节点

2. 切分规则
   - 单个叶子节点文本 <= max_chars：独立成 1 个 chunk
   - 单个叶子节点文本 >  max_chars：按句子边界切分为多个 chunk
   - 同一 section 下连续多个短叶子节点（<= min_chars）：合并成 1 个 chunk

3. 字段语义（与定稿一致）
   - parent_node_id  : 上下文恢复锚点（来源的最细粒度共同祖先，Section 或叶子节点）
   - source_node_ids : 完整出处列表（合并场景为多个叶子节点，溯源用）
   - chunk_seq       : 全局阅读顺序（邻近块扩展的前提）
   - section_path    : 章节路径（属性，非 ID）
   - doc_id          : 稳定文档 ID（内容哈希）

说明：本模块只做「切分 + 对齐」，不涉及 embedding / Milvus 写入（后续步骤处理）。
"""

from __future__ import annotations

import re
from typing import List

from structure_resolver import TreeNode

# 切分阈值（字符数）
MAX_CHUNK_CHARS = 800   # 单 chunk 上限，超过则切分
MIN_CHUNK_CHARS = 150   # 短叶子节点阈值，低于此且同 section 则合并

# 句子边界（用于长文本切分时尽量不切断句子）
_SENT_END_RE = re.compile(r"[。！？!?；;\n]")


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """把长文本按句子边界切分为 <= max_chars 的片段。

    空文本 / 纯空白文本返回空列表（避免产出无法向量化的空 chunk，
    空字符串提交给 embedding 服务会报 400 参数无效）。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 向前找最近的句子边界，避免硬切
            window = text[start:end]
            m = None
            for match in _SENT_END_RE.finditer(window):
                m = match
            if m and m.end() > max_chars * 0.5:
                end = start + m.end()
        pieces.append(text[start:end].strip())
        start = end
    return [p for p in pieces if p]


def _make_chunk(
    text: str,
    parent_node: TreeNode,
    source_nodes: List[TreeNode],
    section_path: List[int],
    doc_id: str,
    chunk_seq: int,
) -> dict:
    """构造一个 Retrieval Chunk（字典形式，不含 vector）。"""
    return {
        "chunk_id": f"{doc_id}:chunk:{chunk_seq}",
        "text": text,
        "parent_node_id": parent_node.node_id,
        "source_node_ids": [n.node_id for n in source_nodes],
        "doc_id": doc_id,
        "section_path": list(section_path),
        "chunk_seq": chunk_seq,
        # 附带：chunk 类型（取来源节点类型，合并时以第一个为准）
        "node_type": source_nodes[0].type if source_nodes else parent_node.type,
    }


def build_chunks(
    root: TreeNode,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> List[dict]:
    """遍历结构树，把叶子节点切分为 Retrieval Chunk 列表。

    Args:
        root: 文档树根节点（TreeNode, type=document）
        max_chars: 单 chunk 最大字符数
        min_chars: 短叶子节点合并阈值

    Returns:
        chunk 列表（按阅读顺序，含 chunk_seq / parent_node_id / source_node_ids 等）
    """
    chunks: List[dict] = []
    doc_id = root.doc_id

    # 处理一个 section 容器内的直属叶子节点（合并短段落、切分长段落）
    def _process_section(section: TreeNode):
        leaves = [c for c in section.children if c.type != "section"]
        buffer: List[TreeNode] = []  # 待合并的短叶子节点
        buffer_chars = 0             # 累计字符数（含分隔符），防止合并超长

        def _flush_buffer():
            nonlocal buffer_chars
            if not buffer:
                return
            # 合并成一个 chunk：parent 是 section，source 是这些叶子节点
            merged_text = "\n\n".join(n.text for n in buffer).strip()
            if not merged_text:
                # 过滤纯空白合并结果（空段落等），避免产出无法向量化的空 chunk
                buffer.clear()
                buffer_chars = 0
                return
            chunks.append(_make_chunk(
                merged_text, section, buffer, section.section_path, doc_id, len(chunks),
            ))
            buffer.clear()
            buffer_chars = 0

        for leaf in leaves:
            if len(leaf.text) <= min_chars:
                # 合并后超过 max_chars 先 flush，避免短段落无限合并成超长 chunk
                # （超长 chunk 超过 embedding 模型 token 上限会报 400 参数无效）
                if buffer and buffer_chars + len(leaf.text) + 2 > max_chars:
                    _flush_buffer()
                buffer.append(leaf)
                buffer_chars += len(leaf.text) + 2  # 2 为 "\n\n" 分隔符
            else:
                _flush_buffer()
                for piece in _split_long_text(leaf.text, max_chars):
                    chunks.append(_make_chunk(
                        piece, leaf, [leaf], leaf.section_path, doc_id, len(chunks),
                    ))

        _flush_buffer()

        # 递归处理子 section
        for c in section.children:
            if c.type == "section":
                _process_section(c)

    _process_section(root)
    return chunks


def chunk_count(root: TreeNode, **kwargs) -> int:
    """返回结构树切分出的 chunk 数量。"""
    return len(build_chunks(root, **kwargs))
