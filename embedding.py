"""
================================================================================
Embedding 向量化模块 — 基于 chunklet-py 语义分块 + OpenAI 兼容 API
================================================================================

核心能力：
  1. 文本 → 父子块语义切分（Parent-Child Chunking）
  2. 子块 → 批量向量化（Embedding）
  3. 支持纯文本 + PDF/DOCX 等文档格式自动提取

切分策略（父子块）：
  ┌────────────────────────────────────────────┐
  │ 父块 0（大语义窗口，512–1024 tokens）      │  ← 粗检索 + LLM 上下文
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
  │  │ 子块 0   │ │ 子块 1   │ │ 子块 2   │   │  ← 小语义单元，精准检索
  │  │ 128–256  │ │ 128–256  │ │ 128–256  │   │
  │  │ tokens   │ │ tokens   │ │ tokens   │   │
  │  └──────────┘ └──────────┘ └──────────┘   │
  └────────────────────────────────────────────┘

  检索流程：用户 query → 子块向量检索 → 找到子块 → 追溯父块 → LLM 上下文

chunklet-py 语义切分原理：
  使用 max_tokens 参数按 token 上限控制 chunk 大小，仍沿自然句子边界切分。
  chunk 会沿句子边界累加，直到即将超过 max_tokens 上限才切分，
  因此每个 chunk 是完整语义单元，且大小接近 token 上限。

依赖：
  - chunklet-py：语义分块引擎（含 PDF/DOCX 解析）
  - openai：OpenAI 兼容 API 调用（用于 embedding 向量化）
================================================================================
"""

import os
import glob
from typing import List, Tuple
from openai import OpenAI

from config_loader import config

# ============================================================================
# chunklet-py 检测
# ============================================================================
try:
    from chunklet import DocumentChunker
    HAS_CHUNKLET = True   # 已安装，可进行语义分块 + 文档解析
except ImportError:
    HAS_CHUNKLET = False  # 未安装，降级为固定大小分块

# ============================================================================
# 配置加载 — 从 config/config.json 读取
# ============================================================================
EMBED_BATCH_SIZE = 100  # 批量向量化时每批最大条数（避免单次 API 调用过大）

_chunk_cfg = config["chunking"]
DEFAULT_PARENT_CHUNK_SIZE = _chunk_cfg["parent_chunk_size"]  # 父块目标大小（字符）
DEFAULT_CHILD_CHUNK_SIZE  = _chunk_cfg["child_chunk_size"]   # 子块目标大小（字符）
DEFAULT_PARENT_OVERLAP    = _chunk_cfg["parent_overlap"]     # 父块间重叠字符数
DEFAULT_CHILD_OVERLAP     = _chunk_cfg["child_overlap"]      # 子块间重叠字符数

# 支持上传的文件格式（从 config/config.json 读取）
SUPPORTED_EXTENSIONS = set(config["supported_extensions"])

# 需要 chunklet-py 原生解析的二进制/文档格式（非纯文本）
# 注：chunklet-py 只支持 pdf/docx/pptx/epub/odt/eml，不含 xlsx/rtf
_BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx", ".epub", ".odt", ".eml"}

# xlsx 由 openpyxl 单独解析（chunklet 不支持，openpyxl 已内置）
_XLSX_EXTENSIONS = {".xlsx"}

# rtf 由 striprtf 单独解析（chunklet 不支持）
_RTF_EXTENSIONS = {".rtf"}


# ============================================================================
# 文件读取工具函数
# ============================================================================

def extract_document_text(filepath: str) -> str:
    """
    提取文档（PDF/DOCX/PPTX 等）的纯文本内容。

    实现要点：
      chunklet-py 的 chunk_file 对 PDF 等格式会报错（要求用 chunk_files 并行），
      而 chunk_files 依赖 mpire 多进程，在 Windows 上存在 pickle bug。
      因此这里绕过 chunklet 的并行逻辑，直接实例化对应格式的 processor，
      调用其 extract_text() 方法提取纯文本，再用 chunk_text 分块。

    Args:
        filepath: 文档文件绝对路径

    Returns:
        提取出的纯文本内容
    """
    if not HAS_CHUNKLET:
        raise ImportError("chunklet-py 未安装")

    ext = os.path.splitext(filepath)[1].lower()
    chunker = DocumentChunker()
    processors = chunker.processors

    if ext not in processors:
        raise ValueError(f"不支持的文档格式: {ext}")

    # 实例化对应格式的 processor（如 PdfProcessor(file_path)）
    processor_cls = processors[ext]
    processor = processor_cls(filepath)

    # extract_text() 返回 Generator（逐页/逐段产出文本），拼接为完整文本
    texts = list(processor.extract_text())
    return "\n\n".join(texts)


def _extract_xlsx_text(filepath: str) -> str:
    """
    提取 Excel（.xlsx）文件的文本内容。

    chunklet-py 不支持 xlsx，这里用 openpyxl 逐 sheet、逐单元格提取，
    单元格之间用制表符、行之间用换行、sheet 之间用分隔符拼接，
    生成近似表格的纯文本。

    Args:
        filepath: .xlsx 文件绝对路径

    Returns:
        提取出的纯文本内容
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("解析 .xlsx 需要安装 openpyxl: pip install openpyxl")

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    sheets_text = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            # 过滤整行为空的行，单元格 None → 空字符串
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                rows.append("\t".join(cells))
        if rows:
            sheets_text.append(f"[Sheet: {ws.title}]\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sheets_text)


def _extract_rtf_text(filepath: str) -> str:
    """
    提取 RTF（富文本）文件的纯文本内容。

    chunklet-py 不支持 rtf，这里用 striprtf 库剥离 RTF 控制字符，
    只保留正文文本。

    Args:
        filepath: .rtf 文件绝对路径

    Returns:
        提取出的纯文本内容
    """
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise ImportError("解析 .rtf 需要安装 striprtf: pip install striprtf")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        rtf_content = f.read()
    return rtf_to_text(rtf_content)


def _read_text_with_fallback(filepath: str) -> str:
    """纯文本文件读取，UTF-8 失败时回退 gbk / gb2312 / latin-1。"""
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in ("gbk", "gb2312", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # 全部失败：用 replace 兜底
    return raw.decode("utf-8", errors="replace")


def _read_file_content(filepath: str) -> str:
    """
    读取任意文件内容为纯文本字符串。

    支持四种类型：
      - 纯文本（.txt / .md / .py / .json 等）：UTF-8 优先，回退 gbk/gb2312/latin-1
      - 文档格式（.pdf / .docx / .pptx / .epub / .odt / .eml）：chunklet processor 提取
      - Excel（.xlsx）：openpyxl 提取
      - RTF（.rtf）：striprtf 提取

    Args:
        filepath: 文件绝对路径

    Returns:
        文件的纯文本内容
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _BINARY_EXTENSIONS and HAS_CHUNKLET:
        # --- 文档格式：用 chunklet-py processor 提取文本（绕开多进程 bug） ---
        return extract_document_text(filepath)
    elif ext in _XLSX_EXTENSIONS:
        # --- Excel 格式：openpyxl 提取 ---
        return _extract_xlsx_text(filepath)
    elif ext in _RTF_EXTENSIONS:
        # --- RTF 格式：striprtf 提取 ---
        return _extract_rtf_text(filepath)
    else:
        # --- 纯文本文件：UTF-8 优先，回退多种编码 ---
        return _read_text_with_fallback(filepath)


def read_files_from_path(path: str) -> List[Tuple[str, str]]:
    """
    从路径读取文件内容。

    支持三种输入形式：
      - 单个文件路径 → 读取该文件
      - 文件夹路径 → 递归遍历，读取所有支持格式的文件
      - 不符合 → 返回空列表

    Args:
        path: 文件或文件夹路径

    Returns:
        [(文件路径, 文本内容), ...] 列表
    """
    results = []
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS or ext == "":
            try:
                results.append((path, _read_file_content(path)))
            except Exception as e:
                print(f"  警告: 读取文件 {path} 失败: {e}")
    elif os.path.isdir(path):
        # os.walk 递归遍历所有子目录
        for root, dirs, files in os.walk(path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(root, filename)
                    try:
                        results.append((filepath, _read_file_content(filepath)))
                    except Exception as e:
                        print(f"  警告: 读取文件 {filepath} 失败: {e}")
    return results


def resolve_inputs(args: list) -> List[Tuple[str, str]]:
    """
    智能解析输入参数，自动区分"文件路径"和"文本字符串"。

    判断逻辑：
      - 如果 arg 在文件系统存在 → 当作文件/文件夹路径读取
      - 否则 → 当作纯文本字符串直接返回

    这使命令行同时支持：
      python __main__.py emb ./docs/           # 文件夹
      python __main__.py emb readme.txt        # 单个文件
      python __main__.py emb "人工智能是..."    # 纯文本

    Args:
        args: 命令行参数列表

    Returns:
        [(来源标识, 文本内容), ...] 文件名为路径，文本字符串标识为 "__text__"
    """
    results = []
    for arg in args:
        if os.path.exists(arg):
            results.extend(read_files_from_path(arg))
        else:
            results.append(("__text__", arg))
    return results


# ============================================================================
# ParentChildChunker — 父子块语义切分器
# ============================================================================

class ParentChildChunker:
    """
    父子块语义切分器（基于 chunklet-py 的 DocumentChunker）。

    核心思想 — "分工合作"：
      ┌──────────────────────────────────────────────────────────────┐
      │ 子块（Child Chunk）：小块、精准                                │
      │   - 用于向量检索，匹配用户问题，确保找得准                      │
      │   - 每个父块内切分，max_tokens=256（128–256 tokens）          │
      │   - 只向量化子块，存入 Milvus 子 Collection                   │
      ├──────────────────────────────────────────────────────────────┤
      │ 父块（Parent Chunk）：大块、完整                               │
      │   - 不直接用于检索                                            │
      │   - 全文切分，max_tokens=1024（512–1024 tokens）              │
      │   - 不向量化，只存原文                                         │
      │   - 当子块被命中后，找到对应父块完整内容送给 LLM               │
      └──────────────────────────────────────────────────────────────┘

    关系：子块通过 parent_id (UUID) 绑定到父块。

    chunklet-py 的 DocumentChunker.chunk_text() 核心参数：
      - max_tokens=M     : 每个 chunk 的 token 硬上限（沿句子边界向上填充）
      - lang="zh"        : 中文模式，自动识别中文句子边界
      - token_counter    : 可插拔 token 计数器（默认按字符数）
      - overlap_percent  : chunk 间重叠百分比

    为什么用 max_tokens 控制大小：
      切分仍然在句子边界进行（不会在句子中间硬切），
      但 chunk 大小由 max_tokens 上限约束，保证落在目标 token 范围内。
    """

    def __init__(
        self,
        parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        child_chunk_size:  int = DEFAULT_CHILD_CHUNK_SIZE,
        parent_overlap:    int = DEFAULT_PARENT_OVERLAP,
        child_overlap:     int = DEFAULT_CHILD_OVERLAP,
    ):
        """
        Args:
            parent_chunk_size: 父块目标字符数（仅用于 fallback 模式）
            child_chunk_size:  子块目标字符数（仅用于 fallback 模式）
            parent_overlap:    父块间重叠字符数
            child_overlap:     子块间重叠字符数
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size  = child_chunk_size
        self.parent_overlap    = parent_overlap
        self.child_overlap     = child_overlap

    # ------------------------------------------------------------------
    # 核心：按 token 数语义分块
    # ------------------------------------------------------------------
    def _chunk_by_tokens(
        self, text: str, max_tokens_limit: int, overlap_chars: int = 0,
    ) -> List[str]:
        """
        使用 chunklet-py 按 token 数（上限）进行语义分块。

        切分仍然遵循自然句子边界（不会在句子中间硬切），
        但分块大小由 max_tokens 控制：chunk 会沿句子边界不断累加，
        直到即将超过 max_tokens_limit 才切分，因此实际大小会接近上限。

        若 chunklet-py 未安装 → 降级为固定字符切分（1 token ≈ 1.5 字符），
        用 overlap_chars 做相邻块字符重叠。

        Args:
            text:             待切分文本
            max_tokens_limit: 每个 chunk 的 token 硬上限
            overlap_chars:    相邻 chunk 的重叠字符数（chunklet 模式下转百分比）

        Returns:
            切分后的文本列表
        """
        if not HAS_CHUNKLET:
            # Fallback: 固定大小字符切分（中文 1 token ≈ 1.5 字符），支持字符重叠
            size = int(max_tokens_limit * 1.5)
            step = max(1, size - int(overlap_chars))
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunks.append(text[start:end])
                if end >= len(text):
                    break
                start += step
            return chunks

        # 准备 token 计数器（优先 tiktoken）
        token_counter = None
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_counter = lambda s: len(enc.encode(s))
        except Exception:
            pass

        # overlap 字符数 → 百分比（chunklet 只接受 0~75 的百分比）
        overlap_percent = 20
        if overlap_chars and max_tokens_limit:
            overlap_percent = int(overlap_chars / max_tokens_limit * 100)
        overlap_percent = max(0, min(75, overlap_percent))

        chunker = DocumentChunker()
        result = chunker.chunk_text(
            text,
            lang="zh",
            max_tokens=max(max_tokens_limit, 12),
            overlap_percent=overlap_percent,
            token_counter=token_counter or (lambda s: len(s)),
        )
        return [c.content for c in result]

    # ------------------------------------------------------------------
    # 父子块切分主入口
    # ------------------------------------------------------------------
    def chunk(self, text: str, source: str = "") -> dict:
        """
        对文本执行完整的父子块语义切分。

        流程：
          Step 1: 全文 → max_tokens=1024 切分父块（512–1024 tokens）
          Step 2: 每个父块 → max_tokens=256 切分子块（128–256 tokens）
          Step 3: 建立 parent_id 绑定关系

        父块不向量化，子块向量化。
        检索命中子块后，通过 parent_id 追溯父块全文送给 LLM。

        大小控制原理：
          chunklet 的 max_tokens 是「硬上限 + 沿句子边界向上填充」，
          因此父块实际落在 512–1024、子块落在 128–256 的范围内。

        Args:
            text:   输入全文
            source: 来源文件名

        Returns:
            {
                "parent_chunks": [{"index":0,"parent_id":"...","text":"父块全文","source":"..."}],
                "child_chunks":  [{"index":0,"parent_id":"...","text":"子块","vector":[...]},...]
            }
        """
        import uuid

        # ---- Step 1: 切分父块（上限 = self.parent_chunk_size） ----
        parent_texts = self._chunk_by_tokens(
            text,
            max_tokens_limit=self.parent_chunk_size,
            overlap_chars=self.parent_overlap,
        )
        if not parent_texts:
            return {"parent_chunks": [], "child_chunks": []}

        # ---- Step 2: 每个父块切分子块（上限 = self.child_chunk_size） ----
        parent_chunks = []
        child_chunks  = []
        child_index   = 0

        for pi, p_text in enumerate(parent_texts):
            # 给每个父块分配唯一 UUID
            pid = str(uuid.uuid4().hex[:12])
            parent_chunks.append({
                "index":     pi,
                "parent_id": pid,
                "text":      p_text,
                "source":    source,
            })

            # 在父块内切分子块
            child_texts = self._chunk_by_tokens(
                p_text,
                max_tokens_limit=self.child_chunk_size,
                overlap_chars=self.child_overlap,
            )

            for ct in child_texts:
                child_chunks.append({
                    "index":      child_index,
                    "text":       ct,
                    "parent_id":  pid,      # ← 子块绑定父块 UUID
                    "parent_index": pi,     # ← 父块在数组中的索引
                    "source":     source,
                })
                child_index += 1

        return {
            "parent_chunks": parent_chunks,
            "child_chunks":  child_chunks,
        }


# ============================================================================
# ChatOpenAIEmbeddingWrapper — Embedding 向量化封装
# ============================================================================

class ChatOpenAIEmbeddingWrapper:
    """
    基于 OpenAI 兼容 API 的 Embedding 向量化封装。

    核心功能：
      1. 单文本向量化 → embed_text()
      2. 批量向量化（自动分批） → embed_texts()
      3. 父子块切分 + 向量化一体化 → embed_with_parent_child()

    配置来源：
      - online=True  → 使用 config.json models.embedding.online 配置
      - online=False → 使用 config.json models.embedding.offline 配置

    内置 ParentChildChunker，支持一步完成切分+向量化。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        dimensions: int = None,
        parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        child_chunk_size:  int = DEFAULT_CHILD_CHUNK_SIZE,
        parent_overlap:    int = DEFAULT_PARENT_OVERLAP,
        child_overlap:     int = DEFAULT_CHILD_OVERLAP,
    ):
        """
        Args:
            model:     embedding 模型名
            api_key:   API 密钥
            base_url:  API 地址（只需写到 /v1，SDK 自动追加 /embeddings）
            dimensions: 向量维度（None=模型默认）
            parent_chunk_size: 父块大小
            child_chunk_size:  子块大小
        """
        self.embedding_model = model
        self.dimensions       = dimensions
        self.api_key          = api_key
        self.base_url         = base_url

        # OpenAI 兼容客户端（直接调 embeddings API）
        self._openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 内置父子块切分器
        self.chunker = ParentChildChunker(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            parent_overlap=parent_overlap,
            child_overlap=child_overlap,
        )

    # ------------------------------------------------------------------
    # 单文本向量化
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """
        将单段文本转为向量。

        内部调用 OpenAI embeddings API：
          POST {base_url}/embeddings
          {"model": "...", "input": "文本"}

        Raises:
            ConnectionError: API 不可达时抛出，包含具体地址信息
        """
        kwargs = {"model": self.embedding_model, "input": text}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        try:
            response = self._openai_client.embeddings.create(**kwargs, timeout=30)
        except Exception as e:
            raise ConnectionError(
                f"Embedding 服务不可用 (地址: {self.base_url}, 模型: {self.embedding_model}): {e}"
            ) from e
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # 批量向量化（自动分批）
    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str], on_batch=None) -> List[List[float]]:
        """
        批量文本向量化。

        特点：
          - 自动分批（每批最多 EMBED_BATCH_SIZE 条），避免单次请求过大
          - 分批间打印进度（大批量处理时可观测）
          - 返回按原始顺序排列的向量列表
          - 内部按 response.data[].index 排序，确保顺序正确

        Args:
            texts: 文本列表
            on_batch: 可选回调，每处理完一个 batch 调用 on_batch(done, total)，
                     用于异步入库任务上报进度。

        Returns:
            向量列表，texts[i] → vectors[i]
        """
        if not texts:
            return []

        total = len(texts)
        all_vectors: List[List[float]] = []

        # 分批循环
        for batch_start in range(0, total, EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, total)
            batch_texts = texts[batch_start:batch_end]

            kwargs = {"model": self.embedding_model, "input": batch_texts}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            try:
                response = self._openai_client.embeddings.create(**kwargs, timeout=60)
            except Exception as e:
                raise ConnectionError(
                    f"Embedding 服务不可用 (地址: {self.base_url}, 模型: {self.embedding_model}): {e}"
                ) from e

            # API 返回顺序可能不一致，按 index 排序保证顺序
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_vectors.extend(item.embedding for item in sorted_data)

            # 进度提示（仅大批量时显示）
            if total > EMBED_BATCH_SIZE:
                print(f"  embedding 进度: {batch_end}/{total}")
            if on_batch is not None:
                on_batch(batch_end, total)

        return all_vectors

    # ------------------------------------------------------------------
    # 父子块切分 + 向量化一体化
    # ------------------------------------------------------------------
    def embed_with_parent_child(self, text: str, source: str = "", on_batch=None) -> dict:
        """
        完整流程：文本 → 父子块切分 → 子块向量化。

        这是上传文件时的核心入口方法。

        流程：
          1. text → ParentChildChunker.chunk() → 父块 + 子块（文本）
          2. 子块文本 → embed_texts() → 子块向量（写入 child_chunks[].vector）
          3. 父块保留原文（不向量化，仅在 LLM 阶段提供上下文）

        Args:
            text:   全文内容
            source: 来源文件名
            on_batch: 可选进度回调，透传给 embed_texts

        Returns:
            {
                "parent_chunks": [{"index":0,"text":"...","source":"..."}, ...],
                "child_chunks":  [{"index":0,"text":"...","parent_index":0,"vector":[...]}, ...]
            }
        """
        # Step 1: 切分
        result = self.chunker.chunk(text, source=source)

        # Step 2: 批量向量化子块
        child_texts = [c["text"] for c in result["child_chunks"]]
        if child_texts:
            print(f"  开始 embedding ({len(child_texts)} 个子块)...")
            child_vectors = self.embed_texts(child_texts, on_batch=on_batch)
            # 向量回写到子块数据中
            for i, vec in enumerate(child_vectors):
                result["child_chunks"][i]["vector"] = vec
            print(f"  embedding 完成")

        return result
