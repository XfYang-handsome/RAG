"""
================================================================================
数据服务 — 数据库检索 / 入库 / 管理（主程序直接调用）
================================================================================

封装「Embedding 向量化 + Milvus 数据库」为统一的数据服务，供主程序直接调用。

核心能力：
  1. 文本向量化（复用 config/models.json 的 embedding 模型配置，current 选择）
  2. 父子块切分 + 入库（insert_documents）
  3. 向量检索（search_documents）
  4. 数据管理（list_parents / list_children / delete_* / clear）
  5. 数据库管理（list_databases / create_database）

说明：
  当前选中数据库（current_db）持久化在 db.json 的 current 字段，
  当前选中 embedding 模型持久化在 models.json 的 current.embedding 字段，
  二者均由 store_config 统一管理，重启不丢失。

依赖：
  embedding（主项目的父子块切分 + 向量化）
  milvus_store（Milvus 封装）
  store_config（数据库/模型配置读写）
================================================================================
"""

from collections import defaultdict

# ---------------------------------------------------------------------------
# 关键修复：torch 必须在最浅调用栈完整加载（下沉到 db_service 层）。
#
# 根因：deepdoc（PDF 解析）/ summarize_tree（langchain_openai）/
# reranker（transformers）都会在【深层调用栈】触发 torch import。若 torch
# 首次在深层栈初始化，会因 torch._library.utils.get_source 的 inspect 崩溃
# 报 "Only a single TORCH_LIBRARY" / "partially initialized module 'torch'"。
#
# 此前修复放在 __main__.py（服务入口），是【脆弱修复】：任何绕过 __main__
# 直接 import db_service 的入库脚本（如临时重新入库脚本）仍会踩坑。
# 这里下沉到 db_service（所有入库/检索路径的共同底层），在任何会触发
# torch 的模块之前浅栈加载并 patch 容错，彻底覆盖所有入口。
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    try:
        import torch._library.utils as _tlu
        _orig_get_source = _tlu.get_source

        def _safe_get_source(stacklevel=1):
            try:
                return _orig_get_source(stacklevel + 1)
            except Exception:
                return "unknown_source"

        _tlu.get_source = _safe_get_source
        import torch.library as _tlib
        _tlib._utils.get_source = _safe_get_source
    except Exception:
        pass
except ImportError:
    pass

from config_loader import config
from embedding import ChatOpenAIEmbeddingWrapper
from milvus_store import (
    create_store_from_config,
    list_local_databases,
    create_local_database,
    LOCAL_COLLECTION_NAME,
    LOCAL_MILVUS_URL,
)
import store_config


# ============================================================================
# 内部单例缓存（embedding / store 懒加载，随配置切换重建）
# ============================================================================

_embedding_wrapper = None
_embedding_sig = None   # 当前 embedding 实例对应的配置签名（model/api_key/base_url）
_store = None
_store_sig = None       # 当前 store 对应的数据库配置签名


def _current_db() -> dict:
    """获取当前数据库配置（按 db.json 的 current，回退第一个）。"""
    name = store_config.get_current_db()
    db = store_config.get_db_by_name(name) if name else None
    if db is None:
        dbs = store_config.list_dbs()
        return dbs[0] if dbs else None
    return db


def get_embedding() -> ChatOpenAIEmbeddingWrapper:
    """获取当前 embedding 实例（按 models.json 的 current.embedding 懒加载）。

    缓存用配置签名（model/api_key/base_url）判断：同名模型改了配置也会重建。
    """
    global _embedding_wrapper, _embedding_sig
    models = store_config.list_models("embedding")
    if not models:
        raise RuntimeError("未配置 embedding 模型（请在设置中添加）")

    name = store_config.get_current("embedding")
    m = store_config.get_model_by_name("embedding", name) if name else None
    if m is None:
        m = models[0]

    sig = (m.get("model"), m.get("api_key"), m.get("base_url"))
    if _embedding_wrapper is not None and _embedding_sig == sig:
        return _embedding_wrapper

    _embedding_wrapper = ChatOpenAIEmbeddingWrapper(
        model=m.get("model"),
        api_key=m.get("api_key"),
        base_url=m.get("base_url"),
    )
    _embedding_sig = sig
    return _embedding_wrapper


def reset_embedding():
    """重置 embedding 缓存（切换模型后调用）。"""
    global _embedding_wrapper, _embedding_sig
    _embedding_wrapper = None
    _embedding_sig = None


def get_store():
    """获取当前 Milvus store 实例（按 db.json 的 current 懒加载）。

    缓存用数据库配置签名（name/type/url/db_name/token）判断：改了配置也会重建。
    """
    global _store, _store_sig
    db = _current_db()
    if db is None:
        raise RuntimeError("未配置数据库（请在设置中添加数据库）")

    sig = (db.get("name"), db.get("type"), db.get("url"),
           db.get("db_name"), db.get("token"))
    if _store is not None and _store_sig == sig:
        return _store

    _store = create_store_from_config(db, LOCAL_COLLECTION_NAME)
    _store_sig = sig
    return _store


def reset_store():
    """重置 store 缓存（数据库配置变更后调用）。"""
    global _store, _store_sig
    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass
    _store = None
    _store_sig = None


# ============================================================================
# 数据服务函数（供主程序调用）
# ============================================================================

def search_documents(query: str, top_k: int = 5,
                     expand_neighbors: bool = True, neighbor_window: int = 1,
                     max_context_chars: int = 4000, hybrid: bool = False) -> list:
    """
    检索：query → embedding → Milvus 检索（父子块回溯父块原文）。

    支持两种召回模式：
      - 纯向量（hybrid=False）：dense COSINE 检索
      - 混合检索（hybrid=True）：dense 向量 + BM25 稀疏，RRF 融合

    上下文补全：
      1. 章节路径恢复（沿树向上还原章节标题，供生成阶段组装）
      2. 邻近块扩展（同一 section 内、阅读顺序相邻的 chunk，解决答案跨块问题）

    Args:
        expand_neighbors:  是否做邻近块扩展
        neighbor_window:   前后各扩展的块数（默认 1）
        max_context_chars: 上下文总字符预算（命中块优先，剩余给相邻块）
        hybrid:            是否启用混合检索（dense + BM25 + RRF）

    Returns:
        [{"text": ..., "score": ..., "parent_id": ..., 可选 "doc_id",
          "section_path_titles": [...], "section_path_str": "...",
          "is_neighbor": bool, "chunk_seq": int}, ...]
    """
    emb = get_embedding()
    store = get_store()
    query_vector = emb.embed_text(query)
    if hybrid and hasattr(store, "search_hybrid"):
        try:
            results = store.search_hybrid(query_vector, query, top_k=top_k) or []
        except Exception as e:
            # 降级：hybrid 检索（RRF 融合）在 Milvus 并发写入期间会间歇性报
            # "unsupported ID type"，降级到纯 dense 保证检索可用性。
            print(f"[WARN] hybrid 检索失败，降级 dense: {e}")
            results = store.search(query_vector, top_k=top_k) or []
    else:
        results = store.search(query_vector, top_k=top_k) or []

    # 邻近块扩展（仅结构树 chunk）
    if expand_neighbors:
        results = _expand_neighbor_chunks(store, results, neighbor_window, max_context_chars)

    # 结构树 chunk 上下文补全：沿树恢复章节标题路径 + 章节摘要
    # 优化：按 doc 分组 + 批量恢复（消除逐条 N+1 查询）
    try:
        import tree_store
        by_doc = defaultdict(list)
        for r in results:
            doc_id = r.get("doc_id")
            if doc_id:
                by_doc[doc_id].append(r.get("parent_id", ""))
        meta_by_doc = {
            doc_id: tree_store.get_section_meta_batch(doc_id, parent_ids)
            for doc_id, parent_ids in by_doc.items()
        }
        for r in results:
            doc_id = r.get("doc_id")
            if not doc_id:
                continue
            m = (meta_by_doc.get(doc_id) or {}).get(r.get("parent_id", "")) or {}
            path = m.get("path_titles") or []
            r["section_path_titles"] = path
            r["section_path_str"] = " > ".join(path) if path else ""
            r["section_summary"] = m.get("section_summary") or ""
    except Exception:
        pass  # 树库不可用不影响检索主流程

    return results


def _expand_neighbor_chunks(store, results: list, window: int, max_chars: int) -> list:
    """邻近块扩展：命中块优先，剩余预算内补充同 section 的相邻块。

    规则（与架构定稿一致）：
      - 命中块优先占预算
      - 相邻块紧跟对应命中块，按阅读顺序
      - (doc_id, chunk_seq) 去重
      - 不跨 section（由 store.get_neighbor_chunks 保证）
    """
    if not results or not hasattr(store, "get_neighbor_chunks"):
        return results

    # 命中块优先占预算
    hit_chars = sum(len(r.get("text", "")) for r in results)
    remaining = max(0, max_chars - hit_chars)
    if remaining <= 0:
        return results

    expanded = []
    seen = set()  # (doc_id, chunk_seq) 已输出
    for r in results:
        if r.get("doc_id"):
            seen.add((r.get("doc_id"), r.get("chunk_seq")))
        expanded.append(r)

        # 仅结构树命中块扩展相邻块
        if not r.get("doc_id") or remaining <= 0:
            continue
        neighbors = store.get_neighbor_chunks(
            r.get("doc_id"), r.get("section_path", ""), r.get("chunk_seq", 0), window
        )
        for nb in neighbors:
            key = (nb.get("doc_id"), nb.get("chunk_seq"))
            if key in seen:
                continue
            nb_chars = len(nb.get("text", ""))
            if nb_chars > remaining:
                continue
            seen.add(key)
            remaining -= nb_chars
            expanded.append({
                "text":         nb.get("text", ""),
                "score":        r.get("score", 0.0),
                "parent_id":    nb.get("parent_id", ""),
                "doc_id":       nb.get("doc_id", ""),
                "section_path": nb.get("section_path", ""),
                "chunk_seq":    nb.get("chunk_seq", 0),
                "is_neighbor":  True,
            })
    return expanded


def _emit(on_progress, stage: str, progress: int = 0):
    """进度回调薄封装（on_progress 为 None 时静默跳过）。"""
    if on_progress:
        on_progress(stage, progress)


def _generate_summaries(root) -> str:
    """生成章节摘要 + 文档级主旨摘要，返回 abstract。

    增强解析的「摘要增强」：章节摘要（summarize_tree，就地填 root 各 section 的
    summary）+ 文档主旨摘要（summarize_document）。二者均为 LLM 增强、失败不阻塞
    入库（只留日志）。供 insert_documents_structured 与 parse_step 复用，避免重复。
    """
    # 章节摘要（检索铺垫；开关控制 + 失败不阻塞）
    try:
        summary_cfg = config.get("summary", {})
        summary_enabled = bool(summary_cfg.get("enabled", True))
        summary_concurrency = int(summary_cfg.get("concurrency", 4))
        from summarizer import summarize_tree
        n = summarize_tree(root, enabled=summary_enabled, max_workers=summary_concurrency)
        if n == 0 and summary_enabled:
            print(f"[WARN] 章节摘要生成 0 条（可能模型未配置或全部失败）", flush=True)
        else:
            print(f"[OK] 章节摘要生成 {n} 条", flush=True)
    except Exception as e:
        # 摘要生成失败不阻塞入库，但必须留日志（修复：原 except:pass 静默吞异常）
        print(f"[WARN] 章节摘要生成失败（不阻塞入库）: {type(e).__name__}: {e}", flush=True)

    # 文档级主旨摘要（跨文档路由索引；失败不阻塞）
    abstract = ""
    try:
        from summarizer import summarize_document
        abstract = summarize_document(root)
        if not abstract:
            print(f"[WARN] 文档主旨摘要生成失败（返回空，不阻塞入库）", flush=True)
    except Exception as e:
        print(f"[WARN] 文档主旨摘要生成失败（不阻塞入库）: {type(e).__name__}: {e}", flush=True)
    return abstract


def insert_documents(text: str, source: str = "", on_progress=None) -> dict:
    """
    文本 → 父子块切分 → 向量化 → 入库。

    Args:
        text:        全文内容
        source:      来源文件名
        on_progress: 可选进度回调 on_progress(stage, progress)，
                     stage 为 CHUNKING/EMBEDDING/INDEXING。

    Returns:
        {"parent_chunks": N, "child_chunks": N, "inserted_count": N}
    """
    emb = get_embedding()
    store = get_store()

    _emit(on_progress, "CHUNKING", 0)
    result = emb.embed_with_parent_child(
        text, source=source,
        on_batch=lambda done, total: _emit(on_progress, "EMBEDDING", int(done / total * 100)),
    )
    parents = result.get("parent_chunks", [])
    children = result.get("child_chunks", [])

    if not children:
        return {
            "parent_chunks": len(parents),
            "child_chunks": 0,
            "inserted_count": 0,
        }

    _emit(on_progress, "INDEXING", 0)
    total = 0
    if hasattr(store, "insert_parent_child") and store._use_pc:
        _, c_count = store.insert_parent_child(source, parents, children)
        total += c_count
    else:
        texts, vectors = [], []
        for c in children:
            if "vector" not in c:
                continue
            texts.append(c["text"])
            vectors.append(c["vector"])
        if texts:
            store.insert(texts, vectors)
        total += len(texts)

    return {
        "parent_chunks": len(parents),
        "child_chunks": len(children),
        "inserted_count": total,
    }


def insert_documents_structured(filepath: str, source: str = "", on_progress=None) -> dict:
    """
    结构化入库（增强解析路径，步骤 3/4）：

      文件 → 结构归位（Structure Tree）→ SQLite 存树
           → 结构树切分（Retrieval Chunk）→ 向量化 → Milvus 存 chunk

    Args:
        filepath:    已落盘的文档路径（临时文件）
        source:      来源文件名（用于展示 / 删除）
        on_progress: 可选进度回调 on_progress(stage, progress)，
                     stage 为 PARSING/CHUNKING/EMBEDDING/INDEXING。

    Returns:
        {"doc_id": ..., "section_count": N, "chunk_count": N, "inserted_count": N}
    """
    import structure_resolver
    import chunk_builder
    import tree_store

    # 1. 结构归位
    _emit(on_progress, "PARSING", 0)
    root = structure_resolver.build_document_tree(filepath, on_progress=on_progress)

    # 1.5/1.6 章节摘要 + 文档主旨摘要（LLM 增强，失败不阻塞入库）
    abstract = _generate_summaries(root)

    # 2. 结构树 → chunk（纯内存操作，失败时尚未写任何库）
    _emit(on_progress, "CHUNKING", 0)
    chunks = chunk_builder.build_chunks(root)
    if not chunks:
        # 无检索单元时仍保存文档树（结构存在，仅无可检索 chunk）
        doc_id = tree_store.save_tree(root, source=source, abstract=abstract)
        return {"doc_id": doc_id, "section_count": _count_sections(root), "chunk_count": 0, "inserted_count": 0}

    # 3. 向量化 chunk 文本（最昂贵、最易失败的步骤，置于所有写库操作之前；
    #    失败时无任何残留，天然保持 SQLite 树库与 Milvus 一致）
    emb = get_embedding()
    texts = [c["text"] for c in chunks]
    _emit(on_progress, "EMBEDDING", 0)
    vectors = emb.embed_texts(
        texts,
        on_batch=lambda done, total: _emit(on_progress, "EMBEDDING", int(done / total * 100)),
    )
    for i, vec in enumerate(vectors):
        chunks[i]["vector"] = vec

    # 4. 存树到 SQLite（含 abstract）
    _emit(on_progress, "INDEXING", 0)
    doc_id = tree_store.save_tree(root, source=source, abstract=abstract)

    # 5. 写入 Milvus（只存 chunk）；失败则回滚已写入的树库，
    #    避免「树库有文档但 Milvus 无 chunk」的不一致
    store = get_store()
    try:
        inserted = store.insert_chunks(source, chunks) if hasattr(store, "insert_chunks") else 0
    except Exception:
        try:
            tree_store.delete_document(doc_id)
        except Exception:
            pass
        raise

    return {
        "doc_id": doc_id,
        "section_count": _count_sections(root),
        "chunk_count": len(chunks),
        "inserted_count": inserted,
    }


def _count_sections(root) -> int:
    """统计结构树中的 section 节点数。"""
    n = 0
    for c in root.children:
        if c.type == "section":
            n += 1 + _count_sections(c)
    return n


def ingest_file(filepath: str, source: str, enhance: bool, on_progress=None) -> dict:
    """
    异步入库统一入口（供 ingest_queue Worker 调用）。

    根据 enhance 分支：
      - enhance=True  ：结构归位 + 摘要 + 树切分 + 向量化（增强解析）
      - enhance=False ：文本提取 + 父子块切分 + 向量化（普通解析）

    入库前先做同名去重（delete_source），避免重复入库。

    Args:
        filepath:    已落盘的文件路径
        source:      文件名
        enhance:     是否增强解析
        on_progress: 可选进度回调 on_progress(stage, progress)

    Returns:
        insert_documents / insert_documents_structured 的返回值 dict。
    """
    # 去重：先删除同 source 的旧数据（树库 + Milvus）
    delete_source(source)

    if enhance:
        return insert_documents_structured(filepath, source, on_progress=on_progress)

    from embedding import _read_file_content
    text = _read_file_content(filepath)
    if not text or not text.strip():
        raise ValueError(f"文件内容为空，无法入库: {source}")
    return insert_documents(text, source, on_progress=on_progress)


# ---------------------------------------------------------------------------
# 步骤化入库（Phase 3）：parse → chunk → embed → index 四步拆解
# ---------------------------------------------------------------------------
# 供 Celery 双队列链式编排使用：每步产物可 JSON 序列化落盘，支持步骤级重跑
# （某步失败只重跑该步，前置步骤产物仍在，幂等跳过）。
# 与一体化 ingest_file 的语义完全一致，只是把「一次调用」拆成「四次调用」。
# ---------------------------------------------------------------------------

def parse_step(filepath: str, source: str, enhance: bool, on_progress=None) -> dict:
    """步骤 1（PARSING）：解析文件，返回可 JSON 序列化的 parse_result。

    Returns:
        enhance=True  : {"enhance": True, "root": {...}, "abstract": "..."}
        enhance=False : {"enhance": False, "text": "..."}
    """
    _emit(on_progress, "PARSING", 0)
    if enhance:
        import structure_resolver
        root = structure_resolver.build_document_tree(filepath, on_progress=on_progress)
        abstract = _generate_summaries(root)
        return {"enhance": True, "root": root.to_dict(), "abstract": abstract}

    from embedding import _read_file_content
    text = _read_file_content(filepath)
    if not text or not text.strip():
        raise ValueError(f"文件内容为空，无法入库: {source}")
    return {"enhance": False, "text": text}


def chunk_step(parse_result: dict, source: str, enhance: bool, on_progress=None) -> dict:
    """步骤 2（CHUNKING）：把 parse_result 切分为 chunk，返回 chunk_result。

    Returns:
        enhance=True  : {"enhance": True, "root": {...}, "abstract": "...", "chunks": [...]}
        enhance=False : {"enhance": False, "parent_chunks": [...], "child_chunks": [...]}
    """
    _emit(on_progress, "CHUNKING", 0)
    if enhance:
        from structure_resolver import TreeNode
        import chunk_builder
        root = TreeNode.from_dict(parse_result["root"])
        chunks = chunk_builder.build_chunks(root)
        return {
            "enhance": True,
            "root": parse_result["root"],
            "abstract": parse_result.get("abstract", ""),
            "chunks": chunks,
        }

    # 普通解析：纯切分（不向量化），切分与向量化解耦
    emb = get_embedding()
    pc = emb.chunker.chunk(parse_result["text"], source=source)
    return {
        "enhance": False,
        "parent_chunks": pc["parent_chunks"],
        "child_chunks": pc["child_chunks"],
    }


def embed_step(chunk_result: dict, enhance: bool, on_progress=None) -> dict:
    """步骤 3（EMBEDDING）：给 chunk_result 里的 chunk 就地回写 vector。

    原地修改并返回 chunk_result（增强路径写 chunks[].vector，普通路径写
    child_chunks[].vector），供 index_step 直接消费。
    """
    _emit(on_progress, "EMBEDDING", 0)
    emb = get_embedding()

    def _progress(done, total):
        _emit(on_progress, "EMBEDDING", int(done / total * 100))

    if enhance:
        chunks = chunk_result.get("chunks", [])
        if chunks:
            texts = [c["text"] for c in chunks]
            vectors = emb.embed_texts(texts, on_batch=_progress)
            for i, vec in enumerate(vectors):
                chunks[i]["vector"] = vec
    else:
        children = chunk_result.get("child_chunks", [])
        if children:
            texts = [c["text"] for c in children]
            vectors = emb.embed_texts(texts, on_batch=_progress)
            for i, vec in enumerate(vectors):
                children[i]["vector"] = vec
    return chunk_result


def index_step(chunk_result: dict, source: str, enhance: bool, on_progress=None) -> dict:
    """步骤 4（INDEXING）：把带 vector 的 chunk_result 写入树库 + Milvus。

    Returns:
        enhance=True  : {"doc_id", "section_count", "chunk_count", "inserted_count"}
        enhance=False : {"parent_chunks", "child_chunks", "inserted_count"}
    """
    _emit(on_progress, "INDEXING", 0)
    if enhance:
        from structure_resolver import TreeNode
        import tree_store
        root = TreeNode.from_dict(chunk_result["root"])
        abstract = chunk_result.get("abstract", "")
        chunks = chunk_result.get("chunks", [])

        # 无检索单元时仍保存文档树（结构存在，仅无可检索 chunk）
        if not chunks:
            doc_id = tree_store.save_tree(root, source=source, abstract=abstract)
            return {"doc_id": doc_id, "section_count": _count_sections(root),
                    "chunk_count": 0, "inserted_count": 0}

        # 先存树，再写 Milvus；Milvus 失败则回滚树库，保证二者一致
        doc_id = tree_store.save_tree(root, source=source, abstract=abstract)
        store = get_store()
        try:
            inserted = store.insert_chunks(source, chunks) if hasattr(store, "insert_chunks") else 0
        except Exception:
            try:
                tree_store.delete_document(doc_id)
            except Exception:
                pass
            raise
        return {"doc_id": doc_id, "section_count": _count_sections(root),
                "chunk_count": len(chunks), "inserted_count": inserted}

    # 普通解析
    parents = chunk_result.get("parent_chunks", [])
    children = chunk_result.get("child_chunks", [])
    if not children:
        return {"parent_chunks": len(parents), "child_chunks": 0, "inserted_count": 0}

    store = get_store()
    if hasattr(store, "insert_parent_child") and store._use_pc:
        _, c_count = store.insert_parent_child(source, parents, children)
        total = c_count
    else:
        texts, vectors = [], []
        for c in children:
            if "vector" not in c:
                continue
            texts.append(c["text"])
            vectors.append(c["vector"])
        if texts:
            store.insert(texts, vectors)
        total = len(texts)
    return {"parent_chunks": len(parents), "child_chunks": len(children), "inserted_count": total}


def match_document_sections(query: str, limit: int = 5) -> list:
    """结构匹配（步骤 4 树侧能力）：query 关键词命中章节标题。"""
    try:
        import tree_store
        return tree_store.match_sections(query, limit=limit)
    except Exception:
        return []


def search_by_section(query: str, section_path: list, doc_id: str, top_k: int = 5) -> list:
    """章节定位型检索（Tree 独立入口）：向量检索 + section_path 前缀过滤。

    Args:
        query:        用户问题（用于向量化）
        section_path: 章节路径 List[int]（来自 match_sections）
        doc_id:       稳定文档 ID
        top_k:        返回数量
    """
    emb = get_embedding()
    store = get_store()
    query_vector = emb.embed_text(query)
    results = store.search_chunks_by_section(query_vector, doc_id, section_path, top_k=top_k) or []

    # 上下文补全：章节路径恢复
    try:
        import tree_store
        for r in results:
            if r.get("doc_id"):
                path = tree_store.get_section_path_titles(r["doc_id"], r.get("parent_id", ""))
                r["section_path_titles"] = path
                r["section_path_str"] = " > ".join(path) if path else ""
    except Exception:
        pass
    return results


def get_toc_text() -> str:
    """返回所有已入库文档的目录结构文本（目录型问题用）。"""
    try:
        import tree_store
        structures = tree_store.list_document_structure()
        if not structures:
            return ""
        parts = []
        for s in structures:
            title = s["doc_title"] or s.get("source") or s["doc_id"][:12]
            parts.append(f"【文档】{title}")
            for sec in s["sections"]:
                indent = "  " * sec["level"]
                parts.append(f"{indent}{'#' * (sec['level'] + 1)} {sec['title']}")
        return "\n".join(parts)
    except Exception:
        return ""


def retrieve_by_section_entry(query: str, top_k: int = 5) -> dict:
    """Tree 独立入口统一封装（章节定位型）。

    流程：match_sections 定位章节 → search_by_section 过滤召回 chunk。
    定位失败返回 {"docs": [], "matched": None}，由上层兜底回退普通检索。
    """
    matched = match_document_sections(query, limit=1)
    if not matched:
        return {"docs": [], "matched": None}

    m = matched[0]
    docs = search_by_section(query, m["section_path"], m["doc_id"], top_k=top_k)
    return {"docs": docs, "matched": m}


def read_neighbor_chunks(doc_id: str, section_path: str, chunk_seq: int, window: int = 2) -> list:
    """读取同一 section 内相邻 chunk（Agentic READ_PARENT / READ_SECTION 用）。

    对已有 evidence 做上下文扩展：返回其前后相邻 chunk（不跨章节）。
    """
    store = get_store()
    try:
        return store.get_neighbor_chunks(doc_id, section_path, chunk_seq, window=window) or []
    except Exception:
        return []


def list_documents() -> list:
    """列出已入库的文档树（SQLite）。"""
    try:
        import tree_store
        return tree_store.list_documents()
    except Exception:
        return []


def delete_document(doc_id: str) -> bool:
    """删除指定文档树（SQLite）及其 Milvus chunk。"""
    try:
        import tree_store
        tree_store.delete_document(doc_id)
    except Exception:
        pass
    # 删除 Milvus 中该 doc_id 的 chunk
    store = get_store()
    try:
        store.client.delete(store.child_col_name, filter=f'doc_id == "{doc_id}"')
    except Exception:
        pass
    return True


def list_parents() -> list:
    """列出所有父块 + 结构树文档（虚拟父块，增强解析入库）。

    数据库管理页面统一展示两类数据：
      1. 普通父子块（parents collection）
      2. 结构树文档（SQLite 树库 + children 里的结构树 chunk）
    """
    store = get_store()
    try:
        parents = store.list_all_parents()
    except AttributeError:
        parents = store.list_parents(limit=1000)
    parents = list(parents or [])

    # 合并结构树文档（虚拟父块，parent_id 用 "tree:" 前缀标识）
    try:
        import tree_store
        tree_docs = tree_store.list_documents()
        if tree_docs:
            chunk_counts = store.count_chunks_by_doc_id() if hasattr(store, "count_chunks_by_doc_id") else {}
            for d in tree_docs:
                title = d.get("title") or d.get("source") or d["doc_id"][:12]
                parents.append({
                    "parent_id": f"tree:{d['doc_id']}",
                    "source": d.get("source") or "（结构树文档）",
                    "parent_index": 0,
                    "text_preview": title[:100],
                    "child_count": chunk_counts.get(d["doc_id"], 0),
                    "is_tree_doc": True,
                    "doc_id": d["doc_id"],
                })
    except Exception:
        pass  # 树库不可用不影响父块列表

    return parents


def list_children(parent_id: str) -> list:
    """列出指定父块 / 结构树文档下的所有子块。"""
    store = get_store()
    if parent_id.startswith("tree:"):
        doc_id = parent_id[len("tree:"):]
        return store.list_chunks_by_doc_id(doc_id)
    return store.list_children(parent_id)


def get_document_tree(doc_id: str) -> dict:
    """返回结构树文档的完整树形结构（章节层级 + chunk 挂载）。

    把 SQLite 树节点与 Milvus chunk（通过 parent_id == node_id 关联）合并，
    返回嵌套 dict，供前端按「文档 → 章节 → 小节 → 段落/表格 → chunk」展示。

    Returns:
        {
          "node_id", "type", "title", "text_preview", "level",
          "chunk_count": 直接挂载的 chunk 数,
          "subtree_chunk_count": 子树总 chunk 数,
          "chunks": [{"id", "text_preview", "chunk_seq"}, ...],
          "children": [...]
        }
        未找到返回 None。
    """
    try:
        import tree_store
        root = tree_store.load_tree(doc_id)
    except Exception:
        root = None
    if root is None:
        return None

    store = get_store()
    chunks = store.list_chunks_by_doc_id(doc_id) if hasattr(store, "list_chunks_by_doc_id") else []

    # chunk 按 parent_id（== 树节点 node_id）分组
    by_parent = defaultdict(list)
    for c in chunks:
        by_parent[c.get("parent_id", "")].append(c)

    def build(node) -> dict:
        node_chunks = sorted(
            by_parent.get(node.node_id, []),
            key=lambda x: x.get("parent_index", 0),
        )
        children = [build(c) for c in node.children]
        subtree = len(node_chunks) + sum(c["subtree_chunk_count"] for c in children)
        return {
            "node_id": node.node_id,
            "doc_id": node.doc_id,
            "type": node.type,
            "title": node.title,
            "summary": node.summary or "",
            "text_preview": (node.text or "")[:120],
            "level": node.level,
            "chunk_count": len(node_chunks),
            "subtree_chunk_count": subtree,
            "chunks": [
                {
                    "id": c.get("id"),
                    "text_preview": c.get("text_preview") or (c.get("text", "") or "")[:120],
                    "chunk_seq": c.get("parent_index", 0),
                }
                for c in node_chunks
            ],
            "children": children,
        }

    return build(root)


def delete_parent(parent_id: str) -> bool:
    """删除父块及其子块（结构树文档走 delete_document）。"""
    if parent_id.startswith("tree:"):
        doc_id = parent_id[len("tree:"):]
        return delete_document(doc_id)
    store = get_store()
    store.delete_parent(parent_id)
    return True


def delete_child(child_id: int) -> bool:
    """删除单个子块。"""
    store = get_store()
    store.delete_child(child_id)
    return True


def delete_source(source: str) -> bool:
    """删除指定文件下的所有分块（兼容结构树文档）。"""
    store = get_store()
    # 若 source 对应结构树文档，删除树 + chunk
    try:
        import tree_store
        docs = tree_store.list_documents()
        for d in docs:
            if d.get("source") == source:
                delete_document(d["doc_id"])
                return True
    except Exception:
        pass
    store.delete_source(source)
    return True


def rename_source(old_source: str, new_source: str) -> bool:
    """重命名文件（source），只改文件名，不改任何内容。

    覆盖两类数据：
      1. 结构树文档：SQLite documents 表的 source 字段
      2. 普通父子块 + 结构树 chunk：Milvus parents/children 的 source 字段

    Args:
        old_source: 原文件名
        new_source: 新文件名
    """
    if not old_source or not new_source or old_source == new_source:
        return False

    # 1) 结构树文档（SQLite documents 表）
    try:
        import tree_store
        tree_store.rename_document_source(old_source, new_source)
    except Exception:
        pass  # 树库不可用不影响 Milvus 侧重命名

    # 2) Milvus parents/children（普通父子块 + 结构树 chunk）
    store = get_store()
    if hasattr(store, "rename_source"):
        store.rename_source(old_source, new_source)
    return True


def clear_all() -> int:
    """清空全部数据（含结构树），返回删除前条数。"""
    store = get_store()
    before = store.count()
    store.delete_all()
    # 同时清空 SQLite 树库
    try:
        import tree_store
        for d in tree_store.list_documents():
            tree_store.delete_document(d["doc_id"])
    except Exception:
        pass
    return before


def count() -> dict:
    """统计：子块数 + 父块数。"""
    store = get_store()
    return {
        "count": store.count(),
        "parent_count": store.parent_count() if hasattr(store, "parent_count") else 0,
    }


def list_databases() -> list:
    """列出本地 Milvus 的所有 database。"""
    db = _current_db()
    uri = db.get("url") if db else LOCAL_MILVUS_URL
    return list_local_databases(uri)


def create_database(db_name: str) -> bool:
    """在本地 Milvus 新建 database。"""
    db = _current_db()
    uri = db.get("url") if db else LOCAL_MILVUS_URL
    return create_local_database(db_name, uri)
