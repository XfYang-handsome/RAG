"""
================================================================================
Milvus 向量数据库封装
================================================================================

两种存储模式：
  ┌─────────────────────────────────────────────────────────────────┐
  │ 本地模式（use_parent_child=True）                                │
  │   父 Collection（{name}_parents）: 粗粒度语义段落 + 粗检索向量    │
  │   子 Collection（{name}_children）: 细粒度语义单元 + 精搜向量     │
  │   关系: 子块通过 parent_id (UUID) 关联父块                        │
  │                                                                  │
  │   检索流程: query向量 → 父Collection COSINE → 命中父块            │
  │           → 在命中父块的子块中 COSINE 精搜 → 合并去重排序          │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │ 云端模式（use_parent_child=False）                                │
  │   单 Collection: 兼容原有 Milvus 格式                             │
  │   检索流程: query向量 → 单次 COSINE 搜索                          │
  └─────────────────────────────────────────────────────────────────┘

语义聚类插入（本地模式特有）：
  新导入文件时，子块向量与已有父块向量做 COSINE 相似度匹配。
  相似度 >= threshold → 归入已有父块（语义聚合）
  相似度 <  threshold → 创建新父块

技术栈：
  - pymilvus 3.x: MilvusClient API（推荐用法）
  - Milvus standalone: Docker 部署，默认端口 19530
  - 索引类型: IVF_FLAT（适合中小规模，nlist=128）
  - 相似度度量: COSINE
================================================================================
"""

import os, math, uuid
from typing import List, Optional
from pymilvus import (
    MilvusClient, DataType,
    Function, FunctionType,
    AnnSearchRequest, RRFRanker,
)

# ============================================================================
# 基础设施常量（具体数据库连接信息由 db.json 提供，这里仅作兜底默认值）
# ============================================================================

# --- 本地 Milvus 默认地址（Docker standalone）---
LOCAL_MILVUS_URL    = "http://localhost:19530"
LOCAL_MILVUS_TOKEN  = ""   # Docker standalone 无需 token
LOCAL_MILVUS_DB_NAME = ""  # 空=默认库

# --- 通用常量 ---
LOCAL_COLLECTION_NAME = "local_rag_docs"  # 本地基础集合名
INSERT_BATCH_SIZE     = 200               # 批量插入每批条数
DEFAULT_VECTOR_DIM    = 1024              # 默认向量维度（BGE-M3）

# Collection 命名约定
_PARENT_COL_SUFFIX = "_parents"   # 本地父 collection 后缀
_CHILD_COL_SUFFIX  = "_children"  # 本地子 collection 后缀

# 混合检索 RRF 融合参数（归一化分数用）
_RRF_K         = 60   # RRF 的 k 值（倒数排名平滑项）
_RRF_NUM_LISTS = 2    # 融合路数（dense + sparse）


# ============================================================================
# 工厂函数
# ============================================================================

def create_store(
    data_mode: str = "cloud",
    db_path: str = None,
    collection_name: str = None,
    dim: int = 1024,
    db_name: str = None,
    token: str = None,
) -> "MilvusStore":
    """
    创建 MilvusStore 实例。

    Args:
        data_mode: "local" 或 "cloud"
          - local: 本地 Milvus，启用父子双 Collection
          - cloud: 远程 Milvus，单 Collection（兼容原有格式）
        db_path: Milvus 服务地址（覆盖默认配置）
        collection_name: 集合基础名称
        dim: 向量维度（需与 embedding 模型输出维度一致）
        db_name: 数据库名（覆盖默认配置）
        token: 认证 token（覆盖默认配置）

    Returns:
        配置好的 MilvusStore 实例
    """
    if data_mode == "local":
        uri = db_path or LOCAL_MILVUS_URL
        return MilvusStore(
            collection_name=collection_name or LOCAL_COLLECTION_NAME,
            dim=dim, uri=uri,
            token=token or LOCAL_MILVUS_TOKEN,
            db_name=db_name or LOCAL_MILVUS_DB_NAME,
            use_parent_child=True,    # ← 本地模式启用父子双 Collection
        )
    else:
        return MilvusStore(
            collection_name=collection_name or "aigc_docs_bge",
            dim=dim, uri=db_path,
            token=token,
            db_name=db_name,
            use_parent_child=False,   # ← 云端模式用单 Collection
        )


def create_store_from_config(db_config: dict, collection_name: str, dim: int = 1024) -> "MilvusStore":
    """
    根据数据库配置字典（来自 db.json）创建 MilvusStore 实例。

    db_config 字段：
      - type: "local" / "online"
      - url: Milvus 服务地址
      - token: 认证 token（可选）
      - db_name: 数据库名（可选）

    Args:
        db_config: 数据库配置字典
        collection_name: 集合基础名称
        dim: 向量维度
    """
    is_local = db_config.get("type") == "local"
    return create_store(
        data_mode="local" if is_local else "cloud",
        db_path=db_config.get("url"),
        collection_name=collection_name,
        dim=dim,
        db_name=db_config.get("db_name"),
        token=db_config.get("token"),
    )


def is_local_uri(uri: str) -> bool:
    """判断 URI 是否为本地地址"""
    return "localhost" in uri or "127.0.0.1" in uri


def list_local_databases(uri: str = None) -> List[str]:
    """
    列出本地 Milvus 服务中已存在的所有 database。

    Args:
        uri: Milvus 服务地址（默认 localhost:19530）

    Returns:
        database 名称列表
    """
    client = None
    try:
        client = MilvusClient(uri=uri or LOCAL_MILVUS_URL)
        dbs = client.list_databases()
        return dbs or []
    except Exception as e:
        print(f"  [WARN] 列出本地 database 失败: {e}")
        return []
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def create_local_database(db_name: str, uri: str = None) -> bool:
    """
    在本地 Milvus 服务中新建一个 database。

    Args:
        db_name: database 名称
        uri: Milvus 服务地址

    Returns:
        是否创建成功
    """
    client = None
    try:
        client = MilvusClient(uri=uri or LOCAL_MILVUS_URL)
        client.create_database(db_name)
        return True
    except Exception as e:
        print(f"  [ERROR] 创建 database 失败: {e}")
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


# ============================================================================
# Collection Schema 定义
# ============================================================================

def _create_parent_schema(client, dim):
    """
    父 Collection 的 Schema。

    父块的设计理念：
      - 不用于检索，只存完整原文
      - 当子块被向量检索命中后，通过 parent_id 找到父块
      - 父块的完整上下文送给 LLM 生成回答

    Milvus 要求 collection 必须至少有一个 vector 字段，所以保留 vector 字段
    但不建索引（has_vector=False），只做存储。
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",           DataType.INT64, is_primary=True)
    schema.add_field("parent_id",    DataType.VARCHAR, max_length=64)
    schema.add_field("source",       DataType.VARCHAR, max_length=512)
    schema.add_field("parent_index", DataType.INT64)
    schema.add_field("text",         DataType.VARCHAR, max_length=65535)
    schema.add_field("vector",       DataType.FLOAT_VECTOR, dim=dim)  # Milvus 要求必须有 vector 字段
    schema.add_field("child_count",  DataType.INT64)
    return schema


def _create_child_schema(client, dim):
    """
    子 Collection 的 Schema。

    字段说明：
      id           : 自增主键
      parent_id    : 关联的父块业务 ID / 结构树节点 ID（parent_node_id）
      source       : 来源文件名
      parent_index : 所属父块索引
      text         : 子块文本（精搜单元）
      vector       : 子块向量（精搜用）
      doc_id       : 稳定文档 ID（结构树场景，普通父子块为空串）
      section_path : 章节路径 JSON（如 "[0, 1]"，结构树场景，普通父子块为空串）
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",           DataType.INT64, is_primary=True)
    schema.add_field("parent_id",    DataType.VARCHAR, max_length=64)
    schema.add_field("source",       DataType.VARCHAR, max_length=512)
    schema.add_field("parent_index", DataType.INT64)
    schema.add_field("text",         DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("vector",       DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("doc_id",       DataType.VARCHAR, max_length=64, default_value="")
    schema.add_field("section_path", DataType.VARCHAR, max_length=512, default_value="")
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # BM25 全文检索：由 text 字段自动生成 sparse_vector（稀疏向量），用于混合检索
    schema.add_function(Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
    ))
    return schema


def _create_legacy_schema(client, dim):
    """
    旧版单 Collection Schema（云端兼容）。

    仅包含 id / text / vector 三个字段。
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",     DataType.INT64, is_primary=True)
    schema.add_field("text",   DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    return schema


def _init_one_collection(client, name, dim, schema_fn, has_vector=True):
    """
    初始化单个 Collection。

    逻辑：
      - Collection 已存在 → load 到内存
      - Collection 不存在 → 创建 → (如有 vector 字段则建索引) → load

    父 Collection 不含 vector 字段（has_vector=False），不建索引。
    """
    if client.has_collection(name):
        try:
            client.load_collection(name)
        except Exception as e:
            print(f"  [WARN] load_collection 失败 ({name}): {e}")
        return

    # 新建 collection
    schema = schema_fn(client, dim)
    client.create_collection(collection_name=name, schema=schema)

    # 只有包含 vector 字段的 collection 才建向量索引
    if has_vector:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(name)


# ============================================================================
# MilvusStore — 核心存储类
# ============================================================================

class MilvusStore:
    """
    Milvus 向量存储封装。

    属性：
      collection_name  : 基础集合名（兼容旧代码）
      parent_col_name  : 父 Collection 名（{base_name}_parents）
      child_col_name   : 子 Collection 名（{base_name}_children）

    本地模式核心方法：
      insert_parent_child()  : 语义聚类插入（父子块）
      search()               : 两阶段检索（父粗搜 → 子精搜）
      list_parents()         : 列出所有父块
      list_children()        : 列出指定父块的子块
      delete_parent()        : 删除父块+子块
      delete_child()         : 删除单个子块
      delete_all()           : 清空全部数据

    云端模式核心方法：
      insert()               : 单 Collection 插入
      search()               : 单次检索
    """

    def __init__(
        self,
        collection_name: str = None,
        dim: int = None,
        uri: str = None,
        token: str = None,
        db_name: str = None,
        use_parent_child: bool = False,
    ):
        """
        Args:
            collection_name:  集合基础名称
            dim:              向量维度（默认 1024，与 BGE-M3 一致）
            uri:              Milvus 服务地址
            token:            认证 token（云端需要）
            db_name:          数据库名（云端需要）
            use_parent_child: True=父子双Collection模式, False=单Collection模式
        """
        self._base_name = collection_name or LOCAL_COLLECTION_NAME
        self.dim        = dim or DEFAULT_VECTOR_DIM
        self._use_pc    = use_parent_child  # 是否启用父子模式
        self._is_local  = is_local_uri(uri or LOCAL_MILVUS_URL)

        # 连接 Milvus
        kwargs = {"uri": uri or LOCAL_MILVUS_URL}
        if token:   kwargs["token"]   = token
        if db_name: kwargs["db_name"] = db_name

        self.client = MilvusClient(**kwargs)
        self._init_collections()

    # ==================================================================
    # 属性
    # ==================================================================
    @property
    def collection_name(self):
        """兼容旧代码：返回基础集合名"""
        return self._base_name

    @property
    def parent_col_name(self):
        """父 Collection 完整名称"""
        return self._base_name + _PARENT_COL_SUFFIX

    @property
    def child_col_name(self):
        """子 Collection 完整名称"""
        return self._base_name + _CHILD_COL_SUFFIX

    # ==================================================================
    # 初始化
    # ==================================================================
    def _init_collections(self):
        """
        初始化所有需要的 Collection。

        父子模式：创建/加载 parent_col（无向量索引） + child_col（有向量索引）
        单Collection模式：创建/加载 基础 collection

        子 Collection 若已存在但缺少 doc_id/section_path 字段（旧 schema），
        自动 drop 重建以完成结构树元数据迁移。
        """
        if self._use_pc:
            _init_one_collection(self.client, self.parent_col_name, self.dim,
                                 _create_parent_schema, has_vector=True)  # 父块 vector 也必须建索引（load 要求）
            self._migrate_child_schema()
            self._init_child_collection()
        else:
            _init_one_collection(self.client, self._base_name, self.dim,
                                 _create_legacy_schema, has_vector=True)

    def _init_child_collection(self):
        """初始化子 Collection（dense 向量索引 + BM25 稀疏索引）。

        子 Collection 承载混合检索的两路召回：
          - vector（FLOAT_VECTOR）: COSINE 语义召回
          - sparse_vector（SPARSE_FLOAT_VECTOR）: BM25 关键词召回（由 text 自动生成）
        """
        name = self.child_col_name
        if self.client.has_collection(name):
            try:
                self.client.load_collection(name)
            except Exception as e:
                print(f"  [WARN] load_collection 失败 ({name}): {e}")
            return

        schema = _create_child_schema(self.client, self.dim)
        self.client.create_collection(collection_name=name, schema=schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            metric_type="BM25",                       # BM25 function 输出字段必须用 BM25 度量
            index_type="SPARSE_INVERTED_INDEX",
        )
        self.client.create_index(collection_name=name, index_params=index_params)
        self.client.load_collection(name)

    def _migrate_child_schema(self):
        """若子 Collection 存在但缺 doc_id / sparse_vector / BM25 function，drop 重建。

        混合检索需要子 Collection 具备：
          - doc_id / section_path（结构树元数据）
          - sparse_vector 字段 + BM25 function（关键词稀疏召回）
        旧版 schema 缺少任一即 drop（数据需重新入库，BM25 需对文本重新建索引）。
        """
        try:
            if not self.client.has_collection(self.child_col_name):
                return
            desc = self.client.describe_collection(self.child_col_name)
            fields = [f.get("name") for f in desc.get("fields", [])]
            funcs = desc.get("functions") or []
            missing = []
            if "doc_id" not in fields:
                missing.append("doc_id")
            if "section_path" not in fields:
                missing.append("section_path")
            if "sparse_vector" not in fields:
                missing.append("sparse_vector")
            if not funcs:
                missing.append("bm25 function")
            if missing:
                print(f"  [INFO] 检测到旧版子 Collection（缺 {', '.join(missing)}），重建 schema（数据需重新入库）...")
                self.client.drop_collection(self.child_col_name)
        except Exception as e:
            print(f"  [WARN] 子 Collection schema 迁移检测失败: {e}")

    # ==================================================================
    # 插入（父子模式）
    # ==================================================================
    def insert_parent_child(
        self,
        source: str,
        parents: List[dict],
        children: List[dict],
    ):
        """
        父子块插入。

        父块只存原文（不向量化），子块存向量 + 文本。
        子块通过 parent_id (UUID) 关联父块。

        流程：
          1. 遍历 parents 列表，每个父块使用其已有的 parent_id (UUID)
          2. 将父块原文插入父 Collection
          3. 将子块（含向量）插入子 Collection

        Args:
            source:  来源文件名
            parents: [{"index":0, "parent_id":"uuid", "text":"父块全文"}, ...]
            children:[{"index":0, "parent_id":"uuid", "text":"子块", "vector":[...]}, ...]

        Returns:
            (新增父块数, 插入子块数)
        """
        if not children:
            return 0, 0

        # ---- 插入父块（存原文 + 零向量占位） ----
        inserted_parents = 0
        parent_data = []
        for i, p in enumerate(parents):
            pid = p.get("parent_id") or str(uuid.uuid4().hex[:12])
            parent_data.append({
                "parent_id":    pid,
                "source":       source,
                "parent_index": p.get("index", i),
                "text":         p["text"],
                "vector":       [0.0] * self.dim,  # 零向量占位（不用做检索）
                "child_count":  0,
            })
        if parent_data:
            self._batch_insert(self.parent_col_name, parent_data)
            inserted_parents = len(parent_data)

        # ---- 插入子块（存向量 + parent_id 关联） ----
        child_data = []
        for c in children:
            if "vector" not in c:
                continue
            child_data.append({
                "parent_id":    c.get("parent_id", ""),
                "source":       source,
                "parent_index": c.get("parent_index", 0),
                "text":         c["text"],
                "vector":       c["vector"],
            })

        inserted_children = 0
        if child_data:
            self._batch_insert(self.child_col_name, child_data)
            inserted_children = len(child_data)

        self.client.flush(self.parent_col_name)
        self.client.flush(self.child_col_name)
        return inserted_parents, inserted_children

    # ==================================================================
    # 插入 — 结构树 chunk（步骤 3/4：Milvus 只存 Retrieval Chunk）
    # ==================================================================
    def insert_chunks(self, source: str, chunks: List[dict]):
        """
        插入结构树切分出的 Retrieval Chunk 到子 Collection。

        chunk 字段（来自 chunk_builder.build_chunks + 向量化）：
          text / vector / parent_node_id / source_node_ids / doc_id /
          section_path (List[int]) / chunk_seq

        映射到子 Collection：
          parent_id    = parent_node_id（结构树节点 ID）
          parent_index = chunk_seq（全局阅读顺序）
          doc_id       = 稳定文档 ID
          section_path = 章节路径 JSON 字符串
          source       = 来源文件名
        """
        if not chunks:
            return 0

        data = []
        for c in chunks:
            if "vector" not in c:
                continue
            data.append({
                "parent_id":    c.get("parent_node_id", ""),
                "source":       source,
                "parent_index": c.get("chunk_seq", 0),
                "text":         c["text"],
                "vector":       c["vector"],
                "doc_id":       c.get("doc_id", ""),
                # section_path 存 "/" 分隔的路径（如 "0/1"），便于标量前缀过滤
                "section_path": "/".join(str(x) for x in c.get("section_path", [])),
            })

        if data:
            self._batch_insert(self.child_col_name, data)
            self.client.flush(self.child_col_name)
        return len(data)

    # ==================================================================
    # 检索 — 章节过滤（Tree 独立入口，章节定位型问题）
    # ==================================================================
    def search_chunks_by_section(self, query_vector, doc_id: str, section_path, top_k: int = 5):
        """
        按章节过滤的向量检索：只召回 ``doc_id`` 下、章节路径以 ``section_path``
        为前缀的 chunk（含该章节直属 chunk 及其子章节 chunk）。

        Args:
            query_vector: 查询向量
            doc_id:       稳定文档 ID
            section_path: 章节路径 List[int]，如 [0, 1]（空列表 = 整文档）
            top_k:        返回数量
        """
        path_str = "/".join(str(x) for x in (section_path or []))
        if path_str:
            filter_expr = (
                f'doc_id == "{doc_id}" && '
                f'(section_path == "{path_str}" || section_path like "{path_str}/%")'
            )
        else:
            filter_expr = f'doc_id == "{doc_id}"'

        results = self.client.search(
            collection_name=self.child_col_name,
            data=[query_vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            filter=filter_expr,
            output_fields=["parent_id", "source", "parent_index", "text", "doc_id", "section_path"],
        )

        if not results or not results[0]:
            return []

        unique = []
        seen = set()
        for h in results[0]:
            entity = h["entity"]
            pid = entity.get("parent_id", "")
            if pid in seen:
                continue
            seen.add(pid)
            unique.append({
                "id":           h["id"],
                "text":         entity.get("text", ""),
                "score":        h["distance"],
                "parent_id":    pid,
                "doc_id":       entity.get("doc_id", ""),
                "section_path": entity.get("section_path", ""),
                "chunk_seq":    entity.get("parent_index", 0),
            })
        return unique

    def get_neighbor_chunks(self, doc_id: str, section_path: str, chunk_seq: int, window: int = 2):
        """邻近块扩展：返回同一 section 内、chunk_seq 相邻的 chunk。

        规则（与架构定稿一致）：
          - 同一 section_path（不跨章节）
          - parent_index 在 [chunk_seq - window, chunk_seq + window] 范围内
          - 排除自身（chunk_seq）

        Args:
            doc_id:       稳定文档 ID
            section_path: 章节路径字符串（"/" 分隔，如 "0/1"）
            chunk_seq:    命中 chunk 的全局阅读顺序
            window:       前后各扩展的块数

        Returns:
            [{"text", "chunk_seq", "parent_id", "doc_id", "section_path"}, ...]
            按 chunk_seq 升序。
        """
        lo = chunk_seq - window
        hi = chunk_seq + window
        # 同一 section 内、按阅读顺序相邻（排除自身）
        if section_path:
            filter_expr = (
                f'doc_id == "{doc_id}" && section_path == "{section_path}" && '
                f'parent_index >= {lo} && parent_index <= {hi} && parent_index != {chunk_seq}'
            )
        else:
            filter_expr = (
                f'doc_id == "{doc_id}" && '
                f'parent_index >= {lo} && parent_index <= {hi} && parent_index != {chunk_seq}'
            )

        try:
            rows = self.client.query(
                collection_name=self.child_col_name,
                filter=filter_expr,
                output_fields=["parent_id", "text", "parent_index", "doc_id", "section_path"],
                limit=window * 2 + 2,
            )
        except Exception:
            return []

        result = []
        for r in rows:
            result.append({
                "text":         r.get("text", ""),
                "chunk_seq":    r.get("parent_index", 0),
                "parent_id":    r.get("parent_id", ""),
                "doc_id":       r.get("doc_id", ""),
                "section_path": r.get("section_path", ""),
            })
        result.sort(key=lambda x: x["chunk_seq"])
        return result

    # ==================================================================
    # 插入 — 单 Collection（云端兼容）
    # ==================================================================
    def insert(self, texts: List[str], vectors: List[List[float]]):
        """
        单 Collection 批量插入（云端模式）。

        Args:
            texts:   文本列表
            vectors: 对应向量列表
        """
        if not texts:
            return
        col  = self._base_name
        data = [{"text": texts[i], "vector": vectors[i]} for i in range(len(texts))]
        self._batch_insert(col, data)
        self.client.flush(col)

    def _batch_insert(self, col_name, data):
        """
        通用批量插入（自动分批）。

        INSERT_BATCH_SIZE 控制每批条数，防止单次请求过大。
        """
        total = len(data)
        for start in range(0, total, INSERT_BATCH_SIZE):
            end = min(start + INSERT_BATCH_SIZE, total)
            self.client.insert(col_name, data[start:end])

    # ==================================================================
    # 检索
    # ==================================================================
    def search(self, query_vector, top_k=5, output_fields=None):
        """
        向量相似搜索。

        父子模式：
          直接在子 Collection 中搜索（子块向量）。
          命中后返回子块文本 + 父块完整原文（通过 parent_id 追溯）。
        """
        if self._use_pc:
            return self._search_with_parents(query_vector, top_k)
        else:
            return self._search_legacy(query_vector, top_k, output_fields)

    def search_hybrid(self, query_vector, query_text: str, top_k=5, limit_multiplier: int = 2):
        """
        混合检索：dense 向量召回 + BM25 稀疏召回，RRF 融合排序。

        两路召回：
          - dense  : query_vector 在 vector 字段做 COSINE 检索（语义）
          - sparse : query_text   通过 BM25 function 在 sparse_vector 字段检索（关键词）

        融合：RRF（Reciprocal Rank Fusion）对两路排名做倒数融合，兼顾语义与精确关键词。

        Args:
            query_vector:     查询向量（dense 路）
            query_text:       原始查询文本（sparse 路，BM25 分词）
            top_k:            最终返回数量
            limit_multiplier: 每路召回 top_k * multiplier 条候选再融合

        Returns:
            与 search() 相同的结构（经 _format_child_hits 统一处理）。
        """
        if not self._use_pc:
            # 单 Collection（云端）无 sparse 字段，回退纯 dense
            return self._search_legacy(query_vector, top_k)

        limit = max(top_k * limit_multiplier, top_k)
        req_dense = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
        )
        req_sparse = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=limit,
        )

        results = self.client.hybrid_search(
            collection_name=self.child_col_name,
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(k=_RRF_K),
            limit=limit,
            output_fields=["parent_id", "source", "parent_index", "text", "doc_id", "section_path"],
        )

        if not results or not results[0]:
            return []
        # 归一化 RRF 分数到 [0, 1]：
        # RRF 原始分数 = Σ 1/(k + rank)，量级约 0.016~0.033，与 COSINE 分数（0~1）
        # 不可比，会导致 grade 用 0.25 阈值误判「不相关」。
        # 理论最大值 = 路数 / k = _RRF_NUM_LISTS / _RRF_K，归一化后单路 top-1≈0.5、
        # 两路 top-1≈1.0，与 dense 分数同量级。
        norm = _RRF_K / _RRF_NUM_LISTS
        for hit in results[0]:
            hit["distance"] = float(hit.get("distance", 0.0) or 0.0) * norm
        return self._format_child_hits(results[0], top_k)

    def _search_with_parents(self, query_vector, top_k):
        """
        父子模式检索：搜子块 → 追溯父块原文（或结构树 chunk 直返）。

        子 Collection 中混合两类数据：
          1. 普通父子块（doc_id 为空）：命中后追溯父块原文
          2. 结构树 chunk（doc_id 非空）：命中后直接返回 chunk 文本 + 树元数据

        返回的 text 格式：
          普通父块:  "[source: xxx] [父块全文]\n\n[命中片段] 子块文本"
          结构树 chunk:  chunk 文本本身（章节路径由上层 tree_store 恢复）
        """
        # ---- Step 1: 子块检索 ----
        child_results = self.client.search(
            collection_name=self.child_col_name,
            data=[query_vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k * 2,  # 多取一些，去重后还能有 top_k
            output_fields=["parent_id", "source", "parent_index", "text", "doc_id", "section_path"],
        )

        if not child_results or not child_results[0]:
            return []

        # ---- Step 2: 追溯父块原文 / 结构树 chunk 直返（去重） ----
        return self._format_child_hits(child_results[0], top_k)

    def _format_child_hits(self, hits, top_k):
        """把子 Collection 命中结果统一格式化（dense / hybrid 两路复用）。

        命中结果两类：
          1. 普通父子块（doc_id 为空）：追溯父块原文
          2. 结构树 chunk（doc_id 非空）：直接返回 chunk 文本 + 树元数据
        """
        seen_keys = set()
        unique = []
        for h in hits:
            entity = h["entity"]
            doc_id = entity.get("doc_id", "")
            pid = entity.get("parent_id", "")

            # 去重键：结构树 chunk 按 parent_id 去重；普通父块也按 parent_id 去重
            if pid in seen_keys:
                continue
            seen_keys.add(pid)

            child_text = entity.get("text", "")

            if doc_id:
                # 结构树 chunk：直接返回，附带树元数据（章节路径由上层恢复）
                unique.append({
                    "id":           h["id"],
                    "text":         child_text,
                    "score":        h["distance"],
                    "parent_id":    pid,
                    "doc_id":       doc_id,
                    "section_path": entity.get("section_path", ""),
                    "chunk_seq":    entity.get("parent_index", 0),
                })
            else:
                # 普通父子块：追溯父块原文
                parent_text = self._get_parent_text(pid)
                full_text = child_text
                if parent_text:
                    full_text = f"[source: {entity.get('source','')}] {parent_text}\n\n[命中片段] {child_text}"
                unique.append({
                    "id":       h["id"],
                    "text":     full_text,
                    "score":    h["distance"],
                    "parent_id": pid,
                })

            if len(unique) >= top_k:
                break

        return unique

    def _get_parent_text(self, parent_id: str) -> str:
        """通过 parent_id 获取父块原文"""
        results = self.client.query(
            collection_name=self.parent_col_name,
            filter=f'parent_id == "{parent_id}"',
            output_fields=["text"],
            limit=1,
        )
        if results:
            return results[0].get("text", "")
        return ""

    def _search_legacy(self, query_vector, top_k, output_fields=None):
        """
        单 Collection 直接搜索（云端兼容）。

        一次 COSINE 搜索，返回 top_k 条结果。
        """
        if output_fields is None:
            output_fields = ["text"]
        results = self.client.search(
            collection_name=self._base_name,
            data=[query_vector],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=output_fields,
        )
        if not results or not results[0]:
            return []
        return [{
            "id":    h["id"],
            "text":  h["entity"].get("text"),
            "score": h["distance"],
        } for h in results[0]]

    # ==================================================================
    # 计数
    # ==================================================================
    def _count_visible(self, collection_name: str) -> int:
        """
        统计 collection 中「实际可见」的实体数。

        为什么不用 get_collection_stats().row_count：
          Milvus 的 delete 是软删除（标记 tombstone），row_count 在删除后
          **不会立即减少**（要等 compaction 才释放），导致删除单个文件/父块后
          数量统计不变；只有 drop+重建（clear_all）才会立即归零。
          query 的 count(*) 会过滤 tombstone，返回真实可见数量。
        """
        try:
            res = self.client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["count(*)"],
            )
            if res:
                return int(res[0].get("count(*)", 0))
        except Exception:
            pass
        # 回退：get_collection_stats（可能含软删除，但至少不报错）
        stats = self.client.get_collection_stats(collection_name)
        return stats.get("row_count", 0)

    def count(self):
        """
        获取数据条数（实际可见，排除软删除）。

        父子模式返回子块数（子块才是检索目标），
        单Collection模式返回该collection的行数。
        """
        if self._use_pc:
            return self._count_visible(self.child_col_name)
        return self._count_visible(self._base_name)

    def parent_count(self):
        """父块数量（实际可见，排除软删除）"""
        if self._use_pc:
            return self._count_visible(self.parent_col_name)
        return 0

    # ==================================================================
    # 管理：查询父块列表
    # ==================================================================
    # Milvus query 单次 limit+offset 上限为 16384，分批查询每批大小
    _QUERY_BATCH_SIZE = 16384

    def _query_all(self, collection_name: str, filter_expr: str, output_fields: List[str]):
        """
        分批查询 collection 的全部数据（突破 Milvus 单次 16384 条限制）。

        Args:
            collection_name: collection 名
            filter_expr:     过滤表达式
            output_fields:   需要返回的字段列表
        """
        all_results = []
        offset = 0
        while True:
            batch = self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=self._QUERY_BATCH_SIZE,
                offset=offset,
            )
            if not batch:
                break
            all_results.extend(batch)
            if len(batch) < self._QUERY_BATCH_SIZE:
                break
            offset += len(batch)
        return all_results

    def list_parents(self, limit=100, offset=0):
        """
        分页获取父块列表。

        每个父块附带：
          - child_count: 实际子块数（批量统计，避免 N+1 查询）
          - text_preview: 文本前 100 字符（前端预览用）

        Args:
            limit:  每页条数
            offset: 偏移量
        """
        if not self._use_pc:
            return []

        results = self.client.query(
            collection_name=self.parent_col_name,
            filter="id >= 0",
            output_fields=["id", "parent_id", "source", "parent_index", "text", "child_count"],
            limit=limit,
            offset=offset,
        )

        # ---- 批量统计子块数（分批查询，突破 16384 限制） ----
        try:
            from collections import Counter
            all_children = self._query_all(
                self.child_col_name, "id >= 0", ["parent_id"]
            )
            child_counts = Counter(c["parent_id"] for c in all_children)
        except Exception:
            child_counts = {}

        for r in results:
            r["child_count"]  = child_counts.get(r["parent_id"], 0)
            r["text_preview"] = (r.get("text") or "")[:100]

        return results

    def list_all_parents(self):
        """获取所有父块（不分页，分批查询突破 16384 限制）"""
        if not self._use_pc:
            return []

        results = self._query_all(
            self.parent_col_name,
            "id >= 0",
            ["id", "parent_id", "source", "parent_index", "text", "child_count"],
        )

        # 批量统计子块数（分批查询）
        try:
            from collections import Counter
            all_children = self._query_all(
                self.child_col_name, "id >= 0", ["parent_id"]
            )
            child_counts = Counter(c["parent_id"] for c in all_children)
        except Exception:
            child_counts = {}

        for r in results:
            r["child_count"]  = child_counts.get(r["parent_id"], 0)
            r["text_preview"] = (r.get("text") or "")[:100]

        return results

    def list_children(self, parent_id: str):
        """
        获取指定父块下的所有子块。

        Args:
            parent_id: 父块业务 ID (UUID)
        """
        if not self._use_pc:
            return []

        results = self._query_all(
            self.child_col_name,
            f'parent_id == "{parent_id}"',
            ["id", "parent_id", "source", "parent_index", "text"],
        )
        for r in results:
            r["text_preview"] = (r.get("text") or "")[:120]
        return results

    def list_chunks_by_doc_id(self, doc_id: str):
        """
        获取指定 doc_id 下的所有结构树 chunk（增强解析入库的数据）。

        Args:
            doc_id: 稳定文档 ID（如 doc_xxxx）
        """
        if not self._use_pc:
            return []

        results = self._query_all(
            self.child_col_name,
            f'doc_id == "{doc_id}"',
            ["id", "parent_id", "source", "parent_index", "text"],
        )
        for r in results:
            r["text_preview"] = (r.get("text") or "")[:120]
        return results

    def count_chunks_by_doc_id(self):
        """
        统计 children collection 里按 doc_id 分组的 chunk 数（仅结构树 chunk）。

        Returns:
            {doc_id: chunk_count}
        """
        if not self._use_pc:
            return {}

        try:
            from collections import Counter
            all_children = self._query_all(
                self.child_col_name,
                'doc_id != ""',
                ["doc_id"],
            )
            return Counter(c["doc_id"] for c in all_children)
        except Exception:
            return {}

    def delete_chunks_by_doc_id(self, doc_id: str):
        """删除指定 doc_id 下的所有结构树 chunk。"""
        if not self._use_pc:
            return
        self.client.delete(self.child_col_name, filter=f'doc_id == "{doc_id}"')

    # ==================================================================
    # 管理：删除
    # ==================================================================
    def delete_child(self, child_id: int):
        """
        删除单个子块。

        Args:
            child_id: Milvus 内部自增 ID（非 parent_id）
        """
        self.client.delete(self.child_col_name, ids=[child_id])

    def delete_parent(self, parent_id: str):
        """
        删除指定父块及其所有子块。

        先删子块（通过 parent_id 过滤），再删父块。

        Args:
            parent_id: 父块业务 ID (UUID)
        """
        self.client.delete(self.child_col_name,  filter=f'parent_id == "{parent_id}"')
        self.client.delete(self.parent_col_name, filter=f'parent_id == "{parent_id}"')

    def delete_source(self, source: str):
        """
        删除指定文件（source）下的所有父块和子块。

        先根据 source 找到该文件的所有父块 parent_id，删除对应子块，
        再删除父块。

        Args:
            source: 来源文件名
        """
        # 找到该 source 的所有父块（分批查询，突破 16384 限制）
        parents = self._query_all(
            self.parent_col_name,
            f'source == "{source}"',
            ["parent_id"],
        )
        parent_ids = [p["parent_id"] for p in parents]

        # 删除这些父块关联的所有子块
        for pid in parent_ids:
            self.client.delete(self.child_col_name, filter=f'parent_id == "{pid}"')

        # 删除父块
        self.client.delete(self.parent_col_name, filter=f'source == "{source}"')

    def rename_source(self, old_source: str, new_source: str) -> int:
        """
        重命名 source 字段（只改文件名，不改任何内容/向量）。

        Milvus 无「只更新单个标量字段」的原生 UPDATE，故用 query + upsert
        （按自增主键 id 覆盖整条记录）实现。同时处理 parents 与 children
        两个 collection 中 source == old_source 的记录。

        Args:
            old_source: 原文件名
            new_source: 新文件名

        Returns:
            更新的记录总数（parents + children）。
        """
        if not self._use_pc:
            return 0  # 单 collection 模式无 source 字段
        if old_source == new_source:
            return 0

        updated = 0
        # 1) parents collection（vector 为零向量占位，upsert 时重填即可）
        try:
            parents = self._query_all(
                self.parent_col_name,
                f'source == "{old_source}"',
                ["id", "parent_id", "parent_index", "text", "child_count"],
            )
            if parents:
                rows = [{
                    "id": p["id"],
                    "parent_id": p["parent_id"],
                    "source": new_source,
                    "parent_index": p["parent_index"],
                    "text": p["text"],
                    "vector": [0.0] * self.dim,
                    "child_count": p["child_count"],
                } for p in parents]
                self.client.upsert(self.parent_col_name, rows)
                updated += len(rows)
        except Exception as e:
            print(f"[rename_source] parents 更新失败: {e}")

        # 2) children collection（vector 为真实向量，需查询回来再 upsert）
        try:
            children = self._query_all(
                self.child_col_name,
                f'source == "{old_source}"',
                ["id", "parent_id", "parent_index", "text", "vector", "doc_id", "section_path"],
            )
            if children:
                rows = [{
                    "id": c["id"],
                    "parent_id": c["parent_id"],
                    "source": new_source,
                    "parent_index": c["parent_index"],
                    "text": c["text"],
                    "vector": c["vector"],
                    "doc_id": c.get("doc_id", ""),
                    "section_path": c.get("section_path", ""),
                } for c in children]
                self.client.upsert(self.child_col_name, rows)
                updated += len(rows)
        except Exception as e:
            print(f"[rename_source] children 更新失败: {e}")

        return updated

    def delete_by_ids(self, ids: List[int]):
        """根据 ID 列表删除（兼容旧接口）"""
        if self._use_pc:
            self.client.delete(self.child_col_name, ids=ids)
        else:
            self.client.delete(self._base_name, ids=ids)

    def delete_all(self):
        """
        清空所有数据。

        父子模式：drop 两个 Collection 后重建（比逐条删除快得多）
        单Collection模式：drop 后重建
        """
        if self._use_pc:
            self.client.drop_collection(self.parent_col_name)
            self.client.drop_collection(self.child_col_name)
            self._init_collections()
        else:
            self.client.drop_collection(self._base_name)
            _init_one_collection(self.client, self._base_name, self.dim, _create_legacy_schema)

    def drop(self):
        """直接删除 Collection（不重建）"""
        if self._use_pc:
            self.client.drop_collection(self.parent_col_name)
            self.client.drop_collection(self.child_col_name)
        else:
            self.client.drop_collection(self._base_name)

    def close(self):
        """关闭 Milvus 连接"""
        self.client.close()
