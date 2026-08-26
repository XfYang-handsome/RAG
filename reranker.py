"""
================================================================================
Reranker 重排序模型封装
================================================================================

两种模式：
  ┌─────────────────────────────────────────────────────────────────┐
  │ online=False（默认）→ 本地 HuggingFace Transformers               │
  │  模型: BAAI/bge-reranker-v2-m3（交叉编码器）                      │
  │  推理: GPU (CUDA) 优先，自动降级 CPU                              │
  │  首次运行自动从 HuggingFace Hub 下载模型（约 1.1GB）               │
  ├─────────────────────────────────────────────────────────────────┤
  │ online=True → HTTP API 远程调用                                   │
  │  调用 POST {base_url}/v1/rerank                                  │
  │  格式兼容 Jina AI / Cohere Rerank API                            │
  └─────────────────────────────────────────────────────────────────┘

Reranker 在 RAG 流程中的位置：
  用户Query → Embedding → Milvus 检索 Top 20 → Reranker 精排 Top 5 → LLM

为什么需要 Reranker？
  向量检索（COSINE）是无交互的单塔模型，只考虑 query 和 doc 各自的向量。
  Reranker 是交叉编码器（Cross-Encoder），将 query+doc 拼接后共同编码，
  能捕获更细粒度的语义交互，排序更准确。

  代价：交叉编码器比向量检索慢，所以只对初筛的 Top 20 做精排。

依赖：
  - transformers + torch: 本地推理
  - requests: 远程 API 调用
================================================================================
"""

import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# 设置环境变量（需在 transformers import 之前）
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests

# 本地 reranker 默认模型名（具体配置从 models.json 读取）
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ============================================================================
# LocalReranker — 本地 HuggingFace 推理
# ============================================================================

class LocalReranker:
    """
    本地 Reranker，基于 HuggingFace Transformers 交叉编码器。

    使用 AutoModelForSequenceClassification 加载 BGE Reranker 模型。
    query 和 document 拼接为 "[query] [SEP] [document]" 格式输入模型，
    模型输出一个相关性分数（logits）。

    为什么不用 FlagEmbedding？
      FlagEmbedding 的 FlagReranker 在较新的 torch+transformers 环境下
      会产生 TORCH_LIBRARY 命名空间冲突错误。原生 HuggingFace API 更稳定。
    """

    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
        device: str = None,
        use_fp16: bool = True,
    ):
        """
        Args:
            model_name: 模型名或路径（默认 BAAI/bge-reranker-v2-m3）
            cache_dir:  模型缓存目录（None=使用 HF 默认缓存）
            device:     推理设备（"cuda"/"cpu"，None=自动检测）
            use_fp16:   是否使用半精度（节省显存，仅 GPU 有效）
        """
        self._model_name = model_name or DEFAULT_RERANKER_MODEL
        self._cache_dir  = cache_dir
        self._use_fp16   = use_fp16

        # 延迟加载（首次调用 rerank() 时才加载模型）
        self._tokenizer = None
        self._hf_model  = None
        self._device    = device  # 延迟到加载时再检测

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self):
        """
        延迟加载模型（首次调用时触发）。

        加载流程：
          1. 从 HuggingFace Hub 下载/加载 tokenizer
          2. 下载/加载模型权重（AutoModelForSequenceClassification）
          3. 移动到指定设备（GPU/CPU）
          4. 设为 eval 模式（关闭 dropout 等训练行为）
        """
        if self._hf_model is not None:
            return  # 已加载（实际模型挂在 _hf_model；_model 是未使用的死字段）

        import warnings
        warnings.filterwarnings("ignore")

        # 延迟 import torch + transformers（避免模块加载阶段的循环导入）
        import torch

        # 关键修复：torch 2.7.1 的 torch._library.utils.get_source 使用
        # inspect.getframeinfo 获取源码位置，在 transformers 深层 lazy import
        # 时（torchvision._meta_registrations 的 register_fake）会因 inspect 无法
        # 找到 frame 源码而抛 "'function' object has no attribute 'endswith'"。
        # 这里 patch 该函数，使其容错返回一个占位路径。
        try:
            import torch._library.utils as _tlu
            _orig_get_source = _tlu.get_source

            def _safe_get_source(stacklevel=1):
                try:
                    return _orig_get_source(stacklevel + 1)
                except Exception:
                    return "unknown_source"

            _tlu.get_source = _safe_get_source
            # 同时 patch 直接引用点
            import torch.library as _tlib
            _tlib._utils.get_source = _safe_get_source
        except Exception:
            pass

        # 使用具体的 XLMRobertaForSequenceClassification 类（bge-reranker 架构），
        # 避免 AutoModel 的全局模型扫描间接 import torchvision 导致循环导入。
        from transformers import XLMRobertaForSequenceClassification, AutoTokenizer

        # 设备检测：显式指定 > CUDA > CPU
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  [Reranker] 加载模型 {self._model_name} → {self._device}...")
        # 优先 local_files_only（本地缓存直接加载，避免联网检查更新超时），
        # 本地无缓存时回退到允许联网下载（如配置了 HF 镜像源）。
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, cache_dir=self._cache_dir, local_files_only=True
            )
            self._hf_model = XLMRobertaForSequenceClassification.from_pretrained(
                self._model_name, cache_dir=self._cache_dir, local_files_only=True
            )
        except Exception as e:
            print(f"  [Reranker] 本地缓存未命中（{e}），尝试联网加载...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, cache_dir=self._cache_dir
            )
            self._hf_model = XLMRobertaForSequenceClassification.from_pretrained(
                self._model_name, cache_dir=self._cache_dir
            )
        self._hf_model.to(self._device)
        self._hf_model.eval()  # 推理模式
        print(f"  [Reranker] 模型加载完成")

    # ------------------------------------------------------------------
    # 模型状态 / 显式加载
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载到内存"""
        return self._hf_model is not None

    def load(self):
        """显式加载模型（下载权重 + 加载到内存），供前端按需触发。"""
        self._load_model()

    # ------------------------------------------------------------------
    # Rerank 主方法
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
    ) -> List[dict]:
        """
        对文档列表按与 query 的相关性重新排序。

        Args:
            query:     用户查询文本
            documents: 待重排序的文档列表（通常来自 Milvus 初筛）
            top_n:     返回前 N 个结果（None=全部）

        Returns:
            [{"index": 原始索引, "text": 文档内容, "score": 相关性分数}, ...]
            按 score 降序排列
        """
        self._load_model()
        return self._rerank_with_hf(query, documents, top_n)

    def _rerank_with_hf(
        self, query: str, documents: List[str], top_n: int = None
    ) -> List[dict]:
        """
        使用 HuggingFace Transformers 交叉编码器进行 rerank。

        工作原理：
          1. 将 query + document 拼接为 "[query] [SEP] [document]"
          2. 批量送入模型，得到每个 pair 的相关性分数（logits）
          3. 按分数降序排序，取 top_n

        输入格式: BGE Reranker 使用 [SEP] 作为 query-doc 分隔符
        """
        import torch

        # ---- Step 1: 构造 query-doc pairs ----
        pairs = [f"{query} [SEP] {doc}" for doc in documents]

        # ---- Step 2: Tokenize + 推理 ----
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,          # BGE Reranker 最大输入长度
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():        # 关闭梯度计算，节省显存
            outputs = self._hf_model(**inputs)
            logits = outputs.logits.squeeze(-1)
            # sigmoid 归一化：BGE reranker 输出的是未归一化 logits（约 -10~10），
            # 映射到 [0,1] 的相关概率，使 score 语义与 grade_relevance_threshold 阈值一致
            scores  = torch.sigmoid(logits).cpu().tolist()

        # 单文档时 scores 是 float 而非 list，统一处理
        if isinstance(scores, float):
            scores = [scores]

        # ---- Step 3: 构造结果 + 排序 ----
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "index": idx,
                "text":  documents[idx],
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]

        return results


# ============================================================================
# Reranker — 统一入口
# ============================================================================

class Reranker:
    """
    Reranker 统一入口。

    根据 online 参数自动选择后端：
      - online=False（默认）→ LocalReranker（本地 HF 推理）
      - online=True        → HTTP API 远程调用

    使用方式：
      reranker = Reranker(online=False)
      results  = reranker.rerank("什么是RAG?", docs, top_n=5)
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        local_model_path: str = None,
        online: bool = None,
    ):
        """
        Args:
            model:            在线模型名称
            base_url:         在线 API 地址
            api_key:          在线 API 密钥
            local_model_path: 离线模型本地路径
            online:           True=在线, False=离线, None=根据参数自动判断
        """
        # 自动判断在线/离线：显式指定 > 根据 local_model_path 是否存在判断
        if online is None:
            online = not bool(local_model_path)

        self.online = online
        self._requests = requests  # online 模式 HTTP 调用用

        if online:
            # --- 远程 API 模式 ---
            self.model    = model
            self.base_url = (base_url or "").rstrip("/")
            self.api_key  = api_key
            self._local_reranker = None
        else:
            # --- 本地推理模式 ---
            self.model    = None
            self.base_url = None
            self.api_key  = None
            self._local_reranker = LocalReranker(
                model_name=local_model_path or DEFAULT_RERANKER_MODEL
            )

    @property
    def is_loaded(self) -> bool:
        """本地模型是否已加载到内存（在线模式恒为 True）"""
        if self.online:
            return True
        return self._local_reranker is not None and self._local_reranker.is_loaded

    def load(self):
        """
        显式加载本地模型（下载权重 + 加载到内存）。

        在线模式无需加载（直接调用 HTTP API），本地模式触发 LocalReranker.load()。
        """
        if not self.online and self._local_reranker is not None:
            self._local_reranker.load()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
    ) -> List[dict]:
        """
        对文档列表按相关性重新排序。

        Args:
            query:     用户查询
            documents: 待排序文档列表
            top_n:     返回前 N 个（None=全部）

        Returns:
            [{"index": 原始索引, "text": 文档内容, "score": 相关性分数}, ...]
        """
        if not self.online and self._local_reranker is not None:
            return self._local_reranker.rerank(query, documents, top_n)

        # --- online 模式：HTTP API 调用 ---
        # 兼容 Jina AI / Cohere Rerank API 格式
        url     = f"{self.base_url}/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":      self.model,
            "query":      query,
            "documents":  documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        response = self._requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            results.append({
                "index": idx,
                "text":  documents[idx],
                "score": item["relevance_score"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
