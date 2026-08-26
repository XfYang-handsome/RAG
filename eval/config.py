# -*- coding: utf-8 -*-
"""
RAG 评测共享配置（不依赖主项目任何模块，纯数据 + 纯 JSON，两个环境通用）。

包含：
  1. 组合矩阵（31 种 = 1 direct + 24 rag + 6 agentic）
  2. 评测指标清单
  3. 模型配置读取（judge LLM / embeddings，从 config/models.json 读）
  4. 路径常量
"""

import itertools
import json
import os

# ============================================================================
# 路径
# ============================================================================
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(EVAL_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
SCORES_DIR = os.path.join(DATA_DIR, "scores")
REPORT_PATH = os.path.join(DATA_DIR, "report.md")
MODELS_PATH = os.path.join(PROJECT_ROOT, "config", "models.json")


def ensure_dirs():
    for d in (DATA_DIR, RUNS_DIR, SCORES_DIR):
        os.makedirs(d, exist_ok=True)


# ============================================================================
# 指标清单（与用户关注点对齐）
# ============================================================================
# 需要 ground_truth 的指标：ContextRecall / ContextPrecision / AnswerCorrectness
# 不需要 ground_truth：Faithfulness / AnswerRelevancy
METRICS = [
    "faithfulness",          # 忠实度 → 幻觉程度
    "context_recall",        # 上下文召回率 → 漏检
    "context_precision",     # 上下文精度 → 噪声/排序
    "answer_relevancy",      # 答案相关性 → 答非所问
    "answer_correctness",    # 答案正确性 → 端到端对错
]

# ============================================================================
# 组合矩阵
# ============================================================================
# 各维度对不同模式的作用范围（与 rag_graph / agentic 实际实现一致）：
#   - direct    : 无检索、无工具、无联网
#   - rag       : retrieval_mode × rewrite × tool_calling × websearch
#   - agentic   : retrieval_mode × websearch
#                 （agentic 内部 query 重写始终启用；无独立 tool_calling 开关）
# ============================================================================

_RETRIEVAL_MODES = ["vector", "hybrid", "tree"]


def build_combinations():
    """生成全部组合，返回 [ {id, mode, retrieval_mode, rewrite, tool_calling, websearch}, ... ]"""
    combos = []

    # direct
    combos.append({
        "id": "direct",
        "mode": "direct",
        "retrieval_mode": None,
        "rewrite": None,
        "tool_calling": None,
        "websearch": None,
    })

    # rag
    for rm, rw, tc, ws in itertools.product(
        _RETRIEVAL_MODES, [False, True], [False, True], [False, True]
    ):
        combos.append({
            "id": f"rag_{rm}_rw{int(rw)}_tc{int(tc)}_ws{int(ws)}",
            "mode": "rag",
            "retrieval_mode": rm,
            "rewrite": rw,
            "tool_calling": tc,
            "websearch": ws,
        })

    # agentic
    for rm, ws in itertools.product(_RETRIEVAL_MODES, [False, True]):
        combos.append({
            "id": f"agentic_{rm}_ws{int(ws)}",
            "mode": "agentic",
            "retrieval_mode": rm,
            "rewrite": None,
            "tool_calling": None,
            "websearch": ws,
        })

    return combos


COMBINATIONS = build_combinations()


# ============================================================================
# 预定义评测子集
# ============================================================================
# 「检索模式对比」：只变 retrieval_mode，关闭 rewrite/tool_calling/websearch，
# 纯净对比 vector / hybrid / tree 三种检索模式对召回/精度的真实影响
# （避免查询改写、工具调用、联网补救机制干扰对比）。
MODE_COMPARISON_IDS = [
    "rag_vector_rw0_tc0_ws0",
    "rag_hybrid_rw0_tc0_ws0",
    "rag_tree_rw0_tc0_ws0",
    "agentic_vector_ws0",
    "agentic_hybrid_ws0",
    "agentic_tree_ws0",
]


def get_subset(name):
    """按名称返回组合子集。name 为 None / 'all' 返回全量。"""
    if name in (None, "all"):
        return COMBINATIONS
    if name == "mode-comparison":
        return [c for c in COMBINATIONS if c["id"] in MODE_COMPARISON_IDS]
    raise ValueError(f"未知子集: {name}")


# ============================================================================
# 模型配置读取（judge LLM / embeddings）
# ============================================================================
def _load_models():
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(kind: str) -> dict:
    """按 kind 取当前选中模型；优先 current，否则第一个。"""
    models = _load_models()
    arr = models.get(kind) or []
    if not arr:
        return {}
    current = (models.get("current") or {}).get(kind)
    for m in arr:
        if m.get("name") == current:
            return m
    return arr[0]


# judge 专用模型：非 reasoning 模型，避免 thinking 导致结构化输出超时/不完整。
# （DeepSeek V4 Pro 是 reasoning 模型，thinking 无法禁用，faithfulness/correctness 会超时）
JUDGE_MODEL = "deepseek-ai/DeepSeek-V3.2"


def judge_llm_config() -> dict:
    """评分用的 judge LLM 配置（非 reasoning，走 llm 的 base_url/api_key）。"""
    cfg = dict(_pick("llm"))
    cfg["model"] = JUDGE_MODEL
    return cfg


def judge_embeddings_config() -> dict:
    """评分用的 embeddings 配置（BGE-M3）。"""
    return _pick("embedding")


if __name__ == "__main__":
    print(f"组合总数: {len(COMBINATIONS)}")
    for c in COMBINATIONS:
        print(c["id"])
