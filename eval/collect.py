# -*- coding: utf-8 -*-
"""
组合遍历采集：对每个组合 × 每个评测题，跑通 pipeline，采集 answer + contexts。

输出：data/runs/{combo_id}.json
  [{"question_id", "question", "ground_truth", "answer", "contexts", "error"}, ...]

运行（主 poetry 环境）：
  poetry run python eval/collect.py                 # 全量（31 组合 × 20 题）
  poetry run python eval/collect.py --combo rag_hybrid_rw0_tc0_ws0   # 只跑一个组合
  poetry run python eval/collect.py --limit 3       # 每个组合只跑前 3 题
  poetry run python eval/collect.py --refresh       # 忽略已有结果，重跑

断点续跑：默认开启。已完成的 (combo, question) 会跳过。
"""

import argparse
import json
import os
import sys
import time
import traceback

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.dirname(EVAL_DIR))

import config


# ============================================================================
# 组件初始化（复用 server 的懒加载单例）
# ============================================================================
def init_components():
    from server import get_reranker, get_llm, get_tool_llm, get_rewrite_llm, _retrieve
    reranker = get_reranker()
    return {
        "reranker": reranker,
        "llm": get_llm(),
        "tool_llm": get_tool_llm(),
        "rewrite_llm": get_rewrite_llm(),
        # 检索函数：与 server 一致，走本地 db_service.search_documents（含 vector/hybrid/tree 三态）
        "retriever": lambda query, top_k, mode="hybrid": _retrieve(query, top_k, mode),
    }


# ============================================================================
# 组合配置开关（通过 config_loader 写 config，采集结束后恢复）
# ============================================================================
def set_combo_config(combo):
    from config_loader import set_config
    if combo.get("tool_calling") is not None:
        set_config("mcp.tool_calling.enabled", bool(combo["tool_calling"]))
    if combo.get("websearch") is not None:
        set_config("mcp.features.websearch", bool(combo["websearch"]))


def snapshot_originals():
    from config_loader import cfg
    return {
        "tool_calling": cfg("mcp.tool_calling.enabled"),
        "websearch": cfg("mcp.features.websearch"),
    }


def restore_config(originals):
    from config_loader import set_config
    if originals.get("tool_calling") is not None:
        set_config("mcp.tool_calling.enabled", originals["tool_calling"])
    if originals.get("websearch") is not None:
        set_config("mcp.features.websearch", originals["websearch"])


# ============================================================================
# 运行单题
# ============================================================================
def run_rag(question, mode, retrieval_mode, use_rewrite, comp):
    """rag / direct 模式（通过 RAGGraph）。"""
    from rag_graph import RAGGraph
    graph = RAGGraph(
        reranker=comp["reranker"],
        llm=comp["llm"],
        retriever=comp["retriever"],
        tool_llm=comp["tool_llm"],
        rewrite_llm=comp["rewrite_llm"],
        use_rewrite=use_rewrite,
        retrieval_mode=retrieval_mode,
    )
    state = graph.run(question, mode=mode)
    answer = state.get("generation", "") or ""
    docs = state.get("filtered") or state.get("reranked") or state.get("documents", [])
    contexts = [d.get("text", "") for d in docs if d.get("text")]
    return answer, contexts


def run_agentic(question, retrieval_mode, comp):
    from agentic_rag import run_agentic
    result = run_agentic(
        question,
        reranker=comp["reranker"],
        retrieval_mode=retrieval_mode,
    )
    answer = result.get("answer", "") or ""
    contexts = [c.get("text", "") for c in result.get("citations", []) if c.get("text")]
    return answer, contexts


def run_one(combo, q, comp):
    """运行单个 (组合, 题) 对，返回 (answer, contexts)。"""
    mode = combo["mode"]
    if mode == "direct":
        return run_rag(q["question"], mode="direct", retrieval_mode=None, use_rewrite=None, comp=comp)
    if mode == "rag":
        return run_rag(
            q["question"], mode="rag",
            retrieval_mode=combo["retrieval_mode"],
            use_rewrite=combo["rewrite"],
            comp=comp,
        )
    if mode == "agentic":
        return run_agentic(q["question"], retrieval_mode=combo["retrieval_mode"], comp=comp)
    raise ValueError(f"未知模式: {mode}")


# ============================================================================
# 结果读写（断点续跑）
# ============================================================================
def load_runs(combo_id):
    path = os.path.join(config.RUNS_DIR, f"{combo_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_runs(combo_id, rows):
    path = os.path.join(config.RUNS_DIR, f"{combo_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ============================================================================
# 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, choices=["all", "mode-comparison"], help="预定义子集")
    ap.add_argument("--combo", default=None, help="只跑指定组合 id（默认全部）")
    ap.add_argument("--limit", type=int, default=None, help="每个组合最多跑多少题")
    ap.add_argument("--refresh", action="store_true", help="忽略已有结果，全部重跑")
    ap.add_argument("--retry-empty", action="store_true", help="重跑检索结果为空（ctx=0 且无 error）的题，其余已完成的跳过")
    args = ap.parse_args()

    config.ensure_dirs()
    with open(config.DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]

    combos = config.get_subset(args.subset)
    if args.combo:
        combos = [c for c in combos if c["id"] == args.combo]
    if not combos:
        print(f"[错误] 未找到组合: subset={args.subset} combo={args.combo}")
        return

    print(f"评测集: {len(dataset)} 题 | 组合: {len(combos)} 个 | 总计 {len(dataset) * len(combos)} 次运行")
    print("初始化组件（LLM / Reranker）...")
    comp = init_components()

    originals = snapshot_originals()
    total_done = 0
    total_err = 0
    try:
        for ci, combo in enumerate(combos, 1):
            print(f"\n===== [{ci}/{len(combos)}] {combo['id']} =====")
            set_combo_config(combo)

            rows = [] if args.refresh else load_runs(combo["id"])
            if args.retry_empty:
                # 过滤掉「无 error 但 contexts 为空」的题（检索失败），使其重跑；
                # 有 error 的题也一并过滤重跑（网络抖动导致的失败，非数据问题）。
                rows = [r for r in rows if r.get("error") or len(r.get("contexts") or []) > 0]
            done_ids = {r.get("question_id") for r in rows}

            for qi, q in enumerate(dataset, 1):
                qid = q["id"]
                if qid in done_ids:
                    continue
                t0 = time.time()
                try:
                    answer, contexts = run_one(combo, q, comp)
                    rows.append({
                        "question_id": qid,
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "answer": answer,
                        "contexts": contexts,
                        "error": "",
                    })
                    save_runs(combo["id"], rows)  # 每题落盘，崩溃可续
                    total_done += 1
                    dt = time.time() - t0
                    print(f"  [{qid}] {dt:.1f}s ctx={len(contexts)}")
                except Exception as e:
                    total_err += 1
                    rows.append({
                        "question_id": qid,
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "answer": "",
                        "contexts": [],
                        "error": f"{type(e).__name__}: {e}",
                    })
                    save_runs(combo["id"], rows)
                    print(f"  [{qid}] 失败: {e}")

            print(f"  → 组合完成，累计成功 {total_done}，失败 {total_err}")
    finally:
        restore_config(originals)

    print(f"\n[完成] 成功 {total_done}，失败 {total_err}，结果在 {config.RUNS_DIR}")


if __name__ == "__main__":
    main()
