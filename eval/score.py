# -*- coding: utf-8 -*-
"""
Ragas 评分：对每个组合的采集结果跑 5 个指标。

输出：data/scores/{combo_id}.json
  {"combo_id", "mode", "metric_names", "per_question": [...], "avg": {...}}

运行（eval 独立 venv）：
  eval/.venv/Scripts/python eval/score.py                 # 全部组合
  eval/.venv/Scripts/python eval/score.py --combo rag_hybrid_rw0_tc0_ws0
  eval/.venv/Scripts/python eval/score.py --limit 3       # 每组合只评分前 3 题（冒烟）

说明：
  - 用旧版指标（ragas.metrics._xxx），因为 ragas 0.4.3 的 evaluate 只接受旧 Metric 基类。
  - judge LLM 用 llm_factory + extra_body 禁用 thinking（DeepSeek V4 Pro 加速约 5 倍）。
"""

import argparse
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

import config


def build_llm():
    from openai import OpenAI
    from ragas.llms import llm_factory
    cfg = config.judge_llm_config()
    client = OpenAI(api_key=cfg.get("api_key"), base_url=cfg.get("base_url"))
    return llm_factory(
        cfg.get("model"),
        client=client,
        max_tokens=4096,  # 默认 1024 会截断 faithfulness/correctness 的复杂 JSON 输出
    )


def build_embeddings():
    from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    cfg = config.judge_embeddings_config()
    emb = LCOpenAIEmbeddings(
        model=cfg.get("model"),
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
    )
    return LangchainEmbeddingsWrapper(emb)


def build_metrics(llm, emb):
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._answer_correctness import AnswerCorrectness
    return [
        Faithfulness(llm=llm),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        AnswerCorrectness(llm=llm, embeddings=emb),
    ]


def score_combo(combo_id, metrics, limit=None):
    from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig

    runs_path = os.path.join(config.RUNS_DIR, f"{combo_id}.json")
    if not os.path.exists(runs_path):
        print(f"  [跳过] 无采集结果: {runs_path}")
        return None

    with open(runs_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    valid = [r for r in rows if not r.get("error")]
    if limit:
        valid = valid[:limit]
    if not valid:
        print(f"  [跳过] 无有效结果（全部失败）")
        return None

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            reference=r["ground_truth"],
            retrieved_contexts=r.get("contexts") or [],
        )
        for r in valid
    ]

    metric_names = [m.name for m in metrics]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=600, max_retries=2),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    per_question = []
    for i, r in enumerate(valid):
        row = {"question_id": r["question_id"]}
        for name in metric_names:
            if name in df.columns:
                v = df.iloc[i][name]
                row[name] = round(float(v), 4) if v is not None else None
        per_question.append(row)

    avg = {}
    for name in metric_names:
        vals = [p[name] for p in per_question if p.get(name) is not None]
        avg[name] = round(sum(vals) / len(vals), 4) if vals else None

    return {
        "combo_id": combo_id,
        "mode": combo_id.split("_")[0],
        "metric_names": metric_names,
        "per_question": per_question,
        "avg": avg,
        "n_questions": len(valid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, choices=["all", "mode-comparison"])
    ap.add_argument("--combo", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    print("构建 judge LLM（禁用 thinking）+ embeddings ...")
    llm = build_llm()
    emb = build_embeddings()
    metrics = build_metrics(llm, emb)

    combos = config.get_subset(args.subset)
    if args.combo:
        combos = [c for c in combos if c["id"] == args.combo]
    for ci, combo in enumerate(combos, 1):
        print(f"\n===== [{ci}/{len(combos)}] 评分 {combo['id']} =====")
        out = score_combo(combo["id"], metrics, limit=args.limit)
        if out is None:
            continue
        out_path = os.path.join(config.SCORES_DIR, f"{combo['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  → 平均分: {out['avg']}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
