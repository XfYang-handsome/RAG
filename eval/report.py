# -*- coding: utf-8 -*-
"""
汇总报告：聚合所有组合的评分结果，生成 Markdown 对比报告。

运行（任意环境）：
  python eval/report.py

输出：data/report.md
"""

import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

import config


METRIC_LABELS = {
    "faithfulness": "忠实度 Faithfulness",
    "context_recall": "召回率 Context Recall",
    "context_precision": "精度 Context Precision",
    "answer_relevancy": "答案相关性 Answer Relevancy",
    "answer_correctness": "正确性 Answer Correctness",
}


def load_scores():
    """加载所有 scores/*.json，返回 {combo_id: score_obj}。"""
    scores = {}
    if not os.path.isdir(config.SCORES_DIR):
        return scores
    for fn in sorted(os.listdir(config.SCORES_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(config.SCORES_DIR, fn), "r", encoding="utf-8") as f:
            obj = json.load(f)
        scores[obj["combo_id"]] = obj
    return scores


def fmt(v):
    return f"{v:.3f}" if v is not None else "  -  "


def render_matrix(scores, combos, metrics):
    """生成 组合 × 指标 的 Markdown 表格。"""
    header = "| 组合 | " + " | ".join(METRIC_LABELS[m] for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines = [header, sep]
    for c in combos:
        obj = scores.get(c["id"])
        if not obj:
            continue
        avg = obj.get("avg", {})
        cells = [fmt(avg.get(m)) for m in metrics]
        lines.append(f"| {c['id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def group_by_dim(scores, combos, dim, metrics):
    """按某维度取值分组，计算各指标的组均值。返回 {value: {metric: mean}}。"""
    groups = {}
    for c in combos:
        val = c.get(dim)
        if val is None:
            continue
        obj = scores.get(c["id"])
        if not obj:
            continue
        avg = obj.get("avg", {})
        key = str(val)
        g = groups.setdefault(key, {m: [] for m in metrics})
        for m in metrics:
            if avg.get(m) is not None:
                g[m].append(avg[m])
    result = {}
    for key, g in groups.items():
        result[key] = {
            m: round(sum(v) / len(v), 4) if v else None for m, v in g.items()
        }
    return result


def render_dim_table(dim, groups, metrics, label_map=None):
    lines = []
    header = f"| {dim} | " + " | ".join(METRIC_LABELS[m] for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines.append(header)
    lines.append(sep)
    for key in sorted(groups.keys()):
        label = label_map.get(key, key) if label_map else key
        cells = [fmt(groups[key].get(m)) for m in metrics]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    config.ensure_dirs()
    scores = load_scores()
    if not scores:
        print("[错误] 无评分结果，请先运行 score.py")
        return

    metrics = config.METRICS
    done_ids = set(scores.keys())
    combos = [c for c in config.COMBINATIONS if c["id"] in done_ids]

    lines = []
    lines.append("# RAG 系统评测报告")
    lines.append("")
    lines.append(f"- 评测集：{config.DATASET_PATH}")
    lines.append(f"- 已评分组合：{len(combos)} / {len(config.COMBINATIONS)}")
    lines.append(f"- 指标：忠实度（幻觉）、召回率、精度、答案相关性、正确性")
    lines.append("")
    lines.append("> 说明：分数区间 0~1，越高越好。忠实度低 = 幻觉严重；召回率低 = 检索漏检；精度低 = 检索噪声/排序差。")
    lines.append("")

    # 按模式分组
    for mode in ["direct", "rag", "agentic"]:
        sub = [c for c in combos if c["mode"] == mode]
        if not sub:
            continue
        lines.append(f"## {mode} 模式")
        lines.append("")
        lines.append(render_matrix(scores, sub, metrics))
        lines.append("")

    # 维度对比（rag）
    rag_combos = [c for c in combos if c["mode"] == "rag"]
    if rag_combos:
        lines.append("## 维度对比（rag 模式）")
        lines.append("")
        dims = [
            ("retrieval_mode", {"vector": "vector", "hybrid": "hybrid", "tree": "tree"}),
            ("rewrite", {"False": "rewrite 关", "True": "rewrite 开"}),
            ("tool_calling", {"False": "工具调用 关", "True": "工具调用 开"}),
            ("websearch", {"False": "联网 关", "True": "联网 开"}),
        ]
        for dim, label_map in dims:
            groups = group_by_dim(scores, rag_combos, dim, metrics)
            if not groups:
                continue
            lines.append(f"### 按 {dim} 分组")
            lines.append("")
            lines.append(render_dim_table(dim, groups, metrics, label_map))
            lines.append("")

    # agentic 维度对比
    agentic_combos = [c for c in combos if c["mode"] == "agentic"]
    if agentic_combos:
        lines.append("## 维度对比（agentic 模式）")
        lines.append("")
        for dim, label_map in [("retrieval_mode", {"vector": "vector", "hybrid": "hybrid", "tree": "tree"}),
                               ("websearch", {"False": "联网 关", "True": "联网 开"})]:
            groups = group_by_dim(scores, agentic_combos, dim, metrics)
            if not groups:
                continue
            lines.append(f"### 按 {dim} 分组")
            lines.append("")
            lines.append(render_dim_table(dim, groups, metrics, label_map))
            lines.append("")

    report = "\n".join(lines)
    with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[完成] 报告已写入 {config.REPORT_PATH}")


if __name__ == "__main__":
    main()
