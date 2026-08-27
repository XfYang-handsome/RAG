# -*- coding: utf-8 -*-
"""
完整评测编排脚本：一键从头到尾跑完评测流程。

流程：前置检查 → 清空旧产物 → (可选)生成评测集 → 采集 → 评分 → 报告

运行（主 poetry 环境，因为采集/报告需要主环境依赖）：
  poetry run python eval/run_full.py                          # 从头完整跑 mode-comparison
  poetry run python eval/run_full.py --regenerate             # 重新生成评测集后再跑
  poetry run python eval/run_full.py --keep                   # 保留已有结果，断点续跑（自动补空结果）
  poetry run python eval/run_full.py --combo rag_hybrid_rw0_tc0_ws0   # 只跑单个组合
  poetry run python eval/run_full.py --skip-collect           # 已有 runs，只评分+报告
  poetry run python eval/run_full.py --skip-score             # 只采集，不评分
  poetry run python eval/run_full.py --subset all             # 全量 31 组合（耗时约 20h，慎用）

环境分工：
  - 采集（collect.py）：主 poetry 环境（复用 server/rag_graph 检索 + 生成）
  - 评分（score.py）  ：eval 独立 venv（ragas + openai 1.x）
  - 报告（report.py） ：主环境（纯读 json，无重依赖）

断点续跑：
  - 采集每题落盘，中断后重跑 `--keep` 会自动跳过已完成题、补跑 ctx=0 的空结果题；
  - 评分按组合落盘，中断后重跑 `--keep` 会跳过已评分的组合。
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATA_DIR = os.path.join(EVAL_DIR, "data")

# 评分用独立 venv 的 python（ragas 环境，与主环境 openai 版本冲突）
if platform.system() == "Windows":
    VENV_PYTHON = os.path.join(EVAL_DIR, ".venv", "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(EVAL_DIR, ".venv", "bin", "python")


# ============================================================================
# 工具函数
# ============================================================================
def _step(title: str):
    print("\n" + "=" * 72)
    print("▶ " + title)
    print("=" * 72, flush=True)


def _run(cmd: list, title: str) -> bool:
    """运行一个子命令，实时转发输出；返回是否成功。"""
    _step(title)
    print(f"命令: {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    try:
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print(f"\n[中断] 用户手动停止「{title}」。已落盘的进度会保留，可用 --keep 续跑。")
        return False
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"\n[失败] 「{title}」退出码 {ret.returncode}（耗时 {dt/60:.1f} 分钟）")
        return False
    print(f"\n[完成] 「{title}」耗时 {dt/60:.1f} 分钟")
    return True


def _clear_dir(rel: str):
    path = os.path.join(DATA_DIR, rel)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f"[清理] 已删除 {rel}/")
    os.makedirs(path, exist_ok=True)


def _check_prereqs(need_venv: bool):
    """前置检查，缺依赖时直接退出并给提示。"""
    ds = os.path.join(DATA_DIR, "dataset.json")
    if not os.path.exists(ds):
        print("[错误] 未找到评测集 eval/data/dataset.json，请先运行 generate_dataset.py 或用 --regenerate")
        sys.exit(1)
    if need_venv and not os.path.exists(VENV_PYTHON):
        print(f"[错误] 未找到评分 venv: {VENV_PYTHON}")
        print("       请先搭建（见 README 16.6）：python -m venv eval/.venv 后 pip install ragas 并降级 langchain-community")
        sys.exit(1)
    print("[检查] 评测集就绪 ✓")


# ============================================================================
# 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="RAG 完整评测一键编排")
    ap.add_argument("--subset", default="mode-comparison", choices=["all", "mode-comparison"],
                    help="组合子集（默认 mode-comparison = 6 组检索模式对比）")
    ap.add_argument("--combo", default=None, help="只跑单个组合 id")
    ap.add_argument("--regenerate", action="store_true", help="重新生成评测集（消耗 LLM token）")
    ap.add_argument("--keep", action="store_true", help="保留已有 runs/scores，断点续跑（不清空）")
    ap.add_argument("--skip-collect", action="store_true", help="跳过采集（已有 runs 时用）")
    ap.add_argument("--skip-score", action="store_true", help="跳过评分（只采集）")
    args = ap.parse_args()

    start = time.time()
    print("RAG 完整评测编排")
    print(f"  子集: {args.subset} | 组合: {args.combo or '（该子集全部）'}")
    print(f"  重新生成评测集: {'是' if args.regenerate else '否'}")
    print(f"  保留已有结果: {'是（断点续跑）' if args.keep else '否（从头清空）'}")

    # 组装采集/评分共用的子集参数
    sub_args = []
    if args.subset:
        sub_args += ["--subset", args.subset]
    if args.combo:
        sub_args += ["--combo", args.combo]

    # ---------- 0. 前置检查 ----------
    _check_prereqs(need_venv=not args.skip_score)

    # ---------- 1. 清空旧产物（除非 --keep） ----------
    if not args.keep:
        _step("清空旧评测产物")
        _clear_dir("runs")
        _clear_dir("scores")
        rp = os.path.join(DATA_DIR, "report.md")
        if os.path.exists(rp):
            os.remove(rp)
            print("[清理] 已删除 report.md")

    # ---------- 2. 生成评测集（可选） ----------
    if args.regenerate:
        if not _run([sys.executable, os.path.join(EVAL_DIR, "generate_dataset.py")],
                    "生成评测集"):
            sys.exit(1)

    # ---------- 3. 采集 ----------
    if not args.skip_collect:
        collect_cmd = [sys.executable, os.path.join(EVAL_DIR, "collect.py")] + sub_args
        if args.keep:
            collect_cmd += ["--retry-empty"]  # 续跑时自动补断网导致的 ctx=0 空结果
        if not _run(collect_cmd, "采集（answer + contexts）"):
            print("\n[提示] 采集未全部完成。修复问题后重跑本脚本加 --keep 可断点续跑，不会重复已完成的题。")
            sys.exit(1)
    else:
        print("\n[跳过] 采集（--skip-collect）")

    # ---------- 4. 评分 ----------
    if not args.skip_score:
        if not _run([VENV_PYTHON, os.path.join(EVAL_DIR, "score.py")] + sub_args,
                    "Ragas 评分（5 指标）"):
            print("\n[提示] 评分未全部完成。重跑本脚本加 --skip-collect --keep 可只补评分。")
            sys.exit(1)
    else:
        print("\n[跳过] 评分（--skip-score）")

    # ---------- 5. 报告 ----------
    if not _run([sys.executable, os.path.join(EVAL_DIR, "report.py")],
                "汇总报告"):
        sys.exit(1)

    total = time.time() - start
    print("\n" + "=" * 72)
    print(f"✅ 全部完成，总耗时 {total/60:.1f} 分钟")
    print(f"   报告位置：{os.path.join(DATA_DIR, 'report.md')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
