#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库系统 —— 一键初始化脚本（跨平台：Windows / Linux / macOS）

目标：在「全新的电脑」上，除 4 个系统依赖外全部自动完成。

用法：
    python deploy/setup.py                # 自动检测 GPU，走完整流程
    python deploy/setup.py --cpu          # 强制 CPU 版依赖（torch CPU + onnxruntime）
    python deploy/setup.py --gpu          # 强制 GPU 版依赖（torch cu128 + onnxruntime-gpu）
    python deploy/setup.py --restore      # 回退：恢复被 --cpu 改写的 pyproject.toml

可跳过某一步（依赖已就绪时）：
    --skip-deps --skip-models --skip-frontend --skip-services

本脚本自动完成：
    1) 检测操作系统与 NVIDIA GPU，自动选择依赖版本（cu128 / CPU）
    2) 从 config/*.example 生成 config/models.json、config/db.json（仅首次）
    3) poetry install（安装 Python 依赖）
    4) 预下载 deepdoc 解析模型（hf-mirror 镜像）
    5) cd frontend && npm install && npm run build（构建前端）
    6) docker compose up -d（拉起 Milvus + Redis）
    7) 健康检查并打印后续步骤
"""

import argparse
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

DEPLOY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEPLOY)
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
PYPROJECT_BAK = os.path.join(ROOT, "pyproject.toml.gpu.bak")
CFG_DIR = os.path.join(ROOT, "config")
COMPOSE_FILE = os.path.join(DEPLOY, "docker-compose.yml")

# Windows 上用 SetConsoleOutputCP 切到 UTF-8，避免中文 git/bash 乱码
_OS_NAME = platform.system()  # Windows / Linux / Darwin


def _stdout_utf8():
    if _OS_NAME == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_stdout_utf8()


# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------

def ok(m):   print(f"  [OK]   {m}")
def info(m): print(f"  [..]   {m}")
def warn(m): print(f"  [!!]   {m}")
def err(m):  print(f"  [ERR]  {m}")


def section(m):
    print("\n" + "=" * 66)
    print("  " + m)
    print("=" * 66)


# ---------------------------------------------------------------------------
# 命令执行
# ---------------------------------------------------------------------------

def run(cmd, shell=False, cwd=None, check=True):
    if isinstance(cmd, list):
        disp = " ".join(cmd)
    else:
        disp = cmd
    info(f"$ {disp}")
    if shell and isinstance(cmd, list):
        cmd = " ".join(cmd)
    return subprocess.run(cmd, shell=shell, cwd=cwd, check=check)


def have(cmd):
    """命令是否在 PATH 中。"""
    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------

def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def check_python():
    v = sys.version_info
    return v.major == 3 and 12 <= v.minor < 13


# ---------------------------------------------------------------------------
# 依赖缺失时的安装指引
# ---------------------------------------------------------------------------

INSTALL_HINTS = {
    "python": {
        "Windows": "到 https://www.python.org/downloads/ 装 Python 3.12（勾选 Add to PATH）",
        "Linux":   "sudo apt install -y python3.12（或编译安装/python3.12 源）",
        "Darwin":  "brew install python@3.12",
    },
    "poetry": {
        "Windows": 'powershell -c "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -"',
        "Linux":   "curl -sSL https://install.python-poetry.org | python3 -",
        "Darwin":  "brew install poetry",
    },
    "docker": {
        "Windows": "安装 Docker Desktop：https://www.docker.com/products/docker-desktop/",
        "Linux":   "curl -fsSL https://get.docker.com | sh && sudo systemctl enable --now docker",
        "Darwin":  "安装 Docker Desktop：https://www.docker.com/products/docker-desktop/",
    },
    "node": {
        "Windows": "到 https://nodejs.org/ 装 LTS（自带 npm）",
        "Linux":   "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs",
        "Darwin":  "brew install node",
    },
}


def check_tools():
    """返回缺失的依赖列表。"""
    missing = []
    if not check_python():
        missing.append("python")
    if not have("poetry"):
        missing.append("poetry")
    if not (have("docker") or have("docker-compose")):
        missing.append("docker")
    if not (have("node") and have("npm")):
        missing.append("node")
    return missing


def print_missing_hints(missing):
    print("\n检测到缺少下列系统依赖，请先安装后重新运行本脚本：\n")
    for m in missing:
        print(f"  * {m}: {INSTALL_HINTS.get(m, {}).get(_OS_NAME, '请自行安装')}")
    print("\n（其中 docker 用于跑 Milvus/Redis 容器；node 用于构建前端。）")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 配置模板
# ---------------------------------------------------------------------------

def init_config_templates():
    """首次运行：从 .example 生成带占位的 models.json / db.json。"""
    for name in ("models.json", "db.json"):
        dst = os.path.join(CFG_DIR, name)
        src = os.path.join(CFG_DIR, name + ".example")
        if os.path.exists(dst):
            info(f"已存在 {dst}，跳过")
            continue
        if not os.path.exists(src):
            warn(f"缺少模板 {src}，将依赖程序运行时生成空配置")
            continue
        shutil.copy(src, dst)
        ok(f"已生成 {dst}（含占位，稍后在 Web 界面填 API Key）")


# ---------------------------------------------------------------------------
# CPU 版依赖切换（torch/onnxruntime 的 cu128 -> CPU）
# ---------------------------------------------------------------------------

def _patch_pyproject_cpu():
    with open(PYPROJECT, "r", encoding="utf-8") as f:
        content = f.read()

    if not os.path.exists(PYPROJECT_BAK):
        shutil.copy(PYPROJECT, PYPROJECT_BAK)
        info(f"已备份原 pyproject.toml -> {os.path.basename(PYPROJECT_BAK)}")

    new = content.replace(
        '"onnxruntime-gpu (>=1.26.0,<1.27.0)",',
        '"onnxruntime (>=1.26.0,<2.0.0)",',
    )
    # 删除 [tool.poetry.dependencies] 里指向 cu128 源的 torch 行，
    # 让 torch 回落到 primary 源（清华镜像，PyPI 上是 CPU 版）。不依赖文件末尾换行。
    new = re.sub(
        r'^torch = \{source = "pytorch-cu128"\}[ \t]*\n?',
        "",
        new,
        flags=re.M,
    )
    if new == content:
        warn("未匹配到需切换的依赖行（可能已切换），跳过改写")
        return False
    with open(PYPROJECT, "w", encoding="utf-8") as f:
        f.write(new)
    ok("已将依赖切换为 CPU 版（torch 走清华源 + onnxruntime）")
    return True


def restore_pyproject():
    if not os.path.exists(PYPROJECT_BAK):
        err("未找到备份 pyproject.toml.gpu.bak，无需恢复")
        sys.exit(1)
    shutil.copy(PYPROJECT_BAK, PYPROJECT)
    ok("已从 pyproject.toml.gpu.bak 恢复 pyproject.toml（GPU/cu128 版）")
    info("恢复后请重新运行：poetry install")


# ---------------------------------------------------------------------------
# 各步骤
# ---------------------------------------------------------------------------

def step_install_deps(gpu: bool, force_cpu: bool):
    section("3/6 安装 Python 依赖 (poetry install)")
    if force_cpu and not gpu:
        info("当前为 CPU 模式，先改写 pyproject.toml 切换依赖")
        _patch_pyproject_cpu()
        info("重新解析依赖锁 (poetry lock)")
        try:
            run(["poetry", "lock", "--no-interaction"], cwd=ROOT)
        except Exception as e:
            warn(f"poetry lock 失败（{e}），继续尝试 install")
    try:
        run(["poetry", "install", "--no-interaction"], cwd=ROOT)
        ok("Python 依赖安装完成")
    except Exception as e:
        err(f"poetry install 失败：{e}")
        print("  常见原因：网络/镜像不稳、磁盘不足、CUDA 版本不匹配（用 --cpu 重试）。")
        sys.exit(1)


def step_download_models():
    section("4/6 预下载 deepdoc 解析模型")
    try:
        run(["poetry", "run", "python", "download_deepdoc_models.py"], cwd=ROOT)
        ok("deepdoc 模型就绪")
    except Exception as e:
        warn(f"模型预下载未完成（{e}）——不影响启动，首次解析时仍会自动补下")


def step_build_frontend():
    section("5/6 构建前端 (npm install + build)")
    fe = os.path.join(ROOT, "frontend")
    if not os.path.isdir(fe):
        warn("未找到 frontend 目录，跳过前端构建")
        return
    try:
        run(["npm", "install"], shell=True, cwd=fe)
        run(["npm", "run", "build"], shell=True, cwd=fe)
        ok("前端构建完成（产物 static/dist/）")
    except Exception as e:
        err(f"前端构建失败：{e}")
        sys.exit(1)


def _compose_cmd():
    if have("docker"):
        return ["docker", "compose"]
    if have("docker-compose"):
        return ["docker-compose"]
    return None


def step_start_services():
    section("6/6 启动外部服务 (Milvus + Redis)")
    base = _compose_cmd()
    if base is None:
        warn("未找到 docker / docker-compose，跳过容器启动（请手动起 Milvus 与 Redis）")
        return
    try:
        run(base + ["-f", COMPOSE_FILE, "up", "-d"], cwd=ROOT)
        ok("Milvus + Redis 容器已拉起")
    except Exception as e:
        err(f"docker compose up -d 失败：{e}")
        print("  请确认 Docker 已启动（Docker Desktop 需先打开），再重试。")
        sys.exit(1)


def _wait_http(url, timeout=90, interval=3):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(interval)
    return False


def _wait_port(port, timeout=60, interval=2):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(interval)
    return False


def health_check():
    section("健康检查")
    milvus_ok = _wait_http("http://127.0.0.1:9091/healthz", timeout=120)
    redis_ok = _wait_port(6379, timeout=60)
    if milvus_ok:
        ok("Milvus (19530) 就绪")
    else:
        warn("Milvus 未在 120s 内就绪（首次启动需拉镜像/初始化，可稍后重试或查看容器日志）")
    if redis_ok:
        ok("Redis (6379) 就绪")
    else:
        warn("Redis 未就绪")


def print_next(force_cpu: bool):
    print("\n" + "=" * 66)
    print("  初始化完成！接下来的步骤：")
    print("=" * 66)
    print()
    print("  1) 填模型配置：")
    print("     编辑 config/models.json（或在 Web 界面「系统设置」里）")
    print("     填入 LLM / Embedding / Reranker 的 API Key（硅基流动等）")
    print()
    print("  2) 启动服务：")
    print("     Windows 双击      deploy/start.bat")
    print("     Linux/macOS 运行  bash deploy/start.sh")
    print("     （等价于 poetry run python __main__.py --mcp --celery）")
    print()
    print("  3) 浏览器打开 http://127.0.0.1:8000")
    if force_cpu:
        print()
        print("  [注意] 当前为 CPU 版依赖；如需切回 GPU（cu128），运行：")
        print("         python deploy/setup.py --restore && poetry install")
    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="RAG 一键初始化")
    p.add_argument("--cpu", action="store_true", help="强制 CPU 版依赖")
    p.add_argument("--gpu", action="store_true", help="强制 GPU 版依赖（cu128）")
    p.add_argument("--restore", action="store_true", help="恢复 pyproject.toml 为 GPU 版")
    p.add_argument("--skip-deps", action="store_true", help="跳过 poetry install")
    p.add_argument("--skip-models", action="store_true", help="跳过 deepdoc 模型下载")
    p.add_argument("--skip-frontend", action="store_true", help="跳过前端构建")
    p.add_argument("--skip-services", action="store_true", help="跳过 docker compose")
    args = p.parse_args()

    os.chdir(ROOT)

    if args.restore:
        restore_pyproject()
        return

    print("=" * 66)
    print("  RAG 知识库系统 —— 一键初始化")
    print("=" * 66)
    print(f"  操作系统 : {_OS_NAME}")
    print(f"  Python   : {platform.python_version()}")
    print(f"  工作目录 : {ROOT}")

    # ---- 0. 检查系统依赖 ----
    section("0/6 检查系统依赖")
    missing = check_tools()
    if missing:
        print_missing_hints(missing)
    ok("Python / Poetry / Docker / Node.js 均已就绪")

    # ---- 1. 检测 GPU，决定依赖版本 ----
    section("1/6 检测 GPU 并选择依赖版本")
    gpu = detect_gpu()
    if args.cpu:
        force_cpu = True
        info("用户指定 --cpu：使用 CPU 版依赖")
    elif args.gpu:
        force_cpu = False
        info("用户指定 --gpu：使用 GPU 版依赖（cu128）")
    else:
        force_cpu = not gpu
        if gpu:
            ok("检测到 NVIDIA GPU：使用 cu128 + onnxruntime-gpu")
        else:
            warn("未检测到 NVIDIA GPU：自动切换为 CPU 版依赖")
            warn("  （若其实有 GPU，请确认驱动已装并重跑，或用 --gpu 强制）")

    # ---- 2. 生成配置模板 ----
    section("2/6 生成配置模板")
    init_config_templates()

    # ---- 3. 安装依赖 ----
    if args.skip_deps:
        info("跳过依赖安装（--skip-deps）")
    else:
        step_install_deps(gpu, force_cpu)

    # ---- 4. 下载模型 ----
    if args.skip_models:
        info("跳过模型下载（--skip-models）")
    else:
        step_download_models()

    # ---- 5. 构建前端 ----
    if args.skip_frontend:
        info("跳过前端构建（--skip-frontend）")
    else:
        step_build_frontend()

    # ---- 6. 启动服务 ----
    if args.skip_services:
        info("跳过容器启动（--skip-services）")
    else:
        step_start_services()

    # ---- 7. 健康检查 ----
    if not args.skip_services:
        health_check()

    print_next(force_cpu)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消.")
        sys.exit(130)