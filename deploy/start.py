#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动：拉起外部服务（Milvus + Redis）+ 主程序（--mcp --celery）。

用法：
    python deploy/start.py            # 完整启动（推荐）
    python deploy/start.py --no-celery
    python deploy/start.py --no-mcp
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys

DEPLOY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEPLOY)
COMPOSE_FILE = os.path.join(DEPLOY, "docker-compose.yml")


def _stdout_utf8():
    if sys.platform.startswith("win"):
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


def have(cmd):
    return shutil.which(cmd) is not None


def compose_cmd():
    return ["docker", "compose"] if have("docker") else (["docker-compose"] if have("docker-compose") else None)


def ensure_python_deps():
    """确认 poetry 环境中已安装依赖（以能 import uvicorn 为准）。

    首次部署若未执行 setup.py（poetry install），直接 poetry run 会新建一个
    空虚拟环境并报 "No module named 'uvicorn'"。这里提前探测并给出明确指引。
    """
    try:
        r = subprocess.run(
            ["poetry", "run", "python", "-c", "import uvicorn"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("\n[!!] 未找到 poetry，请先安装 poetry（见 deploy/setup.py 提示）。")
        return False
    if r.returncode == 0:
        return True
    print("\n[!!] 检测到 Python 依赖尚未安装（缺少 uvicorn 等）。")
    print("     首次部署请先运行一键初始化：")
    print("         Windows : 双击 deploy/setup.bat")
    print("         Linux   : python deploy/setup.py")
    print("     或手动安装依赖： poetry install --no-interaction")
    return False


def port_in_use(port):
    """返回 127.0.0.1:port 是否已被占用。"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


PORT_NAMES = {6379: "Redis (6379)", 9091: "Milvus 健康检查 (9091)"}


def report_compose_error(out):
    """compose up 失败时给出针对性提示。"""
    print("[!!] 外部服务（Milvus + Redis）容器启动失败。")
    if "port is already" in out or "already in use" in out or "already allocated" in out:
        busy = [p for p in PORT_NAMES if port_in_use(p)]
        if busy:
            print("     以下端口已被本机其它进程占用：")
            for p in busy:
                print("       - " + PORT_NAMES[p])
        print("     解决（任选其一）：")
        print('       1) 找出并停止占用进程（Windows）：')
        print('            netstat -ano | findstr ":6379 :9091"')
        print("            然后 taskkill /PID <pid> /F")
        print("          Linux/macOS: lsof -i :6379 -i :9091")
        print("       2) 若为本项目旧容器残留： docker compose -f deploy/docker-compose.yml down")
        print("       3) 若本机已装有原生 Redis / Milvus，可直接复用（跳过容器）继续启动。")
    else:
        print("     请确认 Docker 已启动（Docker Desktop 需先打开）。")
        if out.strip():
            print("     原始输出：")
            for line in out.strip().splitlines()[-6:]:
                print("       " + line)


def main():
    p = argparse.ArgumentParser(description="PrismRAG 一键启动")
    p.add_argument("--no-mcp", action="store_true", help="不启动 MCP 服务器")
    p.add_argument("--no-celery", action="store_true", help="不启动 Celery Worker")
    p.add_argument("--port", type=int, default=None, help="覆盖服务端口（默认读 config.json）")
    p.add_argument("--host", type=str, default=None, help="覆盖监听地址（默认读 config.json）")
    args = p.parse_args()

    os.chdir(ROOT)

    # 1) 先确认 Python 依赖已安装（否则主程序必然因缺 uvicorn 崩溃）
    if not ensure_python_deps():
        sys.exit(1)

    # 2) 确保外部服务已起
    base = compose_cmd()
    if base is None:
        print("[!!] 未找到 docker，跳过容器启动（请确认 Milvus/Redis 已在运行）")
    else:
        print("==> 拉起外部服务 (Milvus + Redis)")
        r = subprocess.run(base + ["-f", COMPOSE_FILE, "ps", "-q", "milvus", "redis"],
                           capture_output=True, text=True)
        running = {line for line in r.stdout.splitlines() if line}
        if len(running) >= 2:
            print("    容器已在运行，跳过启动")
        else:
            up = subprocess.run(base + ["-f", COMPOSE_FILE, "up", "-d"], cwd=ROOT,
                                capture_output=True, text=True)
            if up.returncode != 0:
                report_compose_error((up.stdout or "") + (up.stderr or ""))
            else:
                print("    外部服务已就绪")

    # 3) 组装主程序命令
    cmd = ["poetry", "run", "python", "__main__.py"]
    if not args.no_mcp:
        cmd.append("--mcp")
    if not args.no_celery:
        cmd.append("--celery")
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.host:
        cmd += ["--host", args.host]

    print("==> 启动主程序:", " ".join(cmd))
    print("    浏览器打开 http://127.0.0.1:8000 ，Ctrl+C 或关闭窗口即可停止\n")
    # 前台运行，Ctrl+C 直接交给主程序处理（其内部已处理 MCP/Worker 的清理）
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()