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


def main():
    p = argparse.ArgumentParser(description="RAG 一键启动")
    p.add_argument("--no-mcp", action="store_true", help="不启动 MCP 服务器")
    p.add_argument("--no-celery", action="store_true", help="不启动 Celery Worker")
    p.add_argument("--port", type=int, default=None, help="覆盖服务端口（默认读 config.json）")
    p.add_argument("--host", type=str, default=None, help="覆盖监听地址（默认读 config.json）")
    args = p.parse_args()

    os.chdir(ROOT)

    # 1) 确保外部服务已起
    base = compose_cmd()
    if base is None:
        print("[!!] 未找到 docker，跳过容器启动（请确认 Milvus/Redis 已在运行）")
    else:
        print("==> 拉起外部服务 (Milvus + Redis)")
        r = subprocess.run(base + ["-f", COMPOSE_FILE, "ps", "-q", "milvus", "redis"], capture_output=True, text=True)
        print(r.stdout or "(未在运行)")
        subprocess.run(base + ["-f", COMPOSE_FILE, "up", "-d"], cwd=ROOT)

    # 2) 组装主程序命令
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