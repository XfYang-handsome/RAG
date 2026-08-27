#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键停止：停止外部服务（Milvus + Redis）容器。

主程序（uvicorn）是前台进程，用 Ctrl+C 或关闭启动它的终端窗口即可停止；
本脚本只负责停掉后台容器，避免下次启动端口冲突 / 便于宿主机重启。
"""

import os
import shutil
import subprocess
import sys

DEPLOY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DEPLOY)
COMPOSE_FILE = os.path.join(DEPLOY, "docker-compose.yml")


def have(cmd):
    return shutil.which(cmd) is not None


def compose_cmd():
    return ["docker", "compose"] if have("docker") else (["docker-compose"] if have("docker-compose") else None)


def main():
    os.chdir(ROOT)
    base = compose_cmd()
    if base is None:
        print("[!!] 未找到 docker，无法停止容器")
        sys.exit(1)
    print("==> 停止外部服务 (Milvus + Redis)")
    subprocess.run(base + ["-f", COMPOSE_FILE, "stop"], cwd=ROOT)
    print("    已停止。数据保留在 ./milvus 与 ./redis，下次 start 会继续使用。")


if __name__ == "__main__":
    main()