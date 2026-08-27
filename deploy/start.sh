#!/usr/bin/env bash
# 一键启动（主程序 + MCP + Celery Worker）
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" start.py "$@"