#!/usr/bin/env bash
# 一键停止外部服务（Milvus + Redis 容器）
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" stop.py "$@"