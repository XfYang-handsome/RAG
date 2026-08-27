#!/usr/bin/env bash
# 一键初始化：python3 scripts/setup.py 的封装（Linux / macOS 用）
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" setup.py "$@"