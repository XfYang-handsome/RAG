# -*- coding: utf-8 -*-
"""手动下载 deepdoc 所需的 OCR / 版面 / 表格模型到 models/deepdoc。

运行：poetry run python download_deepdoc_models.py

说明：代码在运行时若检测不到模型会自动下载（见 common/deepdoc_models.py），
本脚本仅供需要提前预下载模型的场景使用。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.deepdoc_models import ensure_deepdoc_models, list_missing_models  # noqa: E402


def main():
    model_dir = ensure_deepdoc_models()
    print("模型目录:", model_dir)

    print("\n目录内容:")
    for f in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p):
            print(f"  - {f}  ({os.path.getsize(p) / 1024 / 1024:.1f} MB)")

    missing = list_missing_models()
    if missing:
        print("\n缺少文件:", missing)
        sys.exit(1)
    print("\n全部模型文件齐全 OK")


if __name__ == "__main__":
    main()
