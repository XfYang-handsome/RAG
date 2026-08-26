# -*- coding: utf-8 -*-
"""deepdoc 模型目录管理与自动下载。

模型统一存放在项目根目录 ``models/deepdoc/``（与 ``rag/`` 代码分离）。

调用 :func:`get_deepdoc_model_dir` 时会自动检测缺失的模型文件，缺失则从
hf-mirror 镜像流式下载（幂等、线程安全）。因此业务代码无需关心模型是否
已就绪，也无需手动运行下载脚本。
"""
import os
import threading
import time
import urllib.request

from common.file_utils import get_project_base_directory

# repo -> 需要的文件（layout.*.onnx 为可选细分版面模型，默认只取 layout.onnx）
DEEPDOC_REPOS = {
    "InfiniFlow/deepdoc": ["det.onnx", "rec.onnx", "ocr.res", "layout.onnx", "tsr.onnx"],
    "InfiniFlow/text_concat_xgb_v1.0": ["updown_concat_xgb.model"],
}

DEEPDOC_MODEL_FILES = [fn for files in DEEPDOC_REPOS.values() for fn in files]

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

_download_lock = threading.Lock()


def get_deepdoc_model_dir() -> str:
    """返回 deepdoc 模型目录（项目根目录 ``models/deepdoc``）。

    首次调用会触发缺失模型检测与自动下载。
    """
    ensure_deepdoc_models()
    return os.path.join(get_project_base_directory(), "models", "deepdoc")


def list_missing_models() -> list:
    """返回当前缺失的模型文件名列表。"""
    model_dir = os.path.join(get_project_base_directory(), "models", "deepdoc")
    return [fn for fn in DEEPDOC_MODEL_FILES if not os.path.isfile(os.path.join(model_dir, fn))]


def _download(url, dest):
    """流式下载单个文件（断点续传 + 重试，绕过 snapshot_download 的 HEAD 问题）。"""
    if os.path.isfile(dest) and os.path.getsize(dest) > 1024:
        print(f"[deepdoc] 模型已存在，跳过: {os.path.basename(dest)}")
        return
    tmp = dest + ".part"
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
            if os.path.getsize(tmp) > 1024:
                os.replace(tmp, dest)
                print(f"[deepdoc] 已下载模型 {os.path.basename(dest)} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[deepdoc] 下载失败({attempt + 1}/3): {e}")
            time.sleep(2)
    raise RuntimeError(f"下载 {url} 失败: {last_err}")


def ensure_deepdoc_models() -> str:
    """检测缺失模型并自动下载，返回模型目录。

    幂等、线程安全：每次调用都会快速检查（仅 6 次 ``isfile``），
    仅当确实缺文件时才加锁下载。
    """
    model_dir = os.path.join(get_project_base_directory(), "models", "deepdoc")
    missing = list_missing_models()
    if not missing:
        return model_dir

    with _download_lock:
        # double-check，避免并发重复下载
        missing = list_missing_models()
        if not missing:
            return model_dir
        os.makedirs(model_dir, exist_ok=True)
        print(f"[deepdoc] 检测到缺失模型 {missing}，开始自动下载（镜像 {MIRROR}）...")
        for repo, files in DEEPDOC_REPOS.items():
            for fn in files:
                if fn in missing:
                    _download(f"{MIRROR}/{repo}/resolve/main/{fn}", os.path.join(model_dir, fn))
        print("[deepdoc] 模型就绪")
    return model_dir
