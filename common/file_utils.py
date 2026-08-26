"""RAGFlow common.file_utils 的轻量替代实现。"""
import os


def get_project_base_directory() -> str:
    # 返回项目根目录（common 的上一级）
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def traversal_files(base: str):
    """递归遍历目录，返回所有文件路径。"""
    results = []
    for root, _dirs, files in os.walk(base):
        for f in files:
            results.append(os.path.join(root, f))
    return results
