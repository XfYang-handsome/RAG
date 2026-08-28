"""
配置加载器 — 从 config/config.json 读取所有非本地模型和云端 Milvus 配置。

用法：
    from config_loader import config
    config["embedding"]["base_url"]

也支持点号路径访问：
    cfg("embedding.base_url") → "https://api.deepseek.com/v1"

配置文件位置：项目根目录下的 config/config.json。
启动时若文件不存在，会自动创建一份默认配置。
"""

import json
import os
import threading
from typing import Any

# ============================================================================
# 默认配置（启动时若 config/config.json 不存在，用它初始化）
# ============================================================================

DEFAULT_CONFIG = {
    "chunking": {
        "parent_chunk_size": 1024,
        "child_chunk_size": 256,
        "parent_overlap": 100,
        "child_overlap": 30
    },
    "search": {
        "retrieval_top_k": 20,
        "rerank_top_n": 5,
        "grade_relevance_threshold": 0.25,
        "rewrite": False,
        "hybrid": True,
        "retrieval_mode": "hybrid"
    },
    "summary": {
        "enabled": True,
        "concurrency": 4
    },
    "deepdoc": {
        "zoomin": 3
    },
    "tools": {
        "web_search": {"enabled": True},
        "calculate_pi": {"enabled": True},
        "calculate_expression": {"enabled": True}
    },
    "structure": {
        "reconstruct": {
            "enabled": True,
            "batch_size": 20,
            "max_llm_calls": 30,
            "root_content_ratio": 0.6
        }
    },
    "agentic": {
        "max_iterations": 5,
        "max_tool_calls": 12,
        "importance_high": 0.8,
        "no_progress_threshold": 2,
        "search": {
            "top_k": 5,
            "multi_query_enabled": True,
            "multi_query_min": 2,
            "multi_query_max": 4,
            "translate_keywords_min": 3,
            "translate_keywords_max": 6
        },
        "context": {
            "neighbor_window": 2
        },
        "evaluator": {
            "new_evidence_chars": 500,
            "history_evidence_chars": 150,
            "summary_chars": 80,
            "retry": 1,
            "prune_top_n": 8
        },
        "controller": {
            "retry": 1
        },
        "planner": {
            "min_requirements": 1,
            "max_requirements": 5
        },
        "synthesizer": {
            "evidence_chars": 1500
        },
        "tree_nav": {
            "node_min_score": 0.2,
            "leaf_min_score": 0.2,
            "max_depth": 4,
            "max_expansions": 8,
            "max_llm_calls": 6,
            "max_leaf_reads": 20,
            "min_evidences": 2
        }
    },
    "theme": {
        "light": {
            "gradient": True,
            "color1": "#0ea5e9",
            "color2": "#06b6d4"
        },
        "dark": {
            "gradient": True,
            "color1": "#6366f1",
            "color2": "#a855f7"
        }
    },
    "system_prompt": "",
    "server": {
        "host": "127.0.0.1",
        "port": 8000
    },
    "mcp": {
        "features": {
            "websearch": True
        },
        "web_search": {
            "proxy": "",
            "timeout": 6.0
        },
        "tool_calling": {
            "enabled": False
        }
    },
    "supported_extensions": [
        ".txt", ".md", ".py", ".js", ".ts", ".java", ".cpp", ".c",
        ".h", ".json", ".xml", ".yaml", ".yml", ".csv", ".html", ".css",
        ".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".odt", ".rtf", ".eml"
    ]
}

# ============================================================================
# 配置文件路径定位
# ============================================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_BASE_DIR, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


def _ensure_config_dir():
    """确保 config 目录存在"""
    os.makedirs(CONFIG_DIR, exist_ok=True)


def _ensure_config_file():
    """启动时若 config.json 不存在，用默认配置创建"""
    if os.path.isfile(CONFIG_PATH):
        return
    _ensure_config_dir()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已创建默认配置文件: {CONFIG_PATH}")


# 启动时初始化（确保配置文件存在）
_ensure_config_file()

_config_cache: dict = None
_config_lock = threading.RLock()  # 保护 _config_cache 的并发读写


def _load_config() -> dict:
    """加载并缓存 config/config.json"""
    global _config_cache
    with _config_lock:
        if _config_cache is None:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = json.load(f)
        return _config_cache


def _get_config() -> dict:
    return _load_config()


def cfg(path: str, default: Any = None) -> Any:
    """
    用点号路径访问配置项。

    示例:
        cfg("embedding.base_url")     → "..."
        cfg("milvus.cloud.url")       → "..."
        cfg("nonexistent", "fallback") → "fallback"
    """
    keys = path.split(".")
    value = _get_config()
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value


def set_config(path: str, value: Any) -> None:
    """
    写入配置项（点号路径），并刷新缓存。

    示例:
        set_config("search.use_langgraph", False)
    """
    global _config_cache
    with _config_lock:
        data = _load_config()
        keys = path.split(".")
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _config_cache = data  # 刷新缓存


# 模块级 config 字典引用（惰性加载）
class _ConfigProxy:
    """代理类，使得 config["embedding"] 等价于 config["embedding"]"""

    def __getitem__(self, key):
        return _get_config()[key]

    def get(self, key, default=None):
        return _get_config().get(key, default)

    def __contains__(self, key):
        return key in _get_config()

    def keys(self):
        return _get_config().keys()

    def values(self):
        return _get_config().values()

    def items(self):
        return _get_config().items()

    def __iter__(self):
        return iter(_get_config())


config = _ConfigProxy()
