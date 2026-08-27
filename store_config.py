"""
================================================================================
配置存储模块 — 管理多模型 / 多数据库配置
================================================================================

模型和数据库配置保存到独立的 JSON 文件中（而非 config.json），
支持 Web 界面动态增删改。

文件结构：
  models.json:
  {
    "llm": [
      {"name": "deepseek", "type": "online", "model": "deepseek-chat",
       "base_url": "https://api.deepseek.com/v1", "api_key": "..."}
    ],
    "embedding": [
      {"name": "bge-m3", "type": "online", "model": "BAAI/bge-m3",
       "base_url": "https://api.siliconflow.cn/v1", "api_key": "..."}
    ],
    "reranker": [
      {"name": "bge-reranker-local", "type": "local",
       "model_path": "BAAI/bge-reranker-v2-m3"},
      {"name": "jina-reranker", "type": "online",
       "model": "jina-reranker-v2-base-multilingual",
       "base_url": "https://api.jina.ai/v1", "api_key": "..."}
    ],
    "current": {
      "llm": "deepseek",
      "reranker": "bge-reranker-local"
    }
  }

  db.json:
  {
    "milvus": [
      {"name": "本地默认库", "type": "local", "url": "http://localhost:19530",
       "db_name": "default"},
      {"name": "云端库", "type": "online", "url": "http://<your-milvus-host>:19531",
       "token": "...", "db_name": "..."}
    ],
    "current": "本地默认库"
  }

模型类型说明：
  - llm / embedding：只支持 online（url + api_key + model name）
  - reranker：支持 online 和 local（local 用 model_path 本地路径）
================================================================================
"""

import copy
import os
import json
import threading
from typing import List, Optional

# 配置文件路径（统一放在项目根目录下的 config/ 目录）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")
MODELS_FILE = os.path.join(_CONFIG_DIR, "models.json")
DB_FILE = os.path.join(_CONFIG_DIR, "db.json")

_lock = threading.RLock()  # 可重入线程锁，保护文件读写 + 内存缓存

# 内存缓存（首次读取后缓存，写操作时刷新；消除检索热路径的重复磁盘 IO）
_models_cache: Optional[dict] = None
_dbs_cache: Optional[dict] = None


# ============================================================================
# 通用 JSON 读写
# ============================================================================

def _ensure_config_dir():
    """确保 config 目录存在"""
    os.makedirs(_CONFIG_DIR, exist_ok=True)


def _ensure_file(path: str, default: dict):
    """确保文件存在，不存在则创建默认内容"""
    if os.path.isfile(path):
        return
    _ensure_config_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default, f, ensure_ascii=False, indent=2)


def _read_json(path: str, default: dict) -> dict:
    """读取 JSON 文件，不存在则创建默认文件"""
    _ensure_file(path, default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if data is not None else default
    except (json.JSONDecodeError, IOError):
        return default


def _write_json(path: str, data: dict):
    """写入 JSON 文件（带线程锁 + 原子写）"""
    with _lock:
        _ensure_config_dir()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子替换，避免写入中断损坏文件


# ============================================================================
# 模型配置管理
# ============================================================================

def _default_models() -> dict:
    """默认模型配置（首次运行时从 config.json 迁移）

    模型类型说明：
      llm        生成模型（最终回答，不绑定工具）
      embedding  向量化模型
      reranker   精排模型（online / local）
      tool_llm   工具决策模型（可选，绑定工具决定调用哪些工具；
                 未配置时回退到 llm。需选择支持 function calling 的模型）
    """
    return {"llm": [], "embedding": [], "reranker": [], "tool_llm": [], "summary": []}


def load_models() -> dict:
    """加载全部模型配置（内存缓存，写操作后自动刷新）。

    返回深拷贝，隔离调用方对返回值的修改，避免污染缓存（与原「每次读文件」
    返回新对象的隔离语义一致）。
    """
    global _models_cache
    with _lock:
        if _models_cache is None:
            _models_cache = _read_json(MODELS_FILE, _default_models())
        return copy.deepcopy(_models_cache)


def save_models(data: dict):
    """保存全部模型配置（同时刷新内存缓存）"""
    global _models_cache
    with _lock:
        _write_json(MODELS_FILE, data)
        _models_cache = data


def list_models(kind: str) -> List[dict]:
    """
    列出指定类型的所有模型。

    Args:
        kind: "llm" / "embedding" / "reranker"
    """
    data = load_models()
    return data.get(kind, [])


def get_model_by_name(kind: str, name: str) -> Optional[dict]:
    """按名称查找模型"""
    for m in list_models(kind):
        if m.get("name") == name:
            return m
    return None


def add_model(kind: str, model: dict) -> dict:
    """新增模型"""
    data = load_models()
    if kind not in data:
        data[kind] = []
    # 重名检查
    for m in data[kind]:
        if m.get("name") == model.get("name"):
            raise ValueError(f"模型名已存在: {model.get('name')}")
    data[kind].append(model)
    save_models(data)
    return model


def delete_model(kind: str, name: str) -> bool:
    """删除模型"""
    data = load_models()
    before = len(data.get(kind, []))
    data[kind] = [m for m in data.get(kind, []) if m.get("name") != name]
    # 若删除的是当前选中模型，清空 current 指向
    if (data.get("current") or {}).get(kind) == name:
        data.setdefault("current", {})[kind] = ""
    save_models(data)
    return len(data[kind]) < before


def update_model(kind: str, old_name: str, model: dict) -> dict:
    """编辑模型（支持改名）。

    Args:
        kind:     "llm" / "embedding" / "reranker" / "tool_llm" / "summary"
        old_name: 原模型名（用于定位待编辑的模型）
        model:    编辑后的完整模型配置（name 为新的名称）

    Returns:
        编辑后的模型配置
    """
    data = load_models()
    models = data.setdefault(kind, [])
    new_name = model.get("name")

    for i, m in enumerate(models):
        if m.get("name") == old_name:
            # 重名检查（排除自身，允许不改名）
            for j, other in enumerate(models):
                if j != i and other.get("name") == new_name:
                    raise ValueError(f"模型名已存在: {new_name}")
            models[i] = model
            # 改名时同步 current 指向
            if (data.get("current") or {}).get(kind) == old_name:
                data.setdefault("current", {})[kind] = new_name
            save_models(data)
            return model

    raise ValueError(f"模型不存在: {old_name}")


# ============================================================================
# 当前选中模型（current）持久化
# ============================================================================

def get_current(kind: str) -> Optional[str]:
    """
    获取指定类型的当前选中模型名（持久化在 models.json 的 current 字段）。

    Args:
        kind: "llm" / "embedding" / "reranker"

    Returns:
        当前选中模型名；未设置或不存在返回 None。
    """
    data = load_models()
    name = (data.get("current") or {}).get(kind)
    if name and get_model_by_name(kind, name):
        return name
    return None


def set_current(kind: str, name: str) -> bool:
    """
    设置指定类型的当前选中模型名（持久化）。

    Args:
        kind: "llm" / "embedding" / "reranker"
        name: 模型名
    """
    data = load_models()
    data.setdefault("current", {})
    data["current"][kind] = name
    save_models(data)
    return True


# ============================================================================
# 当前选中数据库（current_db）持久化
# ============================================================================

def get_current_db() -> Optional[str]:
    """
    获取当前选中的数据库名（持久化在 db.json 的 current 字段）。

    Returns:
        当前选中数据库名；未设置或不存在返回 None。
    """
    data = load_dbs()
    name = data.get("current")
    if name and get_db_by_name(name):
        return name
    return None


def set_current_db(name: str) -> bool:
    """
    设置当前选中的数据库名（持久化）。

    Args:
        name: 数据库名
    """
    data = load_dbs()
    data["current"] = name
    save_dbs(data)
    return True


# ============================================================================
# 数据库配置管理
# ============================================================================

def _default_dbs() -> dict:
    return {"milvus": []}


def load_dbs() -> dict:
    """加载全部数据库配置（内存缓存，写操作后自动刷新）。

    返回深拷贝，隔离调用方对返回值的修改，避免污染缓存（与原「每次读文件」
    返回新对象的隔离语义一致）。
    """
    global _dbs_cache
    with _lock:
        if _dbs_cache is None:
            _dbs_cache = _read_json(DB_FILE, _default_dbs())
        return copy.deepcopy(_dbs_cache)


def save_dbs(data: dict):
    """保存全部数据库配置（同时刷新内存缓存）"""
    global _dbs_cache
    with _lock:
        _write_json(DB_FILE, data)
        _dbs_cache = data


def list_dbs() -> List[dict]:
    """列出所有数据库配置"""
    return load_dbs().get("milvus", [])


def get_db_by_name(name: str) -> Optional[dict]:
    """按名称查找数据库"""
    for d in list_dbs():
        if d.get("name") == name:
            return d
    return None


def add_db(db: dict) -> dict:
    """新增数据库"""
    data = load_dbs()
    for d in data.get("milvus", []):
        if d.get("name") == db.get("name"):
            raise ValueError(f"数据库名已存在: {db.get('name')}")
    data.setdefault("milvus", []).append(db)
    save_dbs(data)
    return db


def delete_db(name: str) -> bool:
    """删除数据库"""
    data = load_dbs()
    before = len(data.get("milvus", []))
    data["milvus"] = [d for d in data.get("milvus", []) if d.get("name") != name]
    save_dbs(data)
    return len(data["milvus"]) < before
