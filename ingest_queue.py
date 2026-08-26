# -*- coding: utf-8 -*-
"""
================================================================================
异步入库任务队列 — 任务表（SQLite）+ Celery 任务分派（Phase 2）
================================================================================

职责：上传接口只「提交任务 + 查状态」，解析 / 切分 / 向量化 / 入库全部在
Celery Worker 进程中异步执行，绝不阻塞 HTTP 请求。

状态机（单向流转，不倒退）：
    PENDING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> DONE
                   \\                                          /
                    \\__________________ FAILED ______________/

进度语义：
  - progress 仅在 EMBEDDING 阶段按 batch 精确更新（其余阶段为 0）。
  - deepdoc 解析阶段无进度回调，故 PARSING/CHUNKING/INDEXING 只更新 stage。

设计说明：
  - 本模块不直接 import db_service（避免循环依赖 + 便于单测），
    实际执行函数通过 set_executor() 注入（Celery Worker 启动时注入 ingest_file）。
  - 任务分派通过 _dispatch()：惰性 import celery_app 并 delay；进程重启后的
    任务恢复由 Celery（task_acks_late + task_reject_on_worker_lost）负责。
  - 任务表独立 SQLite 文件 data/ingest_tasks.db，不污染 tree_store 检索库。
================================================================================
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# 状态常量
# ---------------------------------------------------------------------------
PENDING = "PENDING"
PARSING = "PARSING"
CHUNKING = "CHUNKING"
EMBEDDING = "EMBEDDING"
INDEXING = "INDEXING"
DONE = "DONE"
FAILED = "FAILED"

# 「运行中」状态（用于同名冲突检测）
_RUNNING_STATES = (PENDING, PARSING, CHUNKING, EMBEDDING, INDEXING)

# ---------------------------------------------------------------------------
# 路径与 schema
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DB_PATH = os.path.join(_DATA_DIR, "ingest_tasks.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_task (
    task_id     TEXT PRIMARY KEY,
    source      TEXT,
    enhance     INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'PENDING',
    progress    INTEGER DEFAULT 0,
    error       TEXT,
    doc_id      TEXT,
    stats       TEXT,
    file_path   TEXT,
    created_at  INTEGER,
    updated_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ingest_source ON ingest_task(source);
CREATE INDEX IF NOT EXISTS idx_ingest_status ON ingest_task(status);
"""

# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------
_lock = threading.RLock()

# 可注入的任务执行函数：fn(file_path, source, enhance, on_progress) -> dict
# on_progress(stage: str, progress: int = 0)
_execute_task: Optional[Callable] = None


def _now() -> int:
    return int(time.time() * 1000)


def _get_conn() -> sqlite3.Connection:
    os.makedirs(_DATA_DIR, exist_ok=True)
    # WAL + busy_timeout：多进程（多 Celery worker）并发写任务表时避免「database is
    # locked」。threading.RLock 只保证单进程内互斥，跨进程需靠 SQLite 自身的
    # WAL 与 busy_timeout（写锁竞争时等待 10s 而非立即报错）。
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def set_executor(fn: Callable):
    """注入实际执行函数（由 server 启动时注入 ingest_file）。"""
    global _execute_task
    with _lock:
        _execute_task = fn


# ---------------------------------------------------------------------------
# 任务 CRUD
# ---------------------------------------------------------------------------
def _create_task(source: str, enhance: bool, file_path: str) -> str:
    """创建 PENDING 任务（不入队），返回 task_id。"""
    task_id = uuid.uuid4().hex
    now = _now()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO ingest_task (task_id, source, enhance, status, progress, "
                "file_path, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (task_id, source, 1 if enhance else 0, PENDING, 0, file_path, now, now),
            )
            conn.commit()
        finally:
            conn.close()
    return task_id


def get_task(task_id: str) -> Optional[dict]:
    """查询任务，返回 dict（含解析后的 stats），不存在返回 None。"""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT * FROM ingest_task WHERE task_id = ?", (task_id,)).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    d = dict(row)
    d["enhance"] = bool(d["enhance"])
    if d.get("stats"):
        try:
            d["stats"] = json.loads(d["stats"])
        except Exception:
            pass
    return d


def list_tasks(limit: int = 50) -> List[dict]:
    """列出最近任务（按创建时间倒序）。"""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM ingest_task ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
    out = []
    for row in rows:
        d = dict(row)
        d["enhance"] = bool(d["enhance"])
        if d.get("stats"):
            try:
                d["stats"] = json.loads(d["stats"])
            except Exception:
                pass
        out.append(d)
    return out


def has_running(source: str) -> bool:
    """是否存在同名 source 的「运行中」任务（用于上传时的 409 冲突检测）。"""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM ingest_task WHERE source = ? AND status IN (%s) LIMIT 1"
                % ",".join("?" * len(_RUNNING_STATES)),
                (source, *_RUNNING_STATES),
            ).fetchone()
        finally:
            conn.close()
    return row is not None


def retry_task(task_id: str) -> bool:
    """将 FAILED 任务重置为 PENDING 并重新入队；返回是否成功。"""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT status, source, enhance, file_path FROM ingest_task WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            if row["status"] != FAILED:
                return False
            conn.execute(
                "UPDATE ingest_task SET status = ?, error = NULL, progress = 0, updated_at = ? "
                "WHERE task_id = ?",
                (PENDING, _now(), task_id),
            )
            conn.commit()
        finally:
            conn.close()

    _dispatch(task_id)
    return True


def delete_task(task_id: str) -> bool:
    """删除终态（FAILED/DONE）任务记录及其步骤产物、上传文件。

    运行中任务（PENDING/PARSING/CHUNKING/EMBEDDING/INDEXING）不可删，避免与
    正在执行的 Worker 产生竞态。删除成功后清理中间产物与上传文件。

    Returns:
        是否删除成功（任务不存在 / 运行中 → False）。
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT status, file_path FROM ingest_task WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            if row["status"] in _RUNNING_STATES:
                return False
            conn.execute("DELETE FROM ingest_task WHERE task_id = ?", (task_id,))
            conn.commit()
            file_path = row["file_path"]
        finally:
            conn.close()

    # 清理步骤产物与上传文件（事务外，清理失败不影响删除结果）
    cleanup_steps(task_id)
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
    return True


def interrupt_running(reason: str = "服务已关闭，任务中断") -> int:
    """把「运行中」任务标记为 FAILED（服务关闭时调用，避免任务卡死在中间态）。

    服务（Worker）关闭时，正在 PARSING/CHUNKING/EMBEDDING/INDEXING 的任务会被
    中断，若不做处理，其状态会永久卡在中间态（前端显示「向量化中…」且无从恢复）。
    这里统一标记为 FAILED 并附中断原因，重启后用户可点「重试」从失败步骤继续。

    Returns:
        被中断的任务数量。
    """
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT task_id FROM ingest_task WHERE status IN (%s)"
                % ",".join("?" * len(_RUNNING_STATES)),
                (*_RUNNING_STATES,),
            ).fetchall()
            ids = [r["task_id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE ingest_task SET status = ?, error = ?, updated_at = ? "
                    f"WHERE task_id IN ({placeholders})",
                    (FAILED, reason, _now(), *ids),
                )
                conn.commit()
            return len(ids)
        finally:
            conn.close()


def submit(source: str, enhance: bool, file_path: str) -> str:
    """创建 PENDING 任务并分派到 Celery，立即返回 task_id（不等待执行完成）。

    Raises:
        分派失败（Celery/Redis 不可用）时抛出，此时任务已被标记 FAILED。
    """
    task_id = _create_task(source, enhance, file_path)
    try:
        _dispatch(task_id)
    except Exception as e:
        _update(task_id, status=FAILED, error=f"任务分派失败（Celery 不可用）: {e}")
        raise
    return task_id


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否可自动重试。

    可重试   ：网络超时/连接失败、限流(429)、服务端错误(5xx)、服务不可用。
    不可重试 ：余额不足(402)、参数无效(400)、文件不存在/损坏、内容为空。
    """
    # 明确不可重试的业务/文件异常
    if isinstance(exc, ValueError):
        return False
    if isinstance(exc, FileNotFoundError):
        return False

    msg = str(exc)
    if any(k in msg for k in ("402", "balance", "insufficient", "400", "invalid parameter")):
        return False

    # 网络 / 超时 / 限流 / 服务端错误可重试
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(k in msg for k in ("429", "500", "502", "503", "504",
                                  "timeout", "timed out", "connection", "refused"))


def _dispatch(task_id: str):
    """把任务分派到对应步骤的 Celery 任务（Phase 3 步骤化，惰性 import 避免循环依赖）。

    按步骤产物落盘情况决定起点（见 _current_step），从而天然支持「步骤级重跑」：
    retry 时前置步骤产物仍在，直接跳到失败步骤重新执行，无需重跑 parse/chunk/embed。
    """
    from celery_app import parse_document, chunk_document, embed_document, index_document
    step = _current_step(task_id)
    _STEP_TASKS = {
        STEP_PARSE: parse_document,
        STEP_CHUNK: chunk_document,
        STEP_EMBED: embed_document,
        STEP_INDEX: index_document,
    }
    _STEP_TASKS[step].delay(task_id)


# ---------------------------------------------------------------------------
# 步骤产物落盘（Phase 3 步骤级重跑）
# ---------------------------------------------------------------------------
# 每个步骤的产物落盘为 data/steps/{task_id}/{step}_result.json，供下一步骤读取。
# 作用：
#   1. 步骤级重跑：某步失败重跑时，前置步骤产物仍在，直接跳过（幂等）；
#   2. 跨队列传递：parse/chunk/index 在 parse_queue，embed 在 embedding_queue，
#      不同 worker 进程间通过落盘产物交接，而非内存对象。
# 原子写：先写 .tmp 再 os.replace，worker 崩溃不会留下半写文件。

STEP_PARSE = "parse"
STEP_CHUNK = "chunk"
STEP_EMBED = "embed"
STEP_INDEX = "index"

_STEPS_DIR = os.path.join(_DATA_DIR, "steps")


def _steps_dir(task_id: str) -> str:
    return os.path.join(_STEPS_DIR, task_id)


def save_step(task_id: str, step_name: str, data) -> str:
    """写步骤产物（JSON，原子写），返回文件路径。"""
    d = _steps_dir(task_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{step_name}_result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load_step(task_id: str, step_name: str):
    """读步骤产物，不存在返回 None。"""
    path = os.path.join(_steps_dir(task_id), f"{step_name}_result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_step(task_id: str, step_name: str) -> bool:
    """步骤产物是否已落盘（幂等跳过的判断依据）。"""
    return os.path.exists(os.path.join(_steps_dir(task_id), f"{step_name}_result.json"))


def cleanup_steps(task_id: str) -> None:
    """删除某任务的整个步骤产物目录（DONE 后清理）。"""
    import shutil
    d = _steps_dir(task_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def _current_step(task_id: str) -> str:
    """根据步骤产物落盘情况返回当前应执行的步骤名。

    用于步骤级重跑：产物落盘是各步骤的幂等标记，缺失即从该步骤开始。
      - 无 parse 产物  → parse
      - 有 parse 无 chunk → chunk
      - 有 chunk 无 embed → embed
      - 有 embed        → index
    """
    if not has_step(task_id, STEP_PARSE):
        return STEP_PARSE
    if not has_step(task_id, STEP_CHUNK):
        return STEP_CHUNK
    if not has_step(task_id, STEP_EMBED):
        return STEP_EMBED
    return STEP_INDEX


def _update(task_id: str, **fields):
    """更新任务字段（自动刷新 updated_at）。"""
    fields = {k: v for k, v in fields.items() if k in (
        "status", "progress", "error", "doc_id", "stats", "file_path")}
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [task_id]
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(f"UPDATE ingest_task SET {cols} WHERE task_id = ?", vals)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 任务执行（由 Celery Worker 调用）
# ---------------------------------------------------------------------------
def _run_task(task_id: str, raise_on_retryable: bool = False):
    """执行单个任务（由 Celery Worker 的 ingest_worker 调用）。

    Args:
        task_id:            任务主键。
        raise_on_retryable: True 时可重试异常重新抛出（供 Celery retry）；
                           False 时所有异常均标记 FAILED（供单元测试直接调用）。
    """
    task = get_task(task_id)
    if task is None:
        return
    source = task["source"]
    enhance = bool(task["enhance"])
    file_path = task.get("file_path") or ""

    def on_progress(stage: str, progress: int = 0):
        _update(task_id, status=stage, progress=progress)

    try:
        if _execute_task is None:
            raise RuntimeError("任务执行函数未注入（未调用 set_executor）")
        result = _execute_task(file_path, source, enhance, on_progress) or {}
        stats = {k: v for k, v in result.items() if k != "doc_id"}
        _update(
            task_id,
            status=DONE,
            progress=100,
            doc_id=result.get("doc_id", ""),
            stats=json.dumps(stats, ensure_ascii=False),
        )
    except Exception as e:
        if raise_on_retryable and is_retryable(e):
            raise  # 传播给 Celery retry（不标记 FAILED，保留文件供重试）
        _update(task_id, status=FAILED, error=str(e))
    finally:
        # DONE 后清理上传文件；FAILED / 重试中间态保留文件
        final = get_task(task_id)
        if final and final["status"] == DONE and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
