"""
================================================================================
对话历史存储 — SQLite（与知识库数据隔离）
================================================================================

将对话历史持久化到本地 SQLite 文件（data/chat_history.db），与用户上传的
知识库数据（Milvus / 文档树）完全分开。主程序直接调用。

结构：
  - 表「conversations」：所有对话的元信息（标题/创建时间/消息数）
  - 表「messages」    ：所有消息（role/content/ts），按 conversation_id 关联

对外接口（签名与返回值与旧 Milvus 版完全一致，调用方无需改动）：
  create_conversation(title)         → 新对话
  add_message(conversation_id, role, content) → 写入一条消息
  list_conversations()               → 对话列表（按创建时间倒序）
  get_conversation(conversation_id)  → 对话完整历史（按时间升序）
  get_title(conversation_id)         → 对话标题
  update_title(conversation_id, title) → 更新标题
  delete_conversation(conversation_id) → 删除对话

说明：
  对话历史是纯元数据、不参与向量检索，用 SQLite 存储比向量库更合适（旧版用
  Milvus 零向量占位，是「拿向量库当 KV 库」的反模式）。SQLite 版本：
    - 零额外依赖、本地文件、随项目目录走；
    - WAL + busy_timeout 支持多线程并发读写（add_message 在后台线程调用）；
    - message_count 用 UPDATE 原子累加，无需「读后写」的竞态窗口。

依赖：
  仅标准库 sqlite3。
================================================================================
"""

import os
import sqlite3
import time
import uuid

# 数据库文件位置：项目根目录 data/chat_history.db（与 doc_tree.db、ingest_tasks.db 同级）
_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "chat_history.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    ts              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ts);
"""


def _get_conn() -> sqlite3.Connection:
    """新建连接并确保表结构存在。

    每次操作新建连接（而非复用全局连接）：SQLite 连接对象本身不跨线程共享，
    这样 add_message 在后台线程调用时最安全。WAL + busy_timeout 处理多线程并发写
    （写锁竞争时等待而非立即报 database is locked）。
    """
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


# ============================================================================
# 对外接口（签名与返回结构同旧版）
# ============================================================================

def create_conversation(title: str = "") -> dict:
    """创建新对话，返回 {conversation_id, title, created_at}。"""
    conv_id = uuid.uuid4().hex[:16]
    now = int(time.time() * 1000)
    title = (title or "").strip() or "新对话"

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO conversations (conversation_id, title, created_at, message_count) "
            "VALUES (?, ?, ?, 0)",
            (conv_id, title, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"conversation_id": conv_id, "title": title, "created_at": now}


def add_message(conversation_id: str, role: str, content: str) -> dict:
    """写入一条对话消息（role: user / assistant），并原子累加消息数。"""
    now = int(time.time() * 1000)

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET message_count = message_count + 1 "
            "WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "conversation_id": conversation_id, "role": role}


def list_conversations() -> list:
    """列出所有对话（按创建时间倒序）。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT conversation_id, title, created_at, message_count "
            "FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> list:
    """获取对话完整历史（按时间升序；同毫秒按 id 保证顺序稳定）。"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY ts ASC, id ASC",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_title(conversation_id: str) -> str:
    """获取指定对话的标题（不存在返回空字符串）。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT title FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    finally:
        conn.close()
    return row["title"] if row else ""


def update_title(conversation_id: str, title: str) -> bool:
    """更新指定对话的标题（用于首轮对话后生成摘要标题）。"""
    title = (title or "").strip()
    if not title:
        return False

    conn = _get_conn()
    try:
        cur = conn.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (title, conversation_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_conversation(conversation_id: str) -> bool:
    """删除指定对话（消息 + 元信息）。"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()
    return True
