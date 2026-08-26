# -*- coding: utf-8 -*-
"""
================================================================================
Celery 应用 + 入库任务（Phase 2 异步任务队列）
================================================================================

架构：
  FastAPI 主进程只负责「提交任务」（celery.delay），解析/切分/向量化/入库
  全部在独立的 Celery Worker 进程中执行，主进程绝不等待任务完成。

  ┌──────────────┐   delay    ┌──────────┐   执行    ┌───────────────┐
  │   FastAPI    │ ─────────→ │  Redis   │ ───────→ │ Celery Worker │
  │  submit()    │            │ (broker) │          │  ingest_file  │
  └──────────────┘            └──────────┘          └───────┬───────┘
                                                             │ 回调 on_progress
                                                             ▼
                                                    SQLite 任务表（状态）

启动 Worker（独立进程，与主服务并行）：
  poetry run celery -A celery_app worker --loglevel=info --pool=solo
  （Windows 不支持 prefork，用 solo；任务多为 IO 密集也可用 --pool=threads）

关键点：
  1. 本文件顶部先 import torch（最浅调用栈），否则 deepdoc/summarize 触发
     的 torch 二次 import 会报 "partially initialized module 'torch'"。
  2. Worker 进程 import 本模块时注入执行函数（ingest_file），并触发
     db_service 的 torch 深加载容错 patch。
  3. 任务通过 ingest_queue 的 SQLite 任务表上报状态，状态查询接口与
     Phase 1 完全一致（不感知 Celery）。
================================================================================
"""

# ---------------------------------------------------------------------------
# 关键修复：torch 必须作为 worker 进程的最浅栈第一个 import 完整加载。
# 否则 deepdoc(PDF)/summarize(langchain_openai)/reranker(transformers)
# 在深层调用栈触发 torch 二次 import 会崩溃。
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json
import os

from celery import Celery

import ingest_queue
from config_loader import cfg


# ---------------------------------------------------------------------------
# 配置常量（从 config/config.json 的 ingest 块读取，代码不硬编码）
# ---------------------------------------------------------------------------
_MAX_RETRIES = int(cfg("ingest.max_retries", 3))
_RETRY_BACKOFF_BASE = int(cfg("ingest.retry_backoff_base", 5))
_PARSE_QUEUE = cfg("ingest.parse_queue", "parse_queue")
_EMBEDDING_QUEUE = cfg("ingest.embedding_queue", "embedding_queue")


def _inject_executor():
    """Worker 进程启动时注入入库执行函数。

    同时触发 db_service 的 import（其顶部有 torch 深加载容错 patch）。
    """
    try:
        from db_service import ingest_file
        ingest_queue.set_executor(ingest_file)
    except Exception as e:  # pragma: no cover
        print(f"[celery_app] 注入执行函数失败: {e}")


_inject_executor()

celery_app = Celery(
    "rag_ingest",
    broker=cfg("ingest.redis_broker", "redis://127.0.0.1:6379/0"),
    backend=cfg("ingest.redis_backend", "redis://127.0.0.1:6379/1"),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    # 关键：任务确认后置 + worker 崩溃重新投递（进程重启自动恢复）
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # 每个 worker 一次只取一个任务
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    # worker 崩溃后，未 ACK 任务的重新投递等待时间（秒）。
    # 默认 3600s 太长；设为 600s 兼顾「崩溃快速恢复」与「长任务不被误重投」。
    # 入库任务（解析+向量化）通常 < 10 分钟，故 600s 是安全下限。
    broker_transport_options={"visibility_timeout": int(cfg("ingest.visibility_timeout", 600))},
)

# ---------------------------------------------------------------------------
# Phase 3 双队列路由：parse/chunk/index 进 parse_queue（CPU 密集），
# embed 进 embedding_queue（网络 IO 密集）。默认关闭（dual_queue=false），
# 所有步骤进默认 celery 队列，单 worker 即可消费（向后兼容 Phase 2）。
# 开启后启动两个 worker：
#   celery -A celery_app worker -Q <parse_queue> --pool=solo
#   celery -A celery_app worker -Q <embedding_queue> --pool=threads --concurrency=4
# 修改 Worker 数量/队列只改配置与启动参数，无需改 API 代码。
# ---------------------------------------------------------------------------
if bool(cfg("ingest.dual_queue", False)):
    celery_app.conf.task_routes = {
        "rag.parse_document": {"queue": _PARSE_QUEUE},
        "rag.chunk_document": {"queue": _PARSE_QUEUE},
        "rag.embed_document": {"queue": _EMBEDDING_QUEUE},
        "rag.index_document": {"queue": _PARSE_QUEUE},
    }


def _retry_step(self, exc):
    """可重试异常的指数退避重试（base/base*2/base*4）。"""
    countdown = _RETRY_BACKOFF_BASE * (2 ** self.request.retries)
    raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(bind=True, name="rag.ingest_worker", max_retries=_MAX_RETRIES)
def ingest_worker(self, task_id: str):
    """一体化 Celery 入库任务（Phase 2 保留，作 fallback）：复用 _run_task。

    失败重试策略：
      - 可重试异常（网络超时/连接失败/限流/5xx）→ 指数退避重试（5s/10s/20s）
      - 不可重试异常（余额 402/参数 400/文件问题/空内容）→ _run_task 已标记
        FAILED 且不抛出，不会走到 retry。

    Args:
        task_id: ingest_queue 任务表的主键。
    """
    try:
        ingest_queue._run_task(task_id, raise_on_retryable=True)
    except Exception as exc:
        # 只有可重试异常会传播到这里（不可重试异常已在 _run_task 内吞掉）
        countdown = _RETRY_BACKOFF_BASE * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


# ---------------------------------------------------------------------------
# Phase 3 步骤化任务（parse → chunk → embed → index 链式）
# ---------------------------------------------------------------------------
# 每个步骤：
#   1. 幂等跳过：本步产物已落盘 → 直接 dispatch 下一步（步骤级重跑的关键）；
#   2. 前置产物缺失 → 回退 dispatch 前置步骤（异常状态自愈）；
#   3. 执行步骤函数 → 落盘产物 → dispatch 下一步；
#   4. 可重试异常 → 指数退避重试当前步骤；不可重试异常 → 标记 FAILED（保留产物）。
# ---------------------------------------------------------------------------

def _step_progress(task_id: str):
    """返回该任务的进度回调（更新任务表 stage/progress）。"""
    def on_progress(stage: str, progress: int = 0):
        ingest_queue._update(task_id, status=stage, progress=progress)
    return on_progress


def _step_fail(self, task_id: str, exc: Exception):
    """可重试则 retry，否则标记 FAILED（保留产物供步骤级重跑）。"""
    if ingest_queue.is_retryable(exc):
        _retry_step(self, exc)
        return  # _retry_step 总是 raise；此处 return 仅作语义防御
    ingest_queue._update(task_id, status=ingest_queue.FAILED, error=str(exc))


def _load_or_fallback(task_id: str, dep_step: str, dep_task):
    """读前置步骤产物；缺失则回退 dispatch 前置步骤并返回 None。"""
    if not ingest_queue.has_step(task_id, dep_step):
        dep_task.delay(task_id)
        return None
    return ingest_queue.load_step(task_id, dep_step)


@celery_app.task(bind=True, name="rag.parse_document", max_retries=_MAX_RETRIES)
def parse_document(self, task_id: str):
    """步骤 1（PARSING）：解析文件 → 落盘 parse_result → dispatch chunk。"""
    task = ingest_queue.get_task(task_id)
    if task is None:
        return
    source = task["source"]
    enhance = bool(task["enhance"])
    file_path = task.get("file_path") or ""

    try:
        if not ingest_queue.has_step(task_id, ingest_queue.STEP_PARSE):
            from db_service import parse_step
            pr = parse_step(file_path, source, enhance, on_progress=_step_progress(task_id))
            ingest_queue.save_step(task_id, ingest_queue.STEP_PARSE, pr)
        chunk_document.delay(task_id)
    except Exception as exc:
        _step_fail(self, task_id, exc)


@celery_app.task(bind=True, name="rag.chunk_document", max_retries=_MAX_RETRIES)
def chunk_document(self, task_id: str):
    """步骤 2（CHUNKING）：读 parse_result → 切分 → 落盘 chunk_result → dispatch embed。"""
    task = ingest_queue.get_task(task_id)
    if task is None:
        return
    source = task["source"]
    enhance = bool(task["enhance"])

    try:
        if not ingest_queue.has_step(task_id, ingest_queue.STEP_CHUNK):
            pr = _load_or_fallback(task_id, ingest_queue.STEP_PARSE, parse_document)
            if pr is None:
                return  # 回退到 parse，本轮结束
            from db_service import chunk_step
            cr = chunk_step(pr, source, enhance, on_progress=_step_progress(task_id))
            ingest_queue.save_step(task_id, ingest_queue.STEP_CHUNK, cr)
        embed_document.delay(task_id)
    except Exception as exc:
        _step_fail(self, task_id, exc)


@celery_app.task(bind=True, name="rag.embed_document", max_retries=_MAX_RETRIES)
def embed_document(self, task_id: str):
    """步骤 3（EMBEDDING）：读 chunk_result → 向量化 → 落盘 embedded_result → dispatch index。"""
    task = ingest_queue.get_task(task_id)
    if task is None:
        return
    enhance = bool(task["enhance"])

    try:
        if not ingest_queue.has_step(task_id, ingest_queue.STEP_EMBED):
            cr = _load_or_fallback(task_id, ingest_queue.STEP_CHUNK, chunk_document)
            if cr is None:
                return  # 回退到 chunk，本轮结束
            from db_service import embed_step
            er = embed_step(cr, enhance, on_progress=_step_progress(task_id))
            ingest_queue.save_step(task_id, ingest_queue.STEP_EMBED, er)
        index_document.delay(task_id)
    except Exception as exc:
        _step_fail(self, task_id, exc)


@celery_app.task(bind=True, name="rag.index_document", max_retries=_MAX_RETRIES)
def index_document(self, task_id: str):
    """步骤 4（INDEXING）：去重 → 写树库 + Milvus → 标记 DONE → 清理产物与文件。"""
    task = ingest_queue.get_task(task_id)
    if task is None:
        return
    source = task["source"]
    enhance = bool(task["enhance"])
    file_path = task.get("file_path") or ""

    try:
        er = _load_or_fallback(task_id, ingest_queue.STEP_EMBED, embed_document)
        if er is None:
            return  # 回退到 embed，本轮结束

        from db_service import index_step, delete_source
        # 写库前去重（幂等：每次 index 都删旧数据，步骤级重跑安全）
        delete_source(source)
        stats = index_step(er, source, enhance, on_progress=_step_progress(task_id))

        doc_id = stats.get("doc_id", "")
        # 先清理中间产物 + 上传文件，再标记 DONE：保证「DONE」对外可见时产物已清理，
        # 无「前端/测试看到 DONE 但 steps 目录仍在」的时序竞态。
        ingest_queue.cleanup_steps(task_id)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        ingest_queue._update(
            task_id,
            status=ingest_queue.DONE,
            progress=100,
            doc_id=doc_id,
            stats=json.dumps({k: v for k, v in stats.items() if k != "doc_id"}, ensure_ascii=False),
        )
    except Exception as exc:
        _step_fail(self, task_id, exc)
