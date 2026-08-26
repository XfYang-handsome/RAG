"""
================================================================================
FastAPI 服务 — HTTP API + Web 管理界面
================================================================================

核心接口：
  GET  /                           Web 管理界面
  POST /upload                     上传文件 → 向量化 → 入库
  POST /chat                       对话查询（SSE 流式输出）
  GET  /health                     健康检查
  GET  /config                     获取当前模型/数据库配置
  POST /config/select              切换当前使用的模型/数据库
  GET  /models                     列出所有已保存模型
  POST /models                     新增模型
  DELETE /models/{kind}/{name}     删除模型
  GET  /dbs                        列出所有已保存数据库
  POST /dbs                        新增数据库
  DELETE /dbs/{name}               删除数据库
  GET  /local/databases            列出本地 Milvus 所有 database
  POST /local/databases            在本地 Milvus 新建 database
  GET  /local/parents              列出当前数据库所有父块
  DELETE /local/clear              清空当前数据库

启动方式：
  python __main__.py                 # 默认 127.0.0.1:8000
  python __main__.py --port 8080 --host 0.0.0.0

模型/数据库配置保存在 models.json / db.json，通过 Web 界面动态管理。
================================================================================
"""

import os, sys, json, uuid, asyncio
import threading
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import create_chat_model, DEFAULT_ANSWER_TEMPERATURE
from reranker import Reranker
from rag_graph import RAGGraph

from config_loader import config
import store_config
import ingest_queue

# ============================================================================
# 全局组件（懒加载 + 单例，可动态切换）
# ============================================================================

_llm                = None   # LLM 聊天模型（生成模型，最终回答）
_reranker           = None   # Reranker 重排序
_tool_llm           = None   # 工具决策模型（可选，绑定工具决定调用哪些工具）
_rewrite_llm        = None   # 查询重写模型（可选，未配置则 rewrite 节点透传原问题）

# 组件懒加载锁（RLock：get_tool_llm 回退 get_llm 会重入，避免死锁）
_components_lock = threading.RLock()

# 当前选中的模型名称（从 store_config 的 current 字段读取，持久化，未设置时用第一个）
# 注：数据库 / embedding 已迁移到 MCP 端管理，不再由主程序持有状态。
_current_llm_name: Optional[str] = store_config.get_current("llm")
_current_rerank_name: Optional[str] = store_config.get_current("reranker")
_current_tool_llm_name: Optional[str] = store_config.get_current("tool_llm")

# Reranker 本地模型下载/加载状态（用于前端"下载并加载"功能）
_reranker_load_state = {
    "status": "idle",        # idle=未加载, loading=加载中, loaded=已加载, error=失败
    "message": "",
}

# ============================================================================
# 服务日志（环形缓冲，供前端 /logs 查看）
# ============================================================================
import collections
import time as _time

_log_buffer = collections.deque(maxlen=500)


def log(level: str, msg: str):
    """记录一条服务日志（同时打印到控制台）。"""
    entry = {
        "ts": _time.strftime("%H:%M:%S"),
        "level": level,
        "msg": str(msg),
    }
    _log_buffer.append(entry)
    print(f"[{entry['ts']}] [{level}] {msg}")


# ============================================================================
# 当前模型/数据库解析
# ============================================================================

def _get_current_model(kind: str, name: Optional[str]) -> Optional[dict]:
    """获取指定类型的当前模型配置（按 name 或取第一个）"""
    models = store_config.list_models(kind)
    if not models:
        return None
    if name:
        for m in models:
            if m.get("name") == name:
                return m
    return models[0]  # 默认第一个


# ============================================================================
# 懒加载工厂函数（可动态重建）
# ============================================================================

def get_llm():
    """获取 LLM 实例（按当前配置懒加载，线程安全）"""
    global _llm
    with _components_lock:
        if _llm is None:
            m = _get_current_model("llm", _current_llm_name)
            if m is None:
                raise RuntimeError("未配置 LLM 模型")
            _llm = create_chat_model(
                model=m.get("model"),
                base_url=m.get("base_url"),
                api_key=m.get("api_key"),
                protocol=m.get("protocol", "openai"),
                temperature=m.get("temperature", DEFAULT_ANSWER_TEMPERATURE),
            )
    return _llm


def get_tool_llm():
    """
    获取工具决策模型实例（按当前配置懒加载，线程安全）。

    优先使用 tool_llm 类型配置的模型；未配置时回退到生成模型 LLM。
    该模型需支持 function calling（OpenAI 兼容），用于绑定工具决定调用哪些工具。
    """
    global _tool_llm
    with _components_lock:
        if _tool_llm is not None:
            return _tool_llm
        m = _get_current_model("tool_llm", _current_tool_llm_name)
        if m is None:
            # 未配置工具决策模型：回退到生成模型 LLM，但用温度 0（工具决策需确定性，
            # 不能复用 get_llm() 的 0.7 创造性温度，否则工具调用会不稳定）
            base = _get_current_model("llm", _current_llm_name)
            if base is None:
                return get_llm()
            return create_chat_model(
                model=base.get("model"),
                base_url=base.get("base_url"),
                api_key=base.get("api_key"),
                protocol=base.get("protocol", "openai"),
                temperature=0.0,
                disable_thinking=True,
            )
        _tool_llm = create_chat_model(
            model=m.get("model"),
            base_url=m.get("base_url"),
            api_key=m.get("api_key"),
            protocol=m.get("protocol", "openai"),
            disable_thinking=True,
        )
    return _tool_llm


def get_rewrite_llm():
    """
    获取查询重写模型实例（按当前配置懒加载，线程安全）。

    从 models.json 的 rewrite 字段读取独立的重写模型；未配置时返回 None，
    此时 RAGGraph 的 rewrite 节点会直接透传原问题（不调用 LLM 重写），
    避免用 DeepSeek V4 Pro 等 reasoning 模型做重写导致慢 + 改坏查询。
    """
    global _rewrite_llm
    with _components_lock:
        if _rewrite_llm is not None:
            return _rewrite_llm
        m = _get_current_model("rewrite", None)
        if m is None:
            return None
        _rewrite_llm = create_chat_model(
            model=m.get("model"),
            base_url=m.get("base_url"),
            api_key=m.get("api_key"),
            protocol=m.get("protocol", "openai"),
        )
    return _rewrite_llm


def get_reranker():
    """获取 Reranker 实例（按当前配置懒加载，线程安全）"""
    global _reranker
    with _components_lock:
        if _reranker is None:
            m = _get_current_model("reranker", _current_rerank_name)
            if m is None:
                raise RuntimeError("未配置 Reranker 模型")
            if m.get("type") == "local":
                _reranker = Reranker(local_model_path=m.get("model_path"))
            else:
                _reranker = Reranker(
                    model=m.get("model"),
                    base_url=m.get("base_url"),
                    api_key=m.get("api_key"),
                    online=True,
                )
    return _reranker


def _reset_components():
    """重置 LLM / Reranker / 工具决策模型 / 重写模型（切换模型后调用，线程安全）。"""
    global _llm, _reranker, _tool_llm, _rewrite_llm
    with _components_lock:
        for obj in [_llm, _reranker, _tool_llm, _rewrite_llm]:
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        _llm = None
        _reranker = None
        _tool_llm = None
        _rewrite_llm = None
        # 同步清理 llm_factory 的进程级缓存（否则 planner/evaluator 等仍持有旧模型实例，
        # 切换模型后不会立即生效，且旧实例资源不释放）
        try:
            from llm_factory import clear_cache
            clear_cache()
        except Exception:
            pass


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期。

    主程序直接连接 Milvus 数据库（检索/入库/数据管理/对话历史），
    不依赖 MCP 服务器；MCP 仅用于联网搜索工具（可选）。
    """
    log("INFO", "RAG 服务启动")
    # 启动时把上次遗留的「运行中」入库任务标记为中断：无论上次是强杀、崩溃还是
    # 手动关 Worker，任务都可能卡在 PENDING/PARSING 等中间态。统一标记为 FAILED，
    # 前端即可展示「已中断，可重试」，同名文件也不再被 has_running 误判为占用。
    try:
        import ingest_queue as _iq
        n = _iq.interrupt_running("服务重启，任务中断（可重试）")
        if n:
            log("WARN", f"检测到 {n} 个上次中断的入库任务，已标记为失败（可在前端重试）")
    except Exception as e:
        log("WARN", f"标记中断任务失败: {e}")
    # Phase 2：入库任务由独立 Celery Worker 执行，主进程只负责提交。
    # 需单独启动 worker：poetry run celery -A celery_app worker --pool=solo
    # 这里仅探测 Redis 就绪状态，给出友好提示（未就绪时上传会失败）。
    try:
        import redis as _redis
        _r = _redis.Redis(host="127.0.0.1", port=6379, socket_connect_timeout=2)
        if _r.ping():
            log("INFO", "异步入库队列（Redis）已就绪")
        _r.close()
    except Exception as e:
        log("WARN", f"Redis 未就绪，上传将失败: {e}")
    # 后台预热 Reranker 模型（首次对话时无需等待模型加载）
    _warmup_reranker_async()
    yield
    _reset_components()
    # 强制清理残留进程：Agentic 循环 / Reranker 预热等 daemon 线程在 CPU 密集
    # 操作（torch 推理 / LLM 调用）时可能卡住解释器退出，导致主程序退出后
    # 残留 python 进程。用 os._exit 绕过 daemon 线程的优雅退出等待。
    import os as _os
    log("INFO", "RAG 服务已关闭")
    _os._exit(0)


def _warmup_reranker_async():
    """
    后台线程预热 Reranker：提前加载模型到 GPU，避免首次对话等待。

    若当前配置的 reranker 是本地模型，则在后台线程触发其加载；
    加载失败不影响主流程（首次对话时仍会尝试加载）。
    """
    import threading

    def _warmup():
        global _reranker
        try:
            m = _get_current_model("reranker", _current_rerank_name)
            if m and m.get("type") == "local":
                model_path = m.get("model_path") or ""
                # 直接尝试预热：_load_model 已用 local_files_only=True 优先，
                # 本地 HF 缓存命中时直接加载（不联网），未命中才尝试联网。
                print("[INFO] 后台预热本地 Reranker 模型...")
                r = Reranker(local_model_path=model_path)
                r.rerank("预热", ["预热测试"], top_n=1)
                # 复用预热实例，避免首次对话时 get_reranker() 二次加载模型
                _reranker = r
                print("[INFO] Reranker 预热完成")
        except Exception as e:
            print(f"[INFO] Reranker 预热跳过（{e}）")

    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


app = FastAPI(title="RAG 知识库", version="4.0", lifespan=lifespan)

# ============================================================================
# 挂载 MCP 管理路由（/mcp/*：服务器列表 / 启停 / 工具 / 日志 / 开关）
#
# 架构：工具全部注册在 MCP 服务器上，主程序「不持有」任何工具定义；
# 主程序（rag_graph 的工具决策 / 确定性联网）调用工具时，通过
# mcp_service.tool_bridge 连接 MCP 执行。本路由让 MCP 管理页面
# （/mcp_page）能增删服务器、启停服务器、切换工具、查看日志。
# ============================================================================
from mcp_service.manager import create_router as _create_mcp_router
app.include_router(_create_mcp_router())


# ============================================================================
# GET / — Web 管理界面（Vue3 新前端，构建产物位于 static/dist/）
# ============================================================================
_DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dist")
_DIST_INDEX = os.path.join(_DIST_DIR, "index.html")
_UI_CACHE = None

# 挂载构建产物（JS/CSS/字体等）为静态资源
_DIST_ASSETS = os.path.join(_DIST_DIR, "assets")
if os.path.isdir(_DIST_ASSETS):
    app.mount("/assets", StaticFiles(directory=_DIST_ASSETS), name="ui-assets")

def _load_ui():
    global _UI_CACHE
    if _UI_CACHE is None:
        with open(_DIST_INDEX, "r", encoding="utf-8") as f:
            _UI_CACHE = f.read()
    return _UI_CACHE

@app.get("/")
async def root():
    return HTMLResponse(content=_load_ui())


# ============================================================================
# GET /mcp_page、/tools_page — MCP 管理页面入口（SPA 前端路由 /#/mcp）
# ============================================================================
@app.get("/mcp_page")
async def mcp_page():
    return RedirectResponse(url="/#/mcp")

@app.get("/tools_page")
async def tools_page():
    """兼容旧入口：工具管理页已替换为 MCP 管理页。"""
    return RedirectResponse(url="/#/mcp")


# ============================================================================
# POST /upload — 上传文件向量化入库（异步提交）
# ============================================================================
class UploadTaskResponse(BaseModel):
    """异步入库提交结果（立即返回，解析/向量化在后台执行）。"""
    success:  bool = True
    task_id:  str  = ""
    source:   str  = ""
    status:   str  = "PENDING"
    enhance:  bool = False
    message:  str  = ""


_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")


@app.post("/upload", response_model=UploadTaskResponse)
async def upload_file(
    file: UploadFile = File(...),
    enhance: bool = Form(False),
):
    """上传文件 → 落盘 → 提交异步入库任务 → 立即返回 task_id。

    解析 / 切分 / 向量化 / 入库全部在后台线程异步执行，接口不阻塞。
    enhance=True 时走「结构归位 → Document Tree → Retrieval Chunk」增强解析路径。
    """

    filename = os.path.basename(file.filename or "unknown")

    # 结构归位仅支持部分格式：增强解析开启时，不支持的格式自动回退普通解析
    if enhance:
        ext = os.path.splitext(filename)[1].lower()
        from structure_resolver import STRUCTURED_EXTENSIONS
        if ext not in STRUCTURED_EXTENSIONS:
            log("INFO", f"结构归位不支持 {ext}，增强解析回退为普通解析: {filename}")
            enhance = False

    # 同名冲突检测：存在运行中的同名任务则拒绝（避免并发入库互相覆盖）
    if ingest_queue.has_running(filename):
        log("WARN", f"文件正在入库中，拒绝重复提交: {filename}")
        raise HTTPException(status_code=409, detail=f"文件正在入库中: {filename}")

    # 读取并落盘到持久目录（任务执行期间文件必须存在）
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(save_path, "wb") as f:
        f.write(raw)

    # 提交异步任务，立即返回
    try:
        task_id = ingest_queue.submit(filename, enhance, save_path)
    except Exception as e:
        # 提交失败则清理落盘文件，避免残留
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        log("ERROR", f"提交入库任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交任务失败: {e}")

    log("INFO", f"已提交入库任务: {filename}（task_id={task_id[:8]}，增强解析: {'开' if enhance else '关'}）")
    return UploadTaskResponse(
        success=True, task_id=task_id, source=filename,
        status="PENDING", enhance=enhance,
        message="已提交，后台解析入库中",
    )


@app.get("/upload/tasks")
async def list_upload_tasks():
    """列出最近入库任务（前端展示历史/进度）。"""
    return {"tasks": ingest_queue.list_tasks()}


@app.get("/upload/{task_id}/status")
async def get_upload_status(task_id: str):
    """查询单个入库任务状态。"""
    task = ingest_queue.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.post("/upload/{task_id}/retry")
async def retry_upload_task(task_id: str):
    """重试失败的入库任务。"""
    if not ingest_queue.retry_task(task_id):
        raise HTTPException(status_code=400, detail="任务不存在或不可重试")
    return {"success": True, "task_id": task_id}


@app.delete("/upload/{task_id}")
async def delete_upload_task(task_id: str):
    """删除终态（失败/完成）入库任务及其产物、上传文件；运行中任务不可删。"""
    if not ingest_queue.delete_task(task_id):
        raise HTTPException(status_code=400, detail="任务不存在或正在运行中，不可删除")
    return {"success": True, "task_id": task_id}


# ============================================================================
# POST /chat — 对话（SSE 流式输出）
# ============================================================================
class ChatRequest(BaseModel):
    question:  str        = Field(..., description="用户问题")
    history:   List[dict] = Field(default_factory=list, description="对话历史")
    top_k:     int        = Field(default=5, description="返回给 LLM 的文档数")
    mode:      str        = Field(default="rag", description="运行模式：direct=直接对话 / rag=知识库问答 / agentic=Agentic 检索")
    hybrid:    bool       = Field(default=True, description="是否启用混合检索（dense + BM25 + RRF），向后兼容")
    retrieval_mode: str   = Field(default="", description="检索模式：vector=普通 / hybrid=混合 / tree=树导航（空=读 config）")
    conversation_id: str  = Field(default="", description="对话 ID（用于持久化历史，可为空）")


def _retrieve(query: str, top_k: int, mode: str = "hybrid") -> list:
    """
    检索知识库（主程序直接调用本地数据服务）。

    mode:
      - vector: 纯向量检索（dense）
      - hybrid: 混合检索（dense + BM25 + RRF）
      - tree:   纯 LLM 树导航检索（不碰向量召回）

    返回 [{"text","score","parent_id","doc_id","section_path_str"}, ...]。
    """
    from db_service import search_documents
    mode = (mode or "hybrid").strip().lower()
    if mode == "vector":
        return search_documents(query, top_k=top_k, hybrid=False)
    if mode == "tree":
        # 纯树导航检索：统一走根目录 tree_retrieval 的三级降级
        # （树导航 → 章节定位 → hybrid 补齐），不再自行拼装。
        try:
            import tree_retrieval
            from llm_factory import get_model
            docs = tree_retrieval.tree_search(
                query, top_k=top_k,
                reranker=get_reranker(),
                llm=get_model("tool_llm", "llm"),
            )
            return [{
                "text": d.get("text", ""),
                "score": d.get("score", 0.0),
                "parent_id": d.get("parent_id", ""),
                "doc_id": d.get("doc_id", ""),
                "section_path_str": d.get("section_path", ""),
                "is_neighbor": d.get("is_neighbor", False),
            } for d in docs]
        except Exception:
            # 树导航异常 → 降级 hybrid
            return search_documents(query, top_k=top_k, hybrid=True)
    # 默认 hybrid（含非法值回退）
    return search_documents(query, top_k=top_k, hybrid=True)


def _save_conversation_sync(conversation_id: str, question: str, answer: str):
    """将一轮问答写入对话历史（主程序直接调用本地 chat_history）。

    首轮对话后，若标题仍为默认「新对话」，用 LLM 生成摘要标题并更新。
    """
    import chat_history
    chat_history.add_message(conversation_id, "user", question)
    chat_history.add_message(conversation_id, "assistant", answer)
    _maybe_generate_title(conversation_id, question, answer)


def _maybe_generate_title(conversation_id: str, question: str, answer: str):
    """若对话标题仍为默认「新对话」，用 LLM 生成摘要标题并更新。"""
    import chat_history
    try:
        cur = chat_history.get_title(conversation_id)
        if cur and cur != "新对话":
            return  # 已有自定义标题，跳过
        summary = _generate_title(question, answer)
        if summary:
            chat_history.update_title(conversation_id, summary)
            log("INFO", f"对话标题已更新: {summary}")
    except Exception as e:
        log("WARN", f"生成对话标题失败: {e}")


def _generate_title(question: str, answer: str) -> str:
    """用 LLM 根据首轮问答生成简短中文标题（不超过 20 字）。"""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm import invoke_llm
        llm = get_llm()
        prompt = (
            "请根据以下对话生成一个简短的中文标题（不超过 20 字），"
            "概括对话主题，不要加引号、不要以句号结尾：\n\n"
            f"用户：{question[:200]}\n助手：{answer[:200]}\n\n标题："
        )
        text = invoke_llm(llm, [
            SystemMessage(content="你是对话标题生成助手，只输出简短标题。"),
            HumanMessage(content=prompt),
        ]).strip().strip('"\'').strip()
        if text:
            return text.split("\n")[0].strip()[:20]
    except Exception:
        pass
    return ""


# ============================================================================
# 对话管理 API（历史持久化到独立 Milvus database）
# ============================================================================

@app.post("/conversations")
async def create_conversation(body: dict = None):
    """创建新对话，返回 conversation_id。"""
    body = body or {}
    title = (body.get("title") or "").strip()
    try:
        import chat_history
        conv = chat_history.create_conversation(title)
        return {"success": True, "conversation": conv}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"创建对话失败: {e}")


@app.get("/conversations")
async def list_conversations_api():
    """列出所有历史对话。"""
    try:
        import chat_history
        convs = chat_history.list_conversations()
        return {"success": True, "conversations": convs}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"获取对话列表失败: {e}")


@app.get("/conversations/{conversation_id}")
async def get_conversation_api(conversation_id: str):
    """获取指定对话的完整历史。"""
    try:
        import chat_history
        msgs = chat_history.get_conversation(conversation_id)
        return {"success": True, "conversation_id": conversation_id, "messages": msgs}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"获取对话历史失败: {e}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation_api(conversation_id: str):
    """删除指定对话（及其持久化历史）。"""
    try:
        import chat_history
        chat_history.delete_conversation(conversation_id)
        return {"success": True, "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"删除对话失败: {e}")


@app.post("/chat")
async def chat(req: ChatRequest):
    """对话接口（SSE 流式）：按 mode 走 direct（纯 LLM）或 rag（知识库检索）链路"""

    question, history, top_k = req.question, req.history or [], req.top_k
    mode = req.mode if req.mode in ("direct", "rag", "agentic") else "rag"
    conversation_id = req.conversation_id or ""
    hybrid = bool(req.hybrid)
    # 检索模式：显式传入优先，否则读 config.search.retrieval_mode（默认 hybrid）
    retrieval_mode = (req.retrieval_mode or "").strip().lower()
    if retrieval_mode not in ("vector", "hybrid", "tree"):
        retrieval_mode = config["search"].get("retrieval_mode", "hybrid")

    q_brief = question[:50] + ('...' if len(question) > 50 else '')
    mode_label = {"direct": "直接对话", "rag": "知识库问答", "agentic": "Agentic 检索"}.get(mode, mode)
    rm_label = {"vector": "普通检索", "hybrid": "混合检索", "tree": "树导航检索"}.get(retrieval_mode, retrieval_mode)
    log("INFO", f"收到对话请求: {q_brief}（模式: {mode_label}，检索模式: {rm_label}）")

    gen = _generate_langgraph(question, history, top_k, mode=mode, conversation_id=conversation_id,
                              hybrid=hybrid, retrieval_mode=retrieval_mode)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _generate_langgraph(question: str, history: list, top_k: int, mode: str = "rag", conversation_id: str = "", hybrid: bool = True, retrieval_mode: str = "hybrid"):
    """
    LangGraph 链路（direct / rag 两种模式）。

    direct：model 节点纯 LLM 推理；
    rag：   retrieve → rerank → grade → generate 固定链路。

    状态机在后台线程同步运行，通过 asyncio.Queue 跨线程转发
    status/token 事件到 SSE，保留流式体验。

    若提供 conversation_id，则在生成完成后将本轮问答写入对话历史（本地 Milvus）。
    """
    # Agentic RAG 入口（Phase 10：完整 Requirement–Evidence–Gap 循环）
    if mode == "agentic":
        # Reranker 是 evaluate 的 quality 指标来源，获取失败则报错（与 rag 一致）
        try:
            agentic_reranker = get_reranker()
        except Exception as e:
            yield _sse("error", f"[Agentic 初始化失败] {e}")
            yield _sse("done")
            return

        event_queue = asyncio.Queue()

        def on_status(msg: str):
            log("INFO", f"[Agentic] {msg}")
            event_queue.put_nowait(("status", msg))

        def on_token(text: str, is_reasoning: bool = False):
            typ = "reasoning" if is_reasoning else "token"
            event_queue.put_nowait((typ, text))

        def on_trace(snapshot: dict):
            # 结构化决策轨迹：每轮评估后推状态快照（前端 Agent 工作台可视化）
            event_queue.put_nowait(("agent_trace", json.dumps(snapshot, ensure_ascii=False)))

        result_holder = {}

        def _run_agentic():
            try:
                from agentic_rag import run_agentic
                result_holder["result"] = run_agentic(
                    question,
                    reranker=agentic_reranker,
                    status_callback=on_status,
                    token_callback=on_token,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    conversation_id=conversation_id,
                    trace_callback=on_trace,
                )
            except Exception as e:
                result_holder["error"] = str(e)

        import threading
        t = threading.Thread(target=_run_agentic, daemon=True)
        t.start()

        # 消费事件流（循环轨迹 status + 流式 token/reasoning 推给前端）
        while True:
            finished = not t.is_alive()
            try:
                while True:
                    typ, content = event_queue.get_nowait()
                    yield _sse(typ, content)
                    await asyncio.sleep(0)
            except asyncio.QueueEmpty:
                pass
            if finished and event_queue.empty():
                break
            await asyncio.sleep(0.02)

        if "error" in result_holder:
            yield _sse("error", f"[Agentic] {result_holder['error']}")
        else:
            result = result_holder.get("result", {})
            citations = result.get("citations", [])
            # answer 已通过流式 token 实时推送，这里不再重复 yield 完整 answer
            if citations:
                yield _sse("citations", json.dumps(citations, ensure_ascii=False))

        # 写入对话历史（若提供了 conversation_id）
        if conversation_id:
            generation = result_holder.get("result", {}).get("answer", "")
            if generation:
                try:
                    import threading as _th
                    _th.Thread(
                        target=_save_conversation_sync,
                        args=(conversation_id, question, generation),
                        daemon=True,
                    ).start()
                except Exception as e:
                    log("WARN", f"写入对话历史失败: {e}")

        yield _sse("done")
        return

    # 组装 LangGraph 链路所需的组件（复用懒加载单例）
    try:
        reranker = get_reranker()
        llm      = get_llm()
        tool_llm = get_tool_llm()     # 工具决策模型（未配置时回退到 llm）
        rewrite_llm = get_rewrite_llm()  # 查询重写模型（未配置为 None，rewrite 透传）
    except Exception as e:
        yield _sse("error", f"[初始化失败] {e}")
        yield _sse("done")
        return

    # 事件队列：status / token / reasoning 统一通过队列跨线程传递
    event_queue = asyncio.Queue()

    def on_status(msg: str):
        log("INFO", f"[链路] {msg}")
        event_queue.put_nowait(("status", msg))

    def on_token(text: str, is_reasoning: bool = False):
        # 思考过程用 reasoning 事件，正文用 token 事件
        typ = "reasoning" if is_reasoning else "token"
        event_queue.put_nowait((typ, text))

    # 检索直接走本地数据服务（主程序直接连接数据库）
    def retriever(query: str, top_k: int, mode: str = "hybrid"):
        return _retrieve(query, top_k, mode)

    # 构建 RAGGraph（固定链路；工具决策模型 tool_llm、重写模型 rewrite_llm 独立传入）
    graph = RAGGraph(
        reranker, llm,
        retriever=retriever,
        tool_llm=tool_llm,
        rewrite_llm=rewrite_llm,
        rerank_top_n=top_k,
        use_hybrid=hybrid,
        retrieval_mode=retrieval_mode,
        stream_chunk_callback=on_token,
        status_callback=on_status,
    )

    # 在后台线程运行 langgraph（同步阻塞），主协程消费事件队列
    result_holder = {}

    def _run_graph():
        try:
            result_holder["state"] = graph.run(question, history, mode=mode)
        except Exception as e:
            result_holder["error"] = str(e)

    import threading
    t = threading.Thread(target=_run_graph, daemon=True)
    t.start()

    # 消费事件流，直到线程结束且队列排空
    while True:
        finished = not t.is_alive()
        try:
            while True:
                typ, content = event_queue.get_nowait()
                yield _sse(typ, content)
                await asyncio.sleep(0)
        except asyncio.QueueEmpty:
            pass

        if finished and event_queue.empty():
            break
        await asyncio.sleep(0.02)

    # 处理异常
    if "error" in result_holder:
        yield _sse("error", f"[LangGraph] {result_holder['error']}")

    # 写入对话历史（若提供了 conversation_id）
    if conversation_id:
        generation = (result_holder.get("state") or {}).get("generation", "")
        if generation:
            try:
                # 在后台线程中写入（避免阻塞事件循环；chat_history 为同步阻塞调用）
                import threading as _th
                _th.Thread(
                    target=_save_conversation_sync,
                    args=(conversation_id, question, generation),
                    daemon=True,
                ).start()
            except Exception as e:
                log("WARN", f"写入对话历史失败: {e}")

    yield _sse("done")



def _sse(typ: str, content: str = "") -> str:
    return f"data: {json.dumps({'type': typ, 'content': content}, ensure_ascii=False)}\n\n"


# ============================================================================
# GET /health
# ============================================================================
@app.get("/health")
async def health():
    s = {"status": "ok"}
    try:
        from db_service import count
        result = count() or {}
        s["count"]        = int(result.get("count", 0))
        s["parent_count"] = int(result.get("parent_count", 0))
        s["db_available"] = True
    except Exception:
        s["count"] = 0
        s["parent_count"] = 0
        s["db_available"] = False
    return s


@app.get("/logs")
async def get_logs(limit: int = 200):
    """获取服务日志（最近 limit 条，供前端主页日志面板展示）"""
    logs = list(_log_buffer)
    return {"success": True, "logs": logs[-limit:]}


# ============================================================================
# 模型配置 API
# ============================================================================
class ModelConfig(BaseModel):
    kind: str
    name: str
    type: str = "online"          # online / local（仅 reranker 支持 local）
    model: str = ""               # 在线模型名称
    base_url: str = ""            # 在线 API 地址
    api_key: str = ""             # 在线 API 密钥
    model_path: str = ""          # 离线模型本地路径
    protocol: str = "openai"      # LLM 协议：openai / doubao

@app.get("/config")
async def get_config():
    """获取当前选中的模型/数据库配置（均由 store_config 持久化）。"""
    return {
        "models": store_config.load_models(),
        "dbs": store_config.load_dbs(),
        "current": {
            "llm": _current_llm_name,
            "embedding": store_config.get_current("embedding"),
            "reranker": _current_rerank_name,
            "tool_llm": _current_tool_llm_name,
            "summary": store_config.get_current("summary"),
            "db": store_config.get_current_db(),
        },
        "search": {
            "rewrite": config["search"].get("rewrite", False),
            "hybrid": config["search"].get("hybrid", True),
            "retrieval_mode": config["search"].get("retrieval_mode", "hybrid"),
        },
        "summary": {
            "enabled": config.get("summary", {}).get("enabled", True),
        },
        "tool_calling": {
            "enabled": config.get("mcp", {}).get("tool_calling", {}).get("enabled", False),
        },
        "system_prompt": config.get("system_prompt", ""),
    }

@app.post("/config/search")
async def set_search_config(body: dict):
    """切换查询重写（rewrite）/ 混合检索（hybrid）开关 / 检索模式（retrieval_mode）"""
    from config_loader import set_config
    if "rewrite" in body:
        set_config("search.rewrite", bool(body["rewrite"]))
    if "hybrid" in body:
        set_config("search.hybrid", bool(body["hybrid"]))
    if "retrieval_mode" in body:
        rm = (str(body.get("retrieval_mode") or "").strip().lower())
        if rm not in ("vector", "hybrid", "tree"):
            rm = "hybrid"
        set_config("search.retrieval_mode", rm)
    return {
        "success": True,
        "rewrite": config["search"].get("rewrite", False),
        "hybrid": config["search"].get("hybrid", True),
        "retrieval_mode": config["search"].get("retrieval_mode", "hybrid"),
    }

@app.post("/config/tool_calling")
async def set_tool_calling(body: dict):
    """切换工具调用（可插拔工具决策）开关"""
    from config_loader import set_config
    if "enabled" in body:
        set_config("mcp.tool_calling.enabled", bool(body["enabled"]))
    return {
        "success": True,
        "enabled": config.get("mcp", {}).get("tool_calling", {}).get("enabled", False),
    }


@app.post("/config/summary")
async def set_summary_config(body: dict):
    """切换章节摘要（summary）开关"""
    from config_loader import set_config
    if "enabled" in body:
        set_config("summary.enabled", bool(body["enabled"]))
    return {
        "success": True,
        "enabled": config.get("summary", {}).get("enabled", True),
    }

@app.get("/config/system_prompt")
async def get_system_prompt():
    """获取系统提示词"""
    return {"system_prompt": config.get("system_prompt", "")}

@app.post("/config/system_prompt")
async def set_system_prompt(body: dict):
    """设置系统提示词"""
    from config_loader import set_config
    sp = str(body.get("system_prompt", ""))
    set_config("system_prompt", sp)
    return {"success": True, "system_prompt": sp}

@app.post("/config/select")
async def select_config(body: dict):
    """切换当前使用的模型/数据库（均持久化到 store_config，重启不丢失）。"""
    global _current_llm_name, _current_rerank_name, _current_tool_llm_name
    if "llm" in body:
        _current_llm_name = body["llm"]
        store_config.set_current("llm", body["llm"])
    if "tool_llm" in body:
        _current_tool_llm_name = body["tool_llm"]
        store_config.set_current("tool_llm", body["tool_llm"])
    if "summary" in body:
        # summary 模型由 summarizer 每次现取（入库时用），仅持久化 current，无需重置组件
        store_config.set_current("summary", body["summary"])
    if "embedding" in body:
        store_config.set_current("embedding", body["embedding"])
        from db_service import reset_embedding
        reset_embedding()
    if "reranker" in body:
        _current_rerank_name = body["reranker"]
        store_config.set_current("reranker", body["reranker"])
    if "db" in body:
        store_config.set_current_db(body["db"])
        from db_service import reset_store
        reset_store()
    _reset_components()

    return {"success": True}

@app.get("/browse")
async def browse_directory(path: str = ""):
    """
    浏览服务器文件系统目录，返回子目录列表（用于选择本地模型路径）。

    Args:
        path: 要浏览的目录（空=列出盘符/根目录）
    """
    try:
        if not path:
            # 空路径：Windows 返回盘符列表，Linux 返回根目录
            if os.name == "nt":
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:\\"
                    if os.path.exists(drive):
                        drives.append(drive)
                return {"path": "", "dirs": drives, "parent": None}
            else:
                path = "/"

        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"路径不存在: {path}")

        # 列出所有子目录（不列文件，模型路径是目录）
        dirs = []
        try:
            for entry in os.listdir(path):
                full = os.path.join(path, entry)
                if os.path.isdir(full) and not entry.startswith('.'):
                    dirs.append(entry)
        except PermissionError:
            pass
        dirs.sort()

        # 父目录
        parent = os.path.dirname(path)
        if parent == path:
            parent = None

        return {"path": path, "dirs": dirs, "parent": parent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pick-directory")
async def pick_directory():
    """
    在服务器端（本机）弹出 Windows 原生"选择文件夹"对话框。

    通过 tkinter 的 filedialog.askdirectory 实现，返回用户选中的绝对路径。
    仅适用于服务器和浏览器在同一台机器上的场景（本机访问 localhost）。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise HTTPException(status_code=500, detail="当前 Python 环境未安装 tkinter，无法弹出系统对话框")

    # 在子线程中运行 tkinter（避免阻塞事件循环，且 tkinter 需在主线程但这里用独立线程+隐藏根窗口）
    import threading
    result = {}

    def _run():
        try:
            root = tk.Tk()
            root.withdraw()          # 隐藏主窗口
            root.attributes('-topmost', True)  # 置顶，确保对话框可见
            path = filedialog.askdirectory(title="选择模型文件夹")
            root.destroy()
            result["path"] = path if path else ""
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=300)  # 最多等待 5 分钟

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"对话框错误: {result['error']}")

    path = result.get("path", "")
    return {"success": True, "path": path, "cancelled": not path}


@app.get("/models")
async def list_all_models():
    return store_config.load_models()

@app.post("/models")
async def add_model(m: ModelConfig):
    """新增模型"""
    try:
        model_dict = {
            "name": m.name,
            "type": m.type,
            "model": m.model,
            "base_url": m.base_url,
            "api_key": m.api_key,
            "model_path": m.model_path,
            "protocol": m.protocol,
        }
        store_config.add_model(m.kind, model_dict)
        return {"success": True, "model": model_dict}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/models/{kind}/{name}")
async def delete_model(kind: str, name: str):
    store_config.delete_model(kind, name)
    # 若删除的是 embedding，重置 embedding 缓存
    if kind == "embedding":
        from db_service import reset_embedding
        reset_embedding()
    # 如果删除的是当前使用的，重置
    _reset_components()
    return {"success": True}

@app.put("/models/{kind}/{name}")
async def update_model(kind: str, name: str, m: ModelConfig):
    """编辑模型（路径 name 为原名称，body.name 为编辑后的新名称）。"""
    try:
        # 掩码 key：前端不回显明文，留空表示不修改（沿用已保存的 api_key）
        api_key = m.api_key
        if not api_key:
            old = store_config.get_model_by_name(kind, name)
            api_key = (old or {}).get("api_key", "")

        model_dict = {
            "name": m.name,
            "type": m.type,
            "model": m.model,
            "base_url": m.base_url,
            "api_key": api_key,
            "model_path": m.model_path,
            "protocol": m.protocol,
        }
        store_config.update_model(kind, name, model_dict)
        # 编辑 embedding 后重置缓存，避免沿用旧模型实例
        if kind == "embedding":
            from db_service import reset_embedding
            reset_embedding()
        _reset_components()
        return {"success": True, "model": model_dict}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Reranker 本地模型下载 / 加载 API
# ============================================================================
@app.get("/reranker/status")
async def reranker_status():
    """
    返回当前 reranker 本地模型加载状态。

    status 取值：
      idle     未加载（本地模型尚未下载/加载）
      loading  正在下载+加载中
      loaded   已加载
      error    加载失败（message 含错误信息）
      online   在线模型（无需下载加载）
    """
    # 判断当前 reranker 是本地还是在线
    m = _get_current_model("reranker", _current_rerank_name)
    if m is None:
        return {"type": "none", "status": "idle", "message": "未配置 Reranker 模型"}

    if m.get("type") != "local":
        return {"type": "online", "status": "online", "message": "在线模型，无需下载"}

    # 本地模型：优先用 _reranker_load_state；若已加载则反映实际状态
    if _reranker_load_state["status"] == "loaded":
        return {"type": "local", "status": "loaded", "message": _reranker_load_state["message"]}

    # 若已有 reranker 实例且已加载，同步状态
    if _reranker is not None and _reranker.is_loaded:
        _reranker_load_state["status"] = "loaded"
        _reranker_load_state["message"] = "模型已加载"
        return {"type": "local", "status": "loaded", "message": "模型已加载"}

    return {
        "type": "local",
        "status": _reranker_load_state["status"],
        "message": _reranker_load_state["message"],
    }


@app.post("/reranker/load")
async def reranker_load():
    """
    触发本地 reranker 模型的下载与加载（后台线程执行）。

    下载可能耗时数分钟（BGE-Reranker 约 1.1GB），接口立即返回，
    前端通过轮询 /reranker/status 跟踪进度。
    """
    global _reranker, _reranker_load_state

    m = _get_current_model("reranker", _current_rerank_name)
    if m is None:
        raise HTTPException(status_code=400, detail="未配置 Reranker 模型")
    if m.get("type") != "local":
        raise HTTPException(status_code=400, detail="当前 Reranker 是在线模型，无需下载")

    # 已在加载中，避免重复触发
    if _reranker_load_state["status"] == "loading":
        return {"success": True, "status": "loading", "message": "正在加载中，请稍候..."}

    model_path = m.get("model_path") or "BAAI/bge-reranker-v2-m3"
    _reranker_load_state["status"] = "loading"
    _reranker_load_state["message"] = f"正在下载并加载 {model_path} ..."

    import threading

    def _do_load():
        global _reranker, _reranker_load_state
        try:
            # 构造本地 Reranker 并显式加载（下载权重 + 加载到内存）
            r = Reranker(local_model_path=model_path)
            r.load()
            _reranker = r  # 替换全局单例
            _reranker_load_state["status"] = "loaded"
            _reranker_load_state["message"] = "模型加载完成"
        except Exception as e:
            _reranker_load_state["status"] = "error"
            _reranker_load_state["message"] = f"加载失败: {e}"

    t = threading.Thread(target=_do_load, daemon=True)
    t.start()

    return {"success": True, "status": "loading", "message": "已开始后台下载加载"}


# ============================================================================
# 数据库 / 数据管理 API（主程序直接连接数据库）
# ============================================================================

class ClearResponse(BaseModel):
    success:       bool = True
    deleted_count: int  = 0
    message:       str  = ""

class DBConfig(BaseModel):
    name: str
    type: str = "local"    # local / online
    url: str = ""
    token: str = ""
    db_name: str = ""

@app.get("/dbs")
async def list_all_dbs():
    """列出数据库配置（读 config/db.json）。"""
    return store_config.load_dbs()

@app.post("/dbs")
async def add_db(d: DBConfig):
    """新增数据库配置（持久化到 db.json）。"""
    try:
        db_dict = {
            "name": d.name,
            "type": d.type,
            "url": d.url,
            "token": d.token,
            "db_name": d.db_name,
        }
        if d.type == "local" and not d.url:
            from milvus_store import LOCAL_MILVUS_URL
            db_dict["url"] = LOCAL_MILVUS_URL
        db = store_config.add_db(db_dict)
        return {"success": True, "db": db}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {e}")

@app.delete("/dbs/{name}")
async def delete_db(name: str):
    """删除数据库配置。"""
    try:
        store_config.delete_db(name)
        from db_service import reset_store
        reset_store()
        return {"success": True, "message": f"已删除数据库 {name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

@app.get("/local/databases")
async def list_local_dbs():
    """列出本地 Milvus 服务中所有 database。"""
    try:
        from db_service import list_databases
        databases = list_databases()
        return {"databases": databases or []}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"查询 database 失败: {e}")

@app.post("/local/databases")
async def create_local_db(body: dict):
    """在本地 Milvus 新建 database。"""
    db_name = body.get("db_name", "")
    if not db_name:
        raise HTTPException(status_code=400, detail="database 名称不能为空")
    try:
        from db_service import create_database
        ok = create_database(db_name)
        if not ok:
            raise RuntimeError(f"创建 database 失败: {db_name}")
        return {"success": True, "db_name": db_name}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"创建 database 失败: {e}")

@app.get("/local/parents")
async def list_local_parents():
    """列出当前数据库所有父块。"""
    try:
        from db_service import list_parents
        parents = list_parents() or []
        return {"success": True, "total": len(parents), "parents": parents}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"查询失败: {e}")

@app.get("/local/parents/{parent_id}/children")
async def list_children(parent_id: str):
    """列出指定父块下的所有子块。"""
    try:
        from db_service import list_children
        children = list_children(parent_id) or []
        return {"success": True, "parent_id": parent_id, "children": children, "total": len(children)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"查询失败: {e}")


@app.get("/local/tree/{doc_id}")
async def get_local_tree(doc_id: str):
    """返回结构树文档的完整树形结构（章节层级 + chunk 挂载）。"""
    try:
        from db_service import get_document_tree
        tree = get_document_tree(doc_id)
        if tree is None:
            raise HTTPException(status_code=404, detail="文档树不存在")
        return {"success": True, "doc_id": doc_id, "tree": tree}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"查询失败: {e}")

@app.delete("/local/children/{child_id}")
async def delete_child(child_id: int):
    """删除单个子块。"""
    try:
        from db_service import delete_child
        delete_child(child_id)
        return {"success": True, "message": f"已删除子块 {child_id}"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"删除失败: {e}")

@app.delete("/local/parents/{parent_id}")
async def delete_local_parent(parent_id: str):
    """删除父块及其子块。"""
    try:
        from db_service import delete_parent
        delete_parent(parent_id)
        return {"success": True, "message": f"已删除父块 {parent_id} 及其子块"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"删除失败: {e}")

@app.delete("/local/sources/{source}")
async def delete_source(source: str):
    """删除指定文件（source）下的所有父块和子块。"""
    try:
        from db_service import delete_source
        delete_source(source)
        return {"success": True, "message": f"已删除文件 {source} 及其所有分块"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"删除失败: {e}")

@app.put("/local/sources/rename")
async def rename_source(body: dict):
    """重命名文件（只改文件名，不改任何内容）。"""
    old_source = (body.get("old_source") or "").strip()
    new_source = (body.get("new_source") or "").strip()
    if not old_source or not new_source:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if old_source == new_source:
        return {"success": True, "message": "文件名未变化"}
    try:
        from db_service import rename_source as _rename_source
        _rename_source(old_source, new_source)
        return {"success": True, "message": f"已重命名 {old_source} → {new_source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名失败: {e}")

@app.delete("/local/clear", response_model=ClearResponse)
async def clear_local():
    """清空知识库。"""
    try:
        from db_service import clear_all
        deleted = clear_all()
        return ClearResponse(
            success=True, deleted_count=deleted,
            message=f"已清空，删除 {deleted} 条",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"清空失败: {e}")
