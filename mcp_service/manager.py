"""
================================================================================
MCP 服务器管理器
================================================================================

提供 MCP 服务器的完整管理能力：

1. 服务器配置 CRUD（命令、参数、URL 等，持久化到 config/mcp_servers.json）
2. 服务器生命周期管理（启动 / 停止 / 状态查询）
3. 工具交互（连接服务器 → 列出工具 → 调用工具）
4. 状态监控与日志（运行中 / 已停止 / 出错 + 环形日志缓冲）

管理 API（挂载到主 server 的 /mcp/* 路径）：
  GET    /mcp/servers                    列出所有服务器配置
  POST   /mcp/servers                    新增/更新服务器配置
  DELETE /mcp/servers/{name}             删除服务器配置
  POST   /mcp/servers/{name}/start       启动服务器（子进程）
  POST   /mcp/servers/{name}/stop        停止服务器
  GET    /mcp/servers/{name}/status      服务器运行状态
  GET    /mcp/servers/{name}/tools       列出服务器的工具清单（实时连接）
  POST   /mcp/servers/{name}/call        调用服务器上的工具
  GET    /mcp/servers/{name}/logs        获取服务器日志
  GET    /mcp/settings                   查看 MCP 功能开关
  POST   /mcp/settings                   更新 MCP 功能开关
================================================================================
"""

import json
import os
import subprocess
import sys
import threading
import time
import collections
from typing import Optional

# 项目根目录（mcp 包的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MCP 服务器配置持久化文件（项目根目录下的 config）
SERVERS_PATH = os.path.join(BASE_DIR, "config", "mcp_servers.json")

# 默认服务器配置（首次自动创建）
# command 为 "-m mcp"（以模块方式启动 mcp/__main__.py），args 为启动参数。
DEFAULT_SERVER = {
    "name": "RAG-Service",
    "command": "-m",
    "args": "mcp_service --host 127.0.0.1 --port 8765",
    "url": "http://127.0.0.1:8765/mcp",
    "transport": "streamable-http",
    "auto_start": False,
}

# 进程句柄：{server_name: Popen}
_server_processes = {}

# NO_PROXY 是否已设置（幂等标志，避免每次 _get_client 重复改全局环境变量）
_no_proxy_ensured = False

# 日志缓冲：{server_name: deque(maxlen=500) of (ts, level, msg)}
_log_buffers = collections.defaultdict(
    lambda: collections.deque(maxlen=500)
)

# 工具清单缓存：{server_name: [tool schema dict, ...]}
# 工具 schema（name/description/input_schema）由 mcp_service/__main__.py 静态定义，
# 不会因运行时变化，首次连接后缓存，避免前端每次刷新都实时连接（很慢）。
# 服务器启动/停止时清空，保证重启后重新拉取最新清单。
_tools_cache = {}


def _invalidate_tools_cache(name: str = None) -> None:
    """清空工具清单缓存（name=None 时全部清空）。"""
    if name is None:
        _tools_cache.clear()
    else:
        _tools_cache.pop(name, None)


# ============================================================================
# 服务器配置持久化
# ============================================================================

def _ensure_servers_file() -> None:
    if not os.path.exists(SERVERS_PATH):
        os.makedirs(os.path.dirname(SERVERS_PATH), exist_ok=True)
        with open(SERVERS_PATH, "w", encoding="utf-8") as f:
            json.dump({"servers": [DEFAULT_SERVER]}, f, ensure_ascii=False, indent=2)


def list_servers() -> list:
    """列出所有服务器配置。"""
    _ensure_servers_file()
    try:
        with open(SERVERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("servers", [])
    except Exception:
        return []


def _save_servers(servers: list) -> None:
    os.makedirs(os.path.dirname(SERVERS_PATH), exist_ok=True)
    with open(SERVERS_PATH, "w", encoding="utf-8") as f:
        json.dump({"servers": servers}, f, ensure_ascii=False, indent=2)


def get_server(name: str):
    """按名称获取服务器配置；不存在返回 None。"""
    for s in list_servers():
        if s.get("name") == name:
            return s
    return None


def upsert_server(cfg: dict) -> dict:
    """新增或更新服务器配置。"""
    name = (cfg.get("name") or "").strip()
    if not name:
        raise ValueError("服务器名称不能为空")

    server = {
        "name": name,
        "command": (cfg.get("command") or "").strip(),
        "args": (cfg.get("args") or "").strip(),
        "url": (cfg.get("url") or "").strip(),
        "transport": cfg.get("transport") or "streamable-http",
        "auto_start": bool(cfg.get("auto_start", False)),
    }

    servers = list_servers()
    for i, s in enumerate(servers):
        if s.get("name") == name:
            servers[i] = server
            break
    else:
        servers.append(server)
    _save_servers(servers)
    return server


def delete_server(name: str) -> bool:
    """删除服务器配置（若在运行则先停止）。"""
    stop_server(name)
    servers = [s for s in list_servers() if s.get("name") != name]
    _save_servers(servers)
    _log_buffers.pop(name, None)
    return True


# ============================================================================
# 日志
# ============================================================================

def _log(name: str, level: str, msg: str):
    """记录日志到环形缓冲。"""
    _log_buffers[name].append({
        "ts": time.strftime("%H:%M:%S"),
        "level": level,
        "msg": msg,
    })


def get_logs(name: str, limit: int = 200) -> list:
    """获取服务器日志（最近 limit 条）。"""
    logs = list(_log_buffers.get(name, []))
    return logs[-limit:]


# ============================================================================
# 生命周期管理
# ============================================================================

def _build_command(server: dict) -> list:
    """根据服务器配置构造启动命令。

    支持两种形式：
      - command="-m", args="mcp_service --host ..." → [python, "-m", "mcp_service", "--host", ...]
      - command="xxx.py", args="..."            → [python, BASE_DIR/xxx.py, ...]
    """
    command = server.get("command") or ""
    args = (server.get("args") or "").strip()

    if command == "-m":
        # 以模块方式启动（python -m mcp_service ...）
        cmd = [sys.executable, "-m"]
    elif command.endswith(".py"):
        # 脚本路径（相对项目根目录）
        script = os.path.join(BASE_DIR, command)
        cmd = [sys.executable, script]
    else:
        cmd = [command]

    if args:
        cmd += args.split()
    return cmd


def start_server(name: str) -> dict:
    """启动服务器（后台子进程），并捕获日志。"""
    server = get_server(name)
    if server is None:
        return {"success": False, "message": f"服务器 {name} 不存在"}

    # 已在运行
    if is_running(name):
        return {"success": True, "message": f"服务器 {name} 已在运行"}

    try:
        cmd = _build_command(server)
        _log(name, "INFO", f"启动命令: {' '.join(cmd)}")

        # 子进程输出直接丢弃（不落盘），运行日志仅通过内存环形缓冲 _log 供前端展示
        devnull = open(os.devnull, "wb")

        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=devnull,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0,
        )
        _server_processes[name] = proc
        _invalidate_tools_cache(name)  # 重启后工具清单可能变化，清缓存
        _log(name, "INFO", f"服务器已启动 (pid={proc.pid})")

        # 后台监控进程退出
        def _monitor():
            proc.wait()
            code = proc.returncode
            _log(name, "WARN" if code != 0 else "INFO",
                 f"服务器进程退出 (code={code})")
            _server_processes.pop(name, None)
            _invalidate_tools_cache(name)

        threading.Thread(target=_monitor, daemon=True).start()

        # 后台等待服务器就绪（探测 URL，最多 30 秒）
        def _wait_ready():
            url = server.get("url", "")
            if not url:
                return
            for _ in range(30):
                if proc.poll() is not None:
                    _log(name, "ERROR", "服务器进程提前退出，未能就绪")
                    return
                if _probe_url(url):
                    _log(name, "INFO", "服务器已就绪，可接受请求")
                    return
                time.sleep(1)
            _log(name, "WARN", "等待服务器就绪超时（30s），可能是启动较慢")

        threading.Thread(target=_wait_ready, daemon=True).start()

        return {"success": True, "message": f"服务器 {name} 已启动"}
    except Exception as e:
        _log(name, "ERROR", f"启动失败: {e}")
        return {"success": False, "message": f"启动失败: {e}"}


def _parse_port(url: str) -> Optional[int]:
    """从 URL 提取端口（缺省 http=80 / https=443；无 scheme 或空 URL 返回 None）。"""
    from urllib.parse import urlparse
    if not url or not url.strip():
        return None
    try:
        p = urlparse(url)
        if p.port:
            return p.port
        if p.scheme:
            return 443 if p.scheme == "https" else 80
        return None
    except Exception:
        return None


def _find_pid_by_port(port: int) -> Optional[int]:
    """通过端口找到监听进程的 PID（Windows netstat / Unix lsof）。"""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
                    local = parts[1]  # 如 127.0.0.1:8765 或 [::]:8765
                    if local.rsplit(":", 1)[-1] == str(port):
                        return int(parts[4])
        else:
            out = subprocess.run(
                ["lsof", "-t", "-i", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            first = out.strip().splitlines()[0] if out.strip() else None
            if first:
                return int(first)
    except Exception:
        pass
    return None


def _kill_pid(pid: int) -> bool:
    """结束指定 PID 的进程（Windows taskkill / Unix SIGTERM）。"""
    if pid == os.getpid():
        return False  # 保护：绝不结束自身进程
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def stop_server(name: str) -> dict:
    """停止服务器进程。

    两段式：
      1. 有进程句柄（本进程启动）→ terminate + wait，超时 kill；
      2. 无句柄（外部启动 / 主程序重启后的孤儿进程）→ 按 URL 端口找到 PID 并结束，
         避免「status 显示运行中、但停止按钮无效」。
    """
    proc = _server_processes.get(name)
    if proc is not None:
        # 用 _kill_pid（taskkill /T /F）杀整棵进程树，而非 terminate 只杀直接
        # 子进程：否则 MCP 服务器 spawn 的子进程（如有）会残留成孤儿进程。
        _kill_pid(proc.pid)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        _log(name, "INFO", "服务器已停止")
        _server_processes.pop(name, None)
        _invalidate_tools_cache(name)
        return {"success": True, "message": f"服务器 {name} 已停止"}

    # 无句柄：外部启动 / 孤儿进程 → 按端口兜底结束
    server = get_server(name)
    if server:
        port = _parse_port(server.get("url", ""))
        if port:
            pid = _find_pid_by_port(port)
            if pid:
                if _kill_pid(pid):
                    _log(name, "INFO", f"已停止外部/孤儿进程 (pid={pid})")
                    return {"success": True, "message": f"服务器 {name} 已停止 (pid={pid})"}
                return {"success": False, "message": f"停止进程 {pid} 失败"}

    return {"success": True, "message": f"服务器 {name} 未在运行"}


def is_running(name: str) -> bool:
    """服务器是否在运行。"""
    proc = _server_processes.get(name)
    return proc is not None and proc.poll() is None


def server_status(name: str) -> dict:
    """返回服务器运行状态（含是否配置了 URL）。"""
    server = get_server(name)
    if server is None:
        return {"exists": False, "running": False, "status": "not_found"}

    running = is_running(name)
    # 若进程句柄丢失但 URL 可达，也算"运行中"（外部启动的服务器）
    reachable = False
    if not running and server.get("url"):
        reachable = _probe_url(server["url"])

    status = "running" if (running or reachable) else "stopped"
    # 进程退出码非 0 视为出错
    proc = _server_processes.get(name)
    if proc is not None and proc.poll() not in (None, 0):
        status = "error"

    # PID：优先进程句柄；无句柄但「运行中」时按端口找 PID（外部启动/孤儿进程）
    pid = proc.pid if (proc and proc.poll() is None) else None
    if pid is None and (running or reachable) and server.get("url"):
        port = _parse_port(server.get("url", ""))
        if port:
            pid = _find_pid_by_port(port)

    return {
        "exists": True,
        "running": running or reachable,
        "status": status,
        "pid": pid,
        "url": server.get("url", ""),
    }


def _probe_url(url: str) -> bool:
    """探测 MCP 服务器 URL 是否可达（发送 OPTIONS 或简单 GET）。"""
    try:
        import httpx
        with httpx.Client(timeout=2.0, trust_env=False) as c:
            r = c.get(url, timeout=2.0)
            # 任何非 502/连接错误都视为可达（406 表示 MCP 端点存在）
            return r.status_code < 500
    except Exception:
        return False


# ============================================================================
# 工具交互（连接 MCP 服务器）
# ============================================================================

def _get_client(server: dict, timeout: float = 120.0):
    """构造 fastmcp Client（HTTP 连接）。

    Args:
        server: 服务器配置
        timeout: 客户端超时（秒）。豆包推理模型重写/生成较慢，默认给足 120 秒。
    """
    url = server.get("url")
    if not url:
        raise RuntimeError("服务器未配置 URL")
    # 确保本地地址直连（避免系统代理干扰）。仅首次设置一次（幂等），
    # 避免每次调用都读写全局环境变量（非线程安全的全局状态修改）。
    global _no_proxy_ensured
    if not _no_proxy_ensured:
        _no_proxy = os.environ.get("NO_PROXY", "")
        for h in ("127.0.0.1", "localhost"):
            if h not in _no_proxy:
                _no_proxy = f"{_no_proxy},{h}" if _no_proxy else h
        os.environ["NO_PROXY"] = _no_proxy
        os.environ["no_proxy"] = _no_proxy
        _no_proxy_ensured = True

    from fastmcp import Client
    return Client(url, timeout=timeout)


def _run_with_retry(server: dict, coro_factory, retries: int = 3, interval: float = 1.5):
    """
    带重试地执行 MCP 异步操作（连接就绪中可能瞬态失败）。

    仅在「非事件循环」环境调用（同步场景，如子进程管理）。
    若当前已处于运行中的事件循环，请改用 _run_with_retry_async。

    Args:
        server:       服务器配置
        coro_factory: 返回协程的工厂函数（每次重试重建客户端）
        retries:      最大重试次数
        interval:     重试间隔（秒）

    Returns:
        成功时返回结果，全部失败抛出最后一个异常。
    """
    import asyncio

    last_exc = None
    for attempt in range(retries):
        try:
            return asyncio.run(coro_factory())
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(interval)
    raise last_exc


async def _run_with_retry_async(server: dict, coro_factory, retries: int = 3, interval: float = 1.5):
    """
    _run_with_retry 的异步版本：在运行中的事件循环里带重试执行 MCP 异步操作。

    Args:
        server:       服务器配置
        coro_factory: 返回协程的工厂函数（每次重试重建客户端）
        retries:      最大重试次数
        interval:     重试间隔（秒）

    Returns:
        成功时返回结果，全部失败抛出最后一个异常。
    """
    import asyncio

    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(interval)
    raise last_exc


def list_tools(name: str) -> dict:
    """
    实时连接服务器并列出工具清单。
    返回 {"success": bool, "tools": [...], "message": str}
    """
    server = get_server(name)
    if server is None:
        return {"success": False, "tools": [], "message": f"服务器 {name} 不存在"}

    try:
        async def _list():
            client = _get_client(server)
            async with client as c:
                tools = await c.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": getattr(t, "description", "") or "",
                        "input_schema": getattr(t, "input_schema", None)
                        or getattr(t, "inputSchema", None) or {},
                    }
                    for t in tools
                ]

        # 命中缓存：工具 schema 静态，直接复用并附加实时启用状态（避免每次连接）
        cached = _tools_cache.get(name)
        if cached is not None:
            from config_loader import cfg
            tools = [dict(t) for t in cached]  # 浅拷贝，避免污染缓存
            for t in tools:
                t["enabled"] = bool(cfg(f"tools.{t['name']}.enabled", True))
            return {"success": True, "tools": tools, "message": f"共 {len(tools)} 个工具（缓存）"}

        tools = _run_with_retry(server, _list)
        # 缓存 schema（不含启用状态；启用状态每次实时读 config，见上）
        _tools_cache[name] = tools
        # 附加启用状态（config tools.<name>.enabled，默认启用），供前端展示开关
        from config_loader import cfg
        tools = [dict(t) for t in tools]
        for t in tools:
            t["enabled"] = bool(cfg(f"tools.{t['name']}.enabled", True))
        _log(name, "INFO", f"列出工具: {[t['name'] for t in tools]}")
        return {"success": True, "tools": tools, "message": f"共 {len(tools)} 个工具"}
    except Exception as e:
        _log(name, "ERROR", f"列出工具失败: {e}")
        return {"success": False, "tools": [], "message": f"连接失败: {e}"}


def call_tool(name: str, tool_name: str, arguments: dict) -> dict:
    """
    调用服务器上的指定工具。
    返回 {"success": bool, "result": ..., "message": str}
    """
    server = get_server(name)
    if server is None:
        return {"success": False, "result": None, "message": f"服务器 {name} 不存在"}

    try:
        async def _call():
            client = _get_client(server)
            async with client as c:
                result = await c.call_tool(tool_name, arguments)
                content = getattr(result, "content", None)
                return _parse_content(content)

        result = _run_with_retry(server, _call)
        _log(name, "INFO", f"调用工具 {tool_name}({json.dumps(arguments, ensure_ascii=False)}) 成功")
        return {"success": True, "result": result, "message": "调用成功"}
    except Exception as e:
        _log(name, "ERROR", f"调用工具 {tool_name} 失败: {e}")
        return {"success": False, "result": None, "message": f"调用失败: {e}"}


async def call_tool_async(name: str, tool_name: str, arguments: dict) -> dict:
    """
    调用服务器上的指定工具（异步版，供 FastAPI async 路由在事件循环内调用）。

    返回 {"success": bool, "result": ..., "message": str}
    """
    server = get_server(name)
    if server is None:
        return {"success": False, "result": None, "message": f"服务器 {name} 不存在"}

    try:
        async def _call():
            client = _get_client(server)
            async with client as c:
                result = await c.call_tool(tool_name, arguments)
                content = getattr(result, "content", None)
                return _parse_content(content)

        result = await _run_with_retry_async(server, _call)
        _log(name, "INFO", f"调用工具 {tool_name}({json.dumps(arguments, ensure_ascii=False)}) 成功")
        return {"success": True, "result": result, "message": "调用成功"}
    except Exception as e:
        _log(name, "ERROR", f"调用工具 {tool_name} 失败: {e}")
        return {"success": False, "result": None, "message": f"调用失败: {e}"}


def _parse_content(content):
    """解析 MCP 工具返回的 content。"""
    if isinstance(content, list) and content:
        # 可能多个 text 块
        texts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(text)
        return "\n".join(texts) if texts else content
    if isinstance(content, str):
        return content
    return content


# ============================================================================
# FastAPI 路由
# ============================================================================

def create_router():
    """创建 MCP 管理 API 的 FastAPI 路由。"""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter()

    class ServerIn(BaseModel):
        name: str
        command: str = ""
        args: str = ""
        url: str = ""
        transport: str = "streamable-http"
        auto_start: bool = False

    class CallIn(BaseModel):
        tool_name: str
        arguments: dict = {}

    # ---- 服务器 CRUD ----
    @router.get("/mcp/servers")
    def api_list_servers():
        servers = list_servers()
        # 附加运行状态
        for s in servers:
            s["_status"] = server_status(s["name"])
        return {"success": True, "servers": servers}

    @router.post("/mcp/servers")
    def api_upsert_server(body: ServerIn):
        try:
            server = upsert_server(body.dict())
            return {"success": True, "server": server}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/mcp/servers/{name}")
    def api_delete_server(name: str):
        delete_server(name)
        return {"success": True, "message": f"已删除服务器 {name}"}

    # ---- 生命周期 ----
    @router.post("/mcp/servers/{name}/start")
    def api_start_server(name: str):
        return start_server(name)

    @router.post("/mcp/servers/{name}/stop")
    def api_stop_server(name: str):
        return stop_server(name)

    @router.get("/mcp/servers/{name}/status")
    def api_server_status(name: str):
        return server_status(name)

    # ---- 工具交互 ----
    @router.get("/mcp/servers/{name}/tools")
    def api_list_tools(name: str):
        return list_tools(name)

    @router.post("/mcp/servers/{name}/call")
    def api_call_tool(name: str, body: CallIn):
        return call_tool(name, body.tool_name, body.arguments)

    # ---- 工具启用/禁用（config tools.<name>.enabled，全局生效） ----
    @router.post("/mcp/tools/{tool_name}/toggle")
    def api_toggle_tool(tool_name: str, body: dict):
        """切换某个 MCP 工具的启用状态。"""
        from config_loader import set_config
        enabled = bool(body.get("enabled", True))
        set_config(f"tools.{tool_name}.enabled", enabled)
        return {"success": True, "name": tool_name, "enabled": enabled}

    # ---- 日志 ----
    @router.get("/mcp/servers/{name}/logs")
    def api_get_logs(name: str, limit: int = 200):
        return {"success": True, "logs": get_logs(name, limit)}

    # ---- 功能开关（仅联网搜索） ----
    @router.get("/mcp/settings")
    def api_get_settings():
        """获取 MCP 功能开关（存于 config.json 的 mcp 段）。"""
        from config_loader import config
        mcp_cfg = config.get("mcp", {})
        return {
            "success": True,
            "features": mcp_cfg.get("features", {}),
        }

    @router.post("/mcp/settings")
    def api_set_settings(body: dict):
        """更新功能开关（仅联网搜索 websearch）。"""
        from config_loader import set_config
        if "features" in body:
            set_config("mcp.features", body["features"])
        return {"success": True}

    return router


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MCP 管理器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    from fastapi import FastAPI
    app = FastAPI(title="RAG MCP Manager")
    app.include_router(create_router())

    uvicorn.run(app, host=args.host, port=args.port)
