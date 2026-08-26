# -*- coding: utf-8 -*-
"""
================================================================================
MCP 工具桥接 — 主程序通过 MCP 协议调用工具的唯一入口
================================================================================

架构约束：主程序「不持有」任何工具定义，所有工具都注册在 MCP 服务器上。
主程序（rag_graph 的工具决策 / 确定性联网）想要调用工具，只能通过本模块
连接 MCP 服务器执行，工具清单、参数 schema 全部从 MCP 动态获取。

提供两个能力：
  1. call_tool_by_name(tool_name, args) -> str
       直接调用 MCP 服务器上的单个工具，返回字符串结果（供确定性联网等
       无需 LLM 决策的固定调用使用）。
  2. get_mcp_tools_as_langchain(server_name=None, enabled_only=True)
       列出 MCP 服务器上的启用工具，动态构造成 langchain tool（含 name /
       description / args_schema），供 rag_graph 的「工具决策模型」bind_tools
       做 function calling 决策；工具实际执行时仍走 MCP（tool.invoke 内部
       转 call_tool_by_name）。

工具启用/禁用：统一读 config.json 的 ``tools.<name>.enabled``（默认启用）。
MCP 服务器端在工具函数体内做同样的检查（见 __main__.py 的 _require_enabled），
主程序决策端在 get_mcp_tools_as_langchain 里过滤，两端一致。

降级：MCP 服务器未启动 / 连接失败时，本模块返回空清单或失败文本，绝不抛
异常中断主链路。
================================================================================
"""

from __future__ import annotations

import json
from typing import List, Optional

# JSON Schema 类型 → Python 类型（用于动态构造 langchain args_schema）
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _pick_server_name() -> Optional[str]:
    """选择一个可用的 MCP 服务器名（默认服务器优先，其次第一个）。"""
    from mcp_service import manager
    servers = manager.list_servers()
    if not servers:
        return None
    for s in servers:
        if s.get("name") == "RAG-Service":
            return "RAG-Service"
    return servers[0].get("name") or None


def _is_tool_enabled(tool_name: str) -> bool:
    """读取工具启用状态（config tools.<name>.enabled，默认启用）。"""
    from config_loader import cfg
    return bool(cfg(f"tools.{tool_name}.enabled", True))


def call_tool_by_name(tool_name: str, args: dict, server_name: Optional[str] = None) -> str:
    """通过 MCP 协议调用单个工具，返回字符串结果。

    Args:
        tool_name:   工具名（如 web_search / calculate_pi）
        args:        工具参数 dict
        server_name: MCP 服务器名；None 时自动选择默认服务器

    Returns:
        工具结果字符串；失败时返回 "[工具 xxx 调用失败: ...]"。
    """
    from mcp_service import manager

    if server_name is None:
        server_name = _pick_server_name()
    if server_name is None:
        return f"[错误] 无可用 MCP 服务器"

    result = manager.call_tool(server_name, tool_name, args or {})
    if result.get("success"):
        data = result.get("result")
        # 结构化返回值统一转 JSON 字符串，保持与旧本地工具返回一致
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    return f"[工具 {tool_name} 调用失败: {result.get('message')}]"


def _schema_to_model(tool_name: str, input_schema: dict):
    """把 MCP 工具的 input_schema（JSON Schema）动态构造成 Pydantic 模型。

    用于 langchain StructuredTool 的 args_schema，使 bind_tools 能正确生成
    function calling 的参数 schema。
    """
    from pydantic import BaseModel, Field, create_model

    properties = (input_schema or {}).get("properties") or {}
    required = set((input_schema or {}).get("required") or [])

    fields = {}
    for pname, pspec in properties.items():
        if not isinstance(pspec, dict):
            continue
        ptype = pspec.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(ptype, str)
        desc = pspec.get("description", "") or ""
        if pname in required:
            fields[pname] = (py_type, Field(description=desc))
        else:
            default = pspec.get("default")
            fields[pname] = (py_type, Field(default=default, description=desc))

    if not fields:
        class _Empty(BaseModel):
            pass
        return _Empty

    safe = "".join(ch if ch.isalnum() else "_" for ch in tool_name)
    return create_model(f"{safe}_Args", **fields)


def _build_langchain_tool(server_name: str, tool_name: str, description: str,
                          input_schema: dict):
    """把单个 MCP 工具构造成 langchain tool（invoke 内部走 MCP）。"""
    from langchain_core.tools import StructuredTool

    ArgsModel = _schema_to_model(tool_name, input_schema)

    def _invoke(**kwargs):
        return call_tool_by_name(tool_name, kwargs, server_name=server_name)

    return StructuredTool.from_function(
        func=_invoke,
        name=tool_name,
        description=(description or tool_name),
        args_schema=ArgsModel,
    )


def get_mcp_tools_as_langchain(server_name: Optional[str] = None,
                               enabled_only: bool = True) -> List:
    """列出 MCP 服务器的启用工具，构造成 langchain tool 列表。

    Args:
        server_name:  MCP 服务器名；None 时自动选择默认服务器
        enabled_only: True 时只返回启用工具（读 config tools.<name>.enabled）

    Returns:
        langchain tool 列表（空列表 = 无可用工具 / 服务器不可达）。
    """
    from mcp_service import manager

    if server_name is None:
        server_name = _pick_server_name()
    if server_name is None:
        return []

    res = manager.list_tools(server_name)
    if not res.get("success"):
        return []

    tools = []
    for t in res.get("tools") or []:
        name = t.get("name")
        if not name:
            continue
        if enabled_only and not _is_tool_enabled(name):
            continue
        tools.append(_build_langchain_tool(
            server_name, name,
            t.get("description") or "",
            t.get("input_schema") or {},
        ))
    return tools
