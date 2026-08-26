"""
================================================================================
MCP（Model Context Protocol）子包
================================================================================

本目录包含 MCP 相关的全部程序：

  __main__.py      MCP 服务器入口（可独立运行；主程序 --mcp 时随主程序启停）
  manager.py       MCP 服务器管理器（配置 CRUD + 生命周期管理 + 工具交互 + 日志）
  tool_bridge.py   主程序调用 MCP 工具的唯一桥接（工具清单动态拉取 + 单工具调用）
  websearch.py     联网搜索（Google News RSS → 百度，纯 HTTP 实现）
  math_tools.py    π 计算核心逻辑（Chudnovsky 算法）
  knowledge_tools.py  知识库检索工具封装（search_knowledge_base / list_knowledge_documents / get_knowledge_toc，供外部模型调用）

说明：所有工具都注册在 MCP 服务器上，主程序「不持有」工具定义，调用工具
一律通过 tool_bridge 连接 MCP 执行。

命名说明：
  本目录名为 mcp_service 而非 mcp，是为了避免与官方 `mcp` SDK 包
  （fastmcp 的依赖，import mcp.types）产生命名冲突——Windows 文件系统
  不区分大小写，若叫 mcp 会遮蔽官方 SDK 导致 fastmcp 无法启动。

注意：
  本包内的模块需要访问项目根目录的公共模块（llm / config_loader /
  embedding / milvus_store 等）以及 config/ 目录。因此在导入时，将
  项目根目录加入 sys.path，以便 import 父目录模块。
================================================================================
"""

import os
import sys

# 项目根目录（mcp_service 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保项目根目录在 sys.path 中（供本包内模块 import llm/config_loader 等）
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
