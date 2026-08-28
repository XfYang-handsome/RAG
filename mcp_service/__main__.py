"""
================================================================================
MCP 服务器 — 对外工具服务（HTTP transport）
================================================================================

基于 FastMCP 将「联网搜索」与「本地知识库检索」能力暴露为 MCP 工具，
通过 HTTP 协议对外服务，供外部模型（Claude Desktop / Cursor 等）接入。

工具列表：
  1. web_search(query, num)             联网搜索（Google News RSS 优先，百度备选）
  2. calculate_pi(digits)               高精度计算圆周率 π
  3. calculate_expression(latex, precision)  计算 LaTeX 算式结果（初等数学）
  4. search_knowledge_base(query, top_k, mode)  知识库检索（vector/hybrid/tree）
  5. list_knowledge_documents()         列出已入库文档
  6. get_knowledge_toc()                获取知识库目录结构

运行方式（HTTP）：
  python -m mcp_service --host 127.0.0.1 --port 8765

依赖：
  fastmcp
================================================================================
"""

import sys
import os

# 项目根目录（mcp 包的上一级）加入 sys.path，供 import websearch 等
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 关键修复：torch 必须作为本程序的第一个 import（最浅调用栈）完整加载。
# 否则后续 langchain_openai / transformers 触发的 torch 二次 import 会因
# torch._library.utils.get_source 在深层调用栈下 inspect 崩溃而报错
# （"partially initialized module 'torch'" / "Only a single TORCH_LIBRARY"）。
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    pass

from fastmcp import FastMCP

# ============================================================================
# MCP 服务器实例
# ============================================================================
mcp = FastMCP(
    name="PrismRAG-Service",
    instructions=(
        "PrismRAG 服务：提供联网搜索（web_search）、圆周率计算（calculate_pi）、"
        "LaTeX 算式计算（calculate_expression，初等数学）与本地知识库检索"
        "（search_knowledge_base / list_knowledge_documents / get_knowledge_toc）。"
        "知识库检索默认走混合检索（dense + BM25 + RRF），"
        "也可指定 vector（纯向量）或 tree（纯 LLM 树导航）。"
    ),
)


# ============================================================================
# 联网搜索（从独立模块导入，纯 HTTP 实现，不依赖 fastmcp）
# ============================================================================

from mcp_service.websearch import _web_search
from mcp_service.math_tools import compute_pi, calculate_expression as _calc_expression


# ============================================================================
# 工具启用/禁用检查（config tools.<name>.enabled，默认启用；
# 与主程序 tool_bridge 的过滤保持一致，两端同步）
# ============================================================================

def _require_enabled(name: str):
    """工具被禁用时抛异常，禁止调用。"""
    from config_loader import cfg
    if not bool(cfg(f"tools.{name}.enabled", True)):
        raise RuntimeError(f"工具 {name} 已禁用")


# ============================================================================
# MCP 工具定义
# ============================================================================

@mcp.tool
def web_search(query: str, num: int = 5) -> list:
    """
    联网搜索：优先 Google News，其次百度，两者均不可用时抛异常（调用方应跳过并记录错误）。

    Args:
        query: 搜索关键词
        num:   返回结果数量（默认 5）

    Returns:
        [{"title": 标题, "url": 链接, "snippet": 摘要, "engine": 来源引擎}, ...]
    """
    _require_enabled("web_search")
    results, engine = _web_search(query, num=num)
    for item in results:
        item["engine"] = engine
    return results


@mcp.tool
def calculate_pi(digits: int = 100) -> str:
    """
    高精度计算圆周率 π（Chudnovsky 算法）。

    Args:
        digits: 要计算的 π 小数位数（默认 100，最大 100000）

    Returns:
        π 的字符串（如 "3.141592653589793..."）
    """
    _require_enabled("calculate_pi")
    return compute_pi(digits)


@mcp.tool
def calculate_expression(latex: str, precision: int = 15) -> str:
    """
    计算数学算式的结果（初等数学 + 高等数学）。接受 LaTeX 或普通数学式子。

    若输入不是标准 LaTeX（不含 \\ 命令，如 "sqrt(4)+1/2"、"sin(pi/2)"），
    会自动先用 LLM 重写成 LaTeX 再计算。

    初等数学支持：四则运算、幂、根号、分数、括号、三角函数、反三角、对数、
          指数、常数 π/e、百分号、隐式乘法。

    高等数学支持：
          - 积分：不定积分 \\int f(x) dx、定积分 \\int_{a}^{b} f(x) dx、广义积分；
          - 导数：\\frac{d}{dx}、偏导 \\frac{\\partial}{\\partial x}；
          - 极限：\\lim_{x\\to 0} \\frac{\\sin x}{x} 等；
          - 级数：求和 \\sum、连乘 \\prod（含无穷级数）。

    线性代数支持：矩阵行列式 \\det、绝对值 \\left| x \\right|、纯矩阵字面量。

    Args:
        latex:     数学式子（LaTeX 或普通写法，普通写法会自动重写）
        precision: 数值结果有效数字位数（默认 15，最大 50）

    Returns:
        JSON 字符串：{"latex", "result", "expression"}；含变量时返回符号表达式；失败返回 {"error"}。
    """
    _require_enabled("calculate_expression")
    # 必须用别名 _calc_expression 调用核心函数，否则本函数名被 @mcp.tool 装饰器
    # 重新绑定后递归调用自身（与根目录 tools.py 的说明一致）。
    return _calc_expression(latex, precision)


# ============================================================================
# 知识库检索工具（封装 db_service，让外部模型也能检索本地知识库）
# ============================================================================

from mcp_service import knowledge_tools


@mcp.tool
def search_knowledge_base(query: str, top_k: int = 5, mode: str = "hybrid") -> list:
    """
    检索本地 RAG 知识库，返回与 query 最相关的文档片段。

    Args:
        query: 检索文本 / 问题
        top_k: 返回片段数量（默认 5）
        mode:  检索模式：vector=纯向量 / hybrid=混合检索（默认）/ tree=纯 LLM 树导航

    Returns:
        [{"text","score","doc_id","section_path","section_summary"}, ...]
    """
    _require_enabled("search_knowledge_base")
    return knowledge_tools.search_knowledge_base(query, top_k=top_k, mode=mode)


@mcp.tool
def list_knowledge_documents() -> list:
    """
    列出知识库中已入库的文档，帮助外部模型了解知识库覆盖范围。

    Returns:
        [{"doc_id","title","source","node_count","created_at"}, ...]
    """
    _require_enabled("list_knowledge_documents")
    return knowledge_tools.list_knowledge_documents()


@mcp.tool
def get_knowledge_toc() -> str:
    """
    获取知识库的目录结构（章节大纲），帮助判断问题落在哪个章节。

    Returns:
        目录文本（缩进表示层级，多文档用【文档】分隔）；知识库为空返回空串。
    """
    _require_enabled("get_knowledge_toc")
    return knowledge_tools.get_knowledge_toc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG MCP 服务器（HTTP）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--transport", default="streamable-http",
                        choices=["http", "streamable-http", "sse"],
                        help="传输协议（默认 streamable-http，fastmcp 3.x 推荐）")
    args = parser.parse_args()

    # 以 HTTP 方式启动（阻塞运行）
    mcp.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
