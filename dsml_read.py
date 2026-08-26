"""
================================================================================
DSML 工具调用解析器 — DeepSeek V4 系列 agentic 模型兼容层
================================================================================

背景：
  DeepSeek-V4-Pro 等 agentic 模型在需要调用工具时，可能不遵循 OpenAI 的
  function calling 协议（不返回结构化 tool_calls 字段），而是把 DSML 文本
  标记直接输出到正文（content）里，例如：

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="web_search">
    <｜DSML｜parameter name="query" string="true">圆周率</｜DSML｜parameter>
    <｜DSML｜parameter name="num" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

本模块负责：
  1. 从正文中检测并解析 DSML 工具调用块（invoke 块 + parameter 参数）；
  2. 把解析结果转换为 OpenAI 兼容的结构化 tool_calls 列表；
  3. 剥离正文中的 DSML 标记，返回纯文本（用于展示/生成兜底）。

纯正则 + 标准库实现，不依赖任何第三方包，可被 rag_graph / tools 直接复用。

设计说明：
  - 兼容全角竖线「｜」与半角竖线「|」两种写法；
  - 兼容 parameter 的 string="true"（字符串值）与 string="false"（JSON/数字值）；
  - 兼容单个参数直接内嵌 JSON 的写法（<parameter name="arguments" string="false">...</parameter>）。
================================================================================
"""

import json
import re
from typing import List, Dict, Any, Tuple

# ============================================================================
# 正则：DSML 工具调用块解析
# ============================================================================

# 竖线分隔符（可选）：全角 ｜、半角 |、或无分隔符。
# 实际 DeepSeek 输出两种格式并存：
#   <DSML invoke name="...">          （不带竖线，DSML 与标签名间用空格）
#   <｜DSML｜invoke name="...">        （带全角竖线）
# 故用 [｜|]? 兼容「有/无」竖线；DSML 与标签名之间的分隔统一用 _GAP
# （可选竖线 + 可选空白），兼容「竖线分隔」与「空格分隔」两种写法。
_VBAR = r"[｜|]?"
_GAP = r"[｜|]?\s*"

# 匹配一个完整的 invoke 块：
#   <DSML invoke name="xxx"> ... </DSML invoke>
#   <｜DSML｜invoke name="xxx"> ... </｜DSML｜invoke>
# 兼容：
#   - name 属性带或不带引号
#   - 结束标签 </...DSML...invoke> 或 </...invoke...DSML...>
_INVOKE_BLOCK_RE = re.compile(
    r"<" + _VBAR + r"DSML" + _GAP + r"invoke\s+name=(?:"
    r"\"([^\"]*)\"|'([^']*)'|([^\s>]+)"
    r")[^>]*>(.*?)</" + _VBAR + r"DSML" + _GAP + r"invoke\s*>",
    re.DOTALL,
)

# 匹配一个 parameter 块：
#   <DSML parameter name="xxx" string="true|false">value</DSML parameter>
# 兼容 name / string 属性带或不带引号、顺序任意。
_PARAM_RE = re.compile(
    r"<" + _VBAR + r"DSML" + _GAP + r"parameter\b([^>]*)>(.*?)"
    r"</" + _VBAR + r"DSML" + _GAP + r"parameter\s*>",
    re.DOTALL,
)

# 从 attribute 文本中提取 name="xxx" 与 string="true/false"
_ATTR_NAME_RE = re.compile(r'\bname\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s]+))')
_ATTR_STRING_RE = re.compile(r'\bstring\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s]+))')

# 剥离任意 DSML 标记（完整块或残留的未闭合标签），用于兜底清理。
# 兼容 <DSML xxx> / </DSML xxx> / <｜DSML｜xxx> / </｜DSML｜xxx>。
_DSML_STRIP_RE = re.compile(r"</?" + _VBAR + r"DSML" + r"[^>]*>", re.DOTALL)


# ============================================================================
# 解析函数
# ============================================================================

def _parse_param_attrs(attrs_text: str) -> Tuple[str, str]:
    """
    从 parameter 标签的属性文本中提取 name 与 string 值。

    Returns:
        (name, string) 元组；name 或 string 缺失时对应值为空字符串。
    """
    name = ""
    string = ""
    m = _ATTR_NAME_RE.search(attrs_text)
    if m:
        name = next((g for g in m.groups() if g is not None), "")
    m = _ATTR_STRING_RE.search(attrs_text)
    if m:
        string = next((g for g in m.groups() if g is not None), "")
    return name.strip(), string.strip().lower()


def _decode_value(raw_value: str, is_string: str) -> Any:
    """
    按 string 属性解码参数值。

    - string="true"：直接返回字符串（去掉首尾空白）。
    - string="false"（或未标注）：尝试 JSON 解析（数字/布尔/对象/数组），
      失败则回退为字符串。
    """
    value = raw_value.strip()
    if is_string == "true":
        return value
    # string="false" 或未知：优先按 JSON 解析
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_dsml_tool_calls(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    从模型正文中解析 DSML 工具调用，返回（纯文本, 结构化 tool_calls）。

    参数：
        content: 模型输出的完整正文（可能包含 DSML 标记与普通文本）。

    返回：
        (normal_text, tool_calls)
          - normal_text: 剥离 DSML 标记后的纯文本（用于展示/兜底）；
          - tool_calls:  OpenAI 兼容的工具调用列表，每项：
                {"id", "name", "args", "type"}
            （其中 type 恒为 "function"，id 为自动生成；无 DSML 时为 []）。

    说明：
      - 若正文里同时包含普通文本与 DSML 块，普通文本会保留在 normal_text 中；
      - 支持一个正文里多个 invoke 块、每个块内多个 parameter；
      - 兼容 arguments 单参数内嵌 JSON 的写法。
    """
    if not content:
        return "", []

    tool_calls: List[Dict[str, Any]] = []

    def _sub_invoke(match: re.Match) -> str:
        name = next((g for g in match.groups()[:3] if g is not None), "")
        name = (name or "").strip()
        params_body = match.group(4) or ""

        # 解析 parameter 列表
        raw_params: Dict[str, Any] = {}
        param_texts: List[Tuple[str, str, str]] = []  # (name, string, value)
        for pm in _PARAM_RE.finditer(params_body):
            attrs = pm.group(1) or ""
            value = pm.group(2) or ""
            p_name, p_string = _parse_param_attrs(attrs)
            if not p_name:
                continue
            param_texts.append((p_name, p_string, value))

        # 特例：单个名为 arguments 的 parameter，其值为 JSON，直接展开
        if len(param_texts) == 1 and param_texts[0][0] == "arguments":
            _, _, arg_value = param_texts[0]
            try:
                expanded = json.loads(arg_value.strip())
                if isinstance(expanded, dict):
                    raw_params = expanded
                else:
                    raw_params = {"arguments": expanded}
            except (json.JSONDecodeError, ValueError):
                raw_params = {"arguments": arg_value.strip()}
        else:
            for p_name, p_string, value in param_texts:
                raw_params[p_name] = _decode_value(value, p_string)

        if not name:
            return ""  # 无工具名，无法构成有效调用

        tool_calls.append({
            "id": f"call_{len(tool_calls)}_{name}",
            "type": "function",
            "name": name,
            "args": raw_params,
        })
        return ""  # DSML 块从正文中移除

    # 1. 移除完整 invoke 块并解析
    normal_text = _INVOKE_BLOCK_RE.sub(_sub_invoke, content)

    # 2. 清理残留的 DSML 标记（tool_calls 外层包裹等）
    normal_text = _DSML_STRIP_RE.sub("", normal_text)
    normal_text = normal_text.strip()

    return normal_text, tool_calls


def strip_dsml(content: str) -> str:
    """
    剥离正文中的 DSML 标记（含未闭合标签），返回纯文本。

    用于：模型最终回答里仍残留 DSML 标记时的兜底清除。
    """
    if not content:
        return content
    # 移除完整的 DSML 块（invoke/parameter/tool_calls 等任意 DSML 标签及其内容）
    text = re.sub(
        r"</?" + _VBAR + r"DSML" + _VBAR + r".*?</" + _VBAR + r"DSML" + _VBAR + r".*?>",
        "",
        content,
        flags=re.DOTALL,
    )
    # 清理残留的未闭合 DSML 标签
    text = _DSML_STRIP_RE.sub("", text)
    return text


def has_dsml(content: str) -> bool:
    """判断正文中是否包含 DSML 标记（用于快速判断是否需要解析）。"""
    if not content:
        return False
    return ("<｜DSML" in content) or ("<|DSML" in content) or ("<DSML" in content)
