# -*- coding: utf-8 -*-
"""纯数学工具核心逻辑（compute_pi 零依赖；calculate_expression 懒加载 sympy）。"""
import json
import math
import re
import sys
from decimal import Decimal, getcontext


def compute_pi(digits: int = 100) -> str:
    """高精度计算圆周率 π（Chudnovsky 算法 + decimal 高精度运算）。

    参数规整：非法/布尔值回退默认，超上限（100000）截断。
    """
    if isinstance(digits, bool) or not isinstance(digits, int) or digits < 1:
        digits = 100
    digits = min(digits, 100000)

    # Python 3.11+ 限制大整数转字符串的位数（默认 4300），放宽以支持高精度
    try:
        sys.set_int_max_str_digits(digits * 2 + 100)
    except Exception:
        pass

    # Chudnovsky 算法：每次迭代约增加 14 位精度，多加 2 项做冗余以保证截断后仍精确
    terms = max(math.ceil(digits / 14) + 2, 2)
    getcontext().prec = digits + 20

    c = Decimal(-640320) ** 3
    sq = Decimal(426880) * Decimal(10005).sqrt()

    he = Decimal(13591409)
    fleft = Decimal(1)
    fup = Decimal(13591409)
    fdown = Decimal(1)

    for k in range(1, terms + 1):
        fleft /= (k ** 3) * (3 * k) * (3 * k - 1) * (3 * k - 2)
        fleft *= 6 * k * (6 * k - 1) * (6 * k - 2) * (6 * k - 3) * (6 * k - 4) * (6 * k - 5)
        fup += 545140134
        fdown *= c
        he += fleft * fup / fdown

    pi = sq / he
    s = str(pi)
    # 截断到目标位数（去掉运算缓冲的冗余位）
    if "." in s:
        int_part, frac = s.split(".", 1)
        s = f"{int_part}.{frac[:digits]}"
    return s


# ---------------------------------------------------------------------------
# LaTeX 算式计算（初等数学 + 高等数学）
# ---------------------------------------------------------------------------
def _looks_like_latex(text: str) -> bool:
    """粗判输入是否已是 LaTeX（含反斜杠命令）。"""
    return "\\" in text


def _parse_matrix_body(body):
    """把矩阵环境 body（按 \\\\ 分行、& 分列）解析成 sympy.Matrix；失败返回 None。"""
    from sympy.parsing.latex import parse_latex
    import sympy
    rows = body.split("\\\\")
    cells = []
    for row in rows:
        parsed = []
        for c in row.split("&"):
            c = c.strip()
            if not c:
                return None
            try:
                e = parse_latex(c)
                e = e.subs({sympy.Symbol("pi"): sympy.pi, sympy.Symbol("e"): sympy.E})
                parsed.append(e)
            except Exception:
                return None
        cells.append(parsed)
    if not cells or any(len(r) != len(cells[0]) for r in cells):
        return None
    return sympy.Matrix(cells)


def _matrix_det_to_latex(body):
    """把矩阵环境 body 解析成 Matrix 并求行列式，返回其 LaTeX 字符串；失败返回 None。"""
    import sympy
    mat = _parse_matrix_body(body)
    if mat is None:
        return None
    try:
        val = sympy.simplify(mat.det())
        return sympy.latex(val)
    except Exception:
        return None


def _preprocess_latex(latex: str) -> str:
    """预处理 sympy.parse_latex 不支持的线性代数 LaTeX 记号。

    1. 绝对值：\\left| ... \\right| 或 \\lvert ... \\rvert → | ... |（sympy 原生支持 |x|）；
    2. 行列式：\\det 后跟矩阵环境，或裸 \\begin{vmatrix} / \\begin{Vmatrix}，
       直接求行列式并替换回数值/符号的 LaTeX（后续走正常解析求值）。
    3. 矩阵字面量不在此处理（由 calculate_expression 解析失败时兜底）。
    """
    # 绝对值
    latex = re.sub(r"\\left\s*\|(.*?)\\right\s*\|", r"|\1|", latex, flags=re.DOTALL)
    latex = latex.replace("\\lvert", "|").replace("\\rvert", "|")

    def _det_repl(m):
        out = _matrix_det_to_latex(m.group(2))
        return out if out is not None else m.group(0)

    # \det 后跟矩阵环境 → 行列式
    latex = re.sub(
        r"\\det\s*\\begin\{(vmatrix|Vmatrix|pmatrix|bmatrix|matrix|Bmatrix)\}(.*?)\\end\{\1\}",
        _det_repl, latex, flags=re.DOTALL,
    )
    # 裸竖线矩阵（vmatrix / Vmatrix）= 行列式
    latex = re.sub(
        r"\\begin\{(vmatrix|Vmatrix)\}(.*?)\\end\{\1\}",
        _det_repl, latex, flags=re.DOTALL,
    )
    return latex


def _match_full_matrix(latex):
    """若整个输入就是一个矩阵环境，返回 sympy.Matrix；否则返回 None。"""
    s = latex.strip()
    m = re.fullmatch(
        r"\\begin\{(matrix|pmatrix|bmatrix|vmatrix|Vmatrix|Bmatrix)\}(.*)\\end\{\1\}",
        s, flags=re.DOTALL,
    )
    if not m:
        return None
    return _parse_matrix_body(m.group(2))


def _rewrite_to_latex(text: str, llm=None, precision: int = 15) -> str:
    """用 LLM 把「非标准 LaTeX 的数学式子」重写成标准 LaTeX。

    适用场景：用户输入普通数学式子（如 "sqrt(4)+1/2"、"sin(pi/2)"、"2^10"、
    "(1+2)*3"），这些不是标准 LaTeX 命令，sympy.parse_latex 可能解析失败或
    解析错。此函数让 LLM 改写成标准 LaTeX（\\frac、\\sqrt、\\sin、^ 等）。

    Args:
        text:      待重写的数学式子
        llm:       重写用 LLM；None 时内部通过 llm_factory 懒加载（tool_llm 优先）
        precision: 透传给下层（保留签名一致性，实际未用）

    Returns:
        标准 LaTeX 字符串；LLM 不可用 / 失败时返回原 text（由上层降级处理）。
    """
    if llm is None:
        try:
            from llm_factory import get_model
            llm = get_model("tool_llm", "llm")
        except Exception:
            llm = None
    if llm is None:
        return text

    from llm import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是数学 LaTeX 重写助手。请把用户输入的数学式子改写成标准 LaTeX 表达式。"
        "规则：\n"
        "1. 分数用 \\\\frac{分子}{分母}；\n"
        "2. 根号用 \\\\sqrt{...}，高次根用 \\\\sqrt[n]{...}；\n"
        "3. 乘方用 ^（如 2^10），下标用 _（如 \\\\log_{10}）；\n"
        "4. 函数写成 \\\\sin \\\\cos \\\\tan \\\\log \\\\ln 等，参数放花括号或括号内；\n"
        "5. 圆周率写 \\\\pi，自然常数写 e；\n"
        "6. 积分写 \\\\int（定积分 \\\\int_{a}^{b}），微分写 dx；\n"
        "7. 求和写 \\\\sum_{i=a}^{b}，极限写 \\\\lim_{x\\\\to a}，导数写 \\\\frac{d}{dx}；\n"
        "8. 保留原式子的数值与运算含义，不要改变结果；\n"
        "只输出 LaTeX 表达式本身，不要任何解释、不要引号、不要代码块。"
    )
    prompt = f"数学式子：{text}\n\n请输出对应的标准 LaTeX 表达式。"
    try:
        out = invoke_llm(llm, [SystemMessage(content=system),
                               HumanMessage(content=prompt)]).strip()
        # 先取首行（丢弃多余解释），再去首尾引号
        out = out.split("\n")[0].strip()
        out = out.strip('"\'`')
        return out if out else text
    except Exception:
        return text


def calculate_expression(latex: str, precision: int = 15, rewrite_llm=None) -> str:
    """计算 LaTeX 算式结果（初等数学 + 高等数学），返回 JSON 字符串。

    初等数学：四则运算（+ - × ÷）、幂、根号（\\sqrt，含高次根 \\sqrt[n]{}）、
    分数（\\frac{a}{b}）、括号（含 \\left( \\right)）、三角函数（\\sin \\cos \\tan 等）、
    反三角、对数（\\log \\ln）、指数（e^x \\exp）、常数（\\pi、e）、
    百分号、隐式乘法（2\\pi）。

    高等数学（微积分 / 级数）：
    - 积分：不定积分 \\int x^2 dx、定积分 \\int_{0}^{1} x^2 dx、广义积分 \\int_{0}^{\\infty} e^{-x} dx；
    - 导数：一阶导数 \\frac{d}{dx} x^2、\\frac{d}{dx} \\sin x；
    - 偏导：一阶偏导 \\frac{\\partial}{\\partial x} x^2 y；
    - 极限：\\lim_{x\\to 0} \\frac{\\sin x}{x}、\\lim_{x\\to\\infty} \\frac{1}{x}；
    - 级数：求和 \\sum_{i=1}^{n} i（含无穷级数 \\sum_{n=1}^{\\infty} \\frac{1}{n^2}）、
      连乘 \\prod_{i=1}^{n} i。

    线性代数（预处理支持，sympy.parse_latex 原生不认矩阵环境 / \\left|...\\right|）：
    - 矩阵行列式：\\det \\begin{pmatrix}...\\end{pmatrix}、\\begin{vmatrix}...\\end{vmatrix}；
    - 绝对值：\\left| x \\right|、\\lvert x \\rvert（转成 |x| 后解析为 Abs）；
    - 纯矩阵字面量：直接输入 \\begin{pmatrix}...\\end{pmatrix} 返回矩阵本身。

    实现：sympy.parsing.latex.parse_latex 解析（纯数学解析，非 eval，安全），
    常数替换后，对 Integral/Derivative/Limit/Sum/Product 显式调用 doit() 求值
    （否则只会返回 Integral(x**2, x) 这类未求值符号），再按需数值化。
    结果含变量时返回符号表达式。

    非标准输入自动重写：若输入不含 LaTeX 命令（如 "sqrt(4)+1/2"、"sin(pi/2)"、
    "2^10"、"int_0^1 x^2 dx"），先用 LLM 重写成标准 LaTeX 再计算；重写失败则尝试直接解析兜底。

    Args:
        latex:       LaTeX 算式，如 "\\frac{1}{2}+\\sqrt{4}"、"\\sin(\\frac{\\pi}{2})"、
                     "\\int_{0}^{1} x^2 dx"；也接受普通数学式子（自动重写）
        precision:   数值结果的有效数字位数（默认 15，最大 50）
        rewrite_llm: 重写用 LLM（可选；None 时内部懒加载 tool_llm/llm）

    Returns:
        JSON 字符串：
          - 纯数值：{"latex": ..., "result": 数值, "expression": 精确形式}
          - 含变量：{"latex": ..., "result": 符号表达式, "note": 说明}
          - 失败：   {"error": 原因}
    """
    # ---- 输入校验 ----
    if isinstance(latex, bool) or not latex or not str(latex).strip():
        return json.dumps({"error": "算式为空，请输入 LaTeX 算式"}, ensure_ascii=False)
    original = str(latex).strip()
    if len(original) > 300:
        return json.dumps({"error": "算式过长（>300 字符）"}, ensure_ascii=False)

    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
        precision = 15
    precision = min(precision, 50)

    # ---- 懒加载 sympy（解析 LaTeX 需要，首次调用才 import）----
    import sympy
    from sympy.parsing.latex import parse_latex

    # ---- 预处理：sympy 不支持的常见 LaTeX 记号 ----
    # 百分号：10\% = 10/100
    latex = original.replace("\\%", "/100")

    # ---- 非标准输入自动重写（不含 LaTeX 命令 → 先用 LLM 改写）----
    rewritten = False
    if not _looks_like_latex(latex):
        candidate = _rewrite_to_latex(latex, llm=rewrite_llm, precision=precision)
        if candidate and candidate != latex:
            latex = candidate
            rewritten = True

    # ---- 预处理：绝对值 / 矩阵行列式（sympy.parse_latex 不支持）----
    latex = _preprocess_latex(latex)

    # ---- 解析 ----
    try:
        expr = parse_latex(latex)
    except Exception as e:
        # 兜底：整个输入就是一个矩阵环境 → 直接返回矩阵
        mat = _match_full_matrix(latex)
        if mat is not None:
            return json.dumps({
                "latex": original,
                "result": sympy.latex(mat),
                "note": "矩阵（sympy 不支持解析矩阵环境，已单独返回）",
            }, ensure_ascii=False)
        return json.dumps({"error": f"LaTeX 解析失败：{e}"}, ensure_ascii=False)

    # ---- 常数替换：\pi、e 是数学常数而非变量 ----
    expr = expr.subs({
        sympy.Symbol("pi"): sympy.pi,
        sympy.Symbol("e"): sympy.E,
    })

    # ---- 高等数学求值：Integral / Derivative / Limit / Sum / Product ----
    # parse_latex 对 \int、\frac{d}{dx}、\lim、\sum、\prod 会解析成未求值的
    # sympy 对象，需显式 doit() 求值（如 \int x^2 dx → x^3/3、
    # \lim_{x\to0} sin(x)/x → 1、\sum_{i=1}^{10} i → 55）。
    # 失败则保持原对象，走下方符号分支返回。
    from sympy import Integral, Derivative, Limit, Sum, Product
    if isinstance(expr, (Integral, Derivative, Limit, Sum, Product)):
        try:
            expr = expr.doit()
        except Exception:
            pass

    # ---- 含变量：返回符号表达式（不数值化）----
    if expr.free_symbols:
        vars_str = ", ".join(sorted(str(s) for s in expr.free_symbols))
        out = {
            "latex": latex,
            "result": str(expr),
            "note": f"结果含变量 {vars_str}，已按符号表达式返回",
        }
        if rewritten:
            out["rewritten_from"] = original
        return json.dumps(out, ensure_ascii=False)

    # ---- 纯数值求值 ----
    try:
        num = sympy.N(expr, precision + 5)  # 多算几位防截断误差
    except Exception as e:
        return json.dumps({"error": f"求值失败：{e}"}, ensure_ascii=False)

    # ---- 格式化数值 ----
    # 用 Python float + '.Ng' 格式：自动去掉尾随 0、整数显示整数（如 2 而非 2.0）。
    # 注意：sympy 的 Float == Integer 是「结构比较」恒为 False，不能靠它判断整数，
    # 所以转成 Python float 数值判断。
    try:
        fnum = float(num)
        if abs(fnum) < 1e15:
            result = format(fnum, f".{precision}g")
        else:
            result = str(num.evalf(precision))
    except (ValueError, OverflowError, TypeError):
        # 非实数（复数/无穷）等：退回 sympy 字符串
        result = str(num.evalf(precision))

    out = {
        "latex": latex,
        "result": result,
        "expression": str(expr),
    }
    if rewritten:
        out["rewritten_from"] = original
    return json.dumps(out, ensure_ascii=False)
