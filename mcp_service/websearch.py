"""
================================================================================
联网搜索辅助（Google News RSS 优先，百度网页备选）
================================================================================

纯 HTTP 实现，不依赖 fastmcp，可被主程序与 MCP 服务器共用。

搜索源说明：
  - Google News RSS（news.google.com/rss/search）：返回纯 XML、结构稳定、
    对时效性/新闻类查询召回质量高（需走代理，国内直连超时）；
  - 百度网页搜索（www.baidu.com/s）：国内可直连，作为 Google 失败时的兜底，
    对中文/英文查询均有不错的召回。

注意：
  本模块被主程序（server.py）与 MCP 服务器（mcp/__main__.py）共同 import。
  放在独立模块中，避免主程序 import mcp_service.__main__ 时触发 fastmcp 的
  server 依赖（fastmcp 的 FastMCP 需要额外安装 server extra）。
================================================================================
"""

import time as _time


def _log(level: str, msg: str):
    """打印一条日志到控制台（与主程序 [HH:MM:SS] [LEVEL] 格式保持一致）。

    Windows 控制台默认 GBK，遇到网页标题里的特殊字符（如 \\ue6a3 私用区字符）
    会抛 UnicodeEncodeError，导致联网搜索整个崩掉。这里做安全编码兜底。
    """
    try:
        print(f"[{_time.strftime('%H:%M:%S')}] [{level}] {msg}")
    except UnicodeEncodeError:
        safe = (msg or "").encode("utf-8", "replace").decode("gbk", "replace")
        try:
            print(f"[{_time.strftime('%H:%M:%S')}] [{level}] {safe}")
        except Exception:
            pass

import re
import html as _html
import urllib.parse as _urlparse

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _get_web_search_config() -> dict:
    """读取联网搜索配置（proxy / timeout）。"""
    try:
        from config_loader import config
        return config.get("mcp", {}).get("web_search", {}) or {}
    except Exception:
        return {}


def _fetch_google_news(query: str, num: int, timeout: float):
    """
    请求 Google News RSS 接口并解析，失败返回 None。

    Google News RSS（news.google.com/rss/search）返回纯 XML，结构稳定、
    无需 JS 渲染，且对时效性/新闻类查询的召回质量远高于百度网页搜索
    （后者对「最新进展」类查询常返回百科词条/期刊主页）。需走代理访问
    （国内直连超时）。返回高质量新闻标题 + 来源 + 日期。
    """
    try:
        import requests
        cfg = _get_web_search_config()
        proxy = cfg.get("proxy") or ""
        sess = requests.Session()
        sess.trust_env = False  # 禁用系统代理（改用显式配置的代理）
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}
        # hl/en-US + gl/US + ceid=US:en 用英文源，召回更准（中文整句常召回为空）
        url = ("https://news.google.com/rss/search?q=" + _urlparse.quote(query)
               + "&hl=en-US&gl=US&ceid=US:en")
        r = sess.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        if r.status_code != 200:
            return None
        return _parse_google_news_rss(r.text, num)
    except Exception:
        return None


def _parse_google_news_rss(xml_text: str, num: int = 5):
    """从 Google News RSS 的 XML 中提取标题、真实源链接、来源、日期。"""
    results = []
    item_re = re.compile(r'<item>(.*?)</item>', re.S)
    for block in item_re.findall(xml_text):
        t_m = re.search(r'<title>(.*?)</title>', block, re.S)
        d_m = re.search(r'<pubDate>(.*?)</pubDate>', block, re.S)
        s_m = re.search(r'<source\s+url="([^"]+)"[^>]*>(.*?)</source>', block, re.S)
        if not t_m:
            continue
        title = _html.unescape(t_m.group(1)).strip()
        # 去掉 title 末尾的 " - 来源名" 后缀（如 "xxx - The Conversation"）
        if " - " in title:
            title = re.sub(r'\s*-\s*[^-]+$', '', title).strip()
        date = d_m.group(1) if d_m else ""
        url = _html.unescape(s_m.group(1)).strip() if s_m else ""
        source = _html.unescape(s_m.group(2)).strip() if s_m else ""
        # 组装 snippet：来源 + 日期
        snippet = " · ".join(x for x in (source, date) if x)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= num:
            break
    return results


def _fetch_baidu(query: str, num: int, timeout: float):
    """请求百度搜索结果页并解析，失败返回 None（国内可直连，通常无需代理）。

    百度对高频请求会返回「百度安全验证」页（无 h3 结果），此时解析结果为空，
    由上层回退到 Bing。
    """
    try:
        import requests
        cfg = _get_web_search_config()
        proxy = cfg.get("proxy") or ""
        sess = requests.Session()
        sess.trust_env = False
        if proxy:
            sess.proxies = {"http": proxy, "https": proxy}
        url = "https://www.baidu.com/s?wd=" + _urlparse.quote(query) + f"&rn={num}"
        r = sess.get(url, headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}, timeout=timeout)
        if r.status_code != 200:
            return None
        html = r.text
        # 反爬检测：返回「安全验证」页时视为失败
        if "安全验证" in html or "wappass" in html or len(html) < 5000:
            return None
        return _parse_baidu_html(html)
    except Exception:
        return None


def _fetch_bing(query: str, num: int, timeout: float):
    """请求必应搜索结果页并解析，失败返回 None。

    语言感知：英文 query 走 www.bing.com（en-US 市场，召回英文源，技术类内容质量高）；
    中文/日文 query 走 cn.bing.com（zh-CN，中文专名分词准确、国内直连稳定）。
    """
    try:
        import requests
        sess = requests.Session()
        sess.trust_env = False  # Bing 直连，不走代理

        if _contains_cjk(query):
            domain = "cn.bing.com"
            extra = "&mkt=zh-CN&setlang=zh-CN"
            accept_lang = "zh-CN,zh;q=0.9,en;q=0.8"
        else:
            # 英文 query：ensearch=1 是唯一能在国内 IP 下强制英文结果的参数
            # （mkt/setlang/cc 均被 Bing 按 IP 重定向到中文市场而失效）
            domain = "www.bing.com"
            extra = "&ensearch=1"
            accept_lang = "en-US,en;q=0.9"

        url = ("https://" + domain + "/search?q=" + _urlparse.quote(query)
               + f"&count={num}" + extra)
        r = sess.get(url, headers={
            "User-Agent": _UA,
            "Accept-Language": accept_lang,
        }, timeout=timeout)
        if r.status_code != 200:
            return None
        return _parse_bing_html(r.text, num)
    except Exception:
        return None


def _parse_bing_html(html_text: str, num: int = 5):
    """从 Bing 搜索结果页提取标题、链接、摘要。

    Bing 结构：
      - 结果容器：<li class="b_algo">
      - 标题：<h2><a href="真实链接">标题</a></h2>
      - 摘要：<div class="b_caption"><p>...</p></div>
    """
    results = []
    # 按 b_algo 块切分，逐个提取
    blocks = re.split(r'<li class="b_algo"', html_text)[1:]
    for block in blocks:
        # 标题 + 链接
        tm = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not tm:
            continue
        url = _html.unescape(tm.group(1)).strip()
        title = _html.unescape(re.sub(r"<[^>]+>", "", tm.group(2))).strip()
        if not title:
            continue
        # 摘要（b_caption 内的 p）
        snippet = ""
        cm = re.search(r'<div class="b_caption"[^>]*>(.*?)</div>', block, re.S)
        if cm:
            pm = re.search(r"<p[^>]*>(.*?)</p>", cm.group(1), re.S)
            if pm:
                snippet = _html.unescape(re.sub(r"<[^>]+>", "", pm.group(1))).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= num:
            break
    return results


def _parse_baidu_html(html_text: str):
    """从百度搜索结果页提取标题、真实链接（摘要尽力而为）。

    百度新版（cos 版）的真实链接放在 JS 变量 bds.comm.iaurl 数组里，
    按结果顺序与 <h3> 标题一一对应；标题 a 标签的 href 是百度跳转链接
    （link?url=...），作为无 iaurl 条目时的兜底 url。
    """
    results = []

    # 1. 真实链接数组：bds.comm.iaurl=["url1","url2",...]（含 JS 转义 \/）
    iaurl = []
    m = re.search(r'bds\.comm\.iaurl\s*=\s*\[([^\]]*)\]', html_text)
    if m:
        iaurl = [u.replace("\\/", "/") for u in re.findall(r'"([^"]*)"', m.group(1))]

    # 2. 标题 + 跳转链接：<h3 ...><a href="...">标题</a></h3>
    title_items = []
    for h3m in re.finditer(r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.S):
        href = h3m.group(1)
        title = _html.unescape(re.sub(r'<[^>]+>', '', h3m.group(2))).strip()
        if title:
            title_items.append((title, href))

    # 3. 组装：真实链接优先，跳转链接兜底
    for i, (title, href) in enumerate(title_items):
        url = iaurl[i] if i < len(iaurl) else href
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= len(title_items):
            break
    return results


def _contains_cjk(text: str) -> bool:
    """判断文本是否含中文/日文（CJK 统一表意 + 假名）。"""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff":
            return True
    return False


def _is_definition_query(text: str) -> bool:
    """判断是否为百科/定义类问题（「XX 是什么/介绍/定义」）。"""
    return bool(re.search(r"(是什么|什么是|介绍|定义|什么意思|含义|简介|概述|科普|怎么回事)", text))


# 疑问词 / 泛化词（作为 token 尾部剔除，避免「纸上的魔法使 是什么」搜不到）
_QUERY_STOPWORDS = (
    "是什么", "什么是", "什么意思", "介绍", "定义", "含义", "简介", "概述",
    "怎么样", "如何", "为什么", "是谁", "游戏", "动漫", "作品", "剧情",
)

def _sanitize_query(query: str) -> str:
    """清洗搜索 query：多语言专名去重 + 去疑问词/泛化词。

    场景：用户问「《纸上的魔法使》（紙の上の魔法使い）是什么」，LLM 提炼
    可能输出「纸上的魔法使 紙の上の魔法使い 是什么」。日文假名 token 与
    疑问词 token 都会让百度召回失败，故统一清洗：

      - 丢弃「含日文假名」的 token（保留纯中文/英文，优先中文）
      - 剔除 token 尾部的疑问词/泛化词（「是什么」「游戏」等）
      - 若清洗后为空（如纯日文查询「君の名は」），返回原 query 不丢
    """
    q = (query or "").strip()
    if not q:
        return q
    tokens = re.split(r"[\s,，、;；]+", q)
    filtered = []
    for t in tokens:
        if not t:
            continue
        # 1. 丢弃含日文假名的 token（同一专名的日文写法）
        if any("\u3040" <= c <= "\u30ff" for c in t):
            continue
        # 2. 剔除疑问词/泛化词（整个 token 等于或尾部为疑问词）
        for sw in _QUERY_STOPWORDS:
            if t == sw:
                t = ""
                break
            if t.endswith(sw) and len(t) > len(sw):
                t = t[: -len(sw)]
                break
        if not t:
            continue
        filtered.append(t)
    if filtered:
        return " ".join(filtered).strip()
    return q


def _web_search(query: str, num: int = 5, timeout: float = None):
    """
    联网搜索：按查询类型智能选择搜索源。

    timeout 未指定时从 config 的 mcp.web_search.timeout 读取（默认 6 秒）。
    返回 (结果列表, 引擎名)。

    搜索源选择策略：
      - 定义/百科类问题 或 含中文/日文 → 百度优先（百科召回强、中文匹配准）
      - 纯英文非定义问题 → Google News RSS 优先（时效/新闻召回质量高）
      - 优先源失败/无结果时回退另一源

    两者均不可用时抛 RuntimeError（调用方应跳过并记录）。
    """
    if timeout is None:
        timeout = float(_get_web_search_config().get("timeout", 6.0))

    # 查询清洗：同一专名多语言写法只保留中文（如去掉日文假名 token）
    query = _sanitize_query(query)
    if not query:
        raise RuntimeError("搜索关键词为空")

    baidu_first = _is_definition_query(query) or _contains_cjk(query)
    gn_timeout = min(timeout, 4.0)

    # 统一回退链：优先源 → Bing（稳定兜底）→ 最后源
    # Bing 直连可用、中文专名准确、无百度式安全验证，是最可靠的兜底。
    def _baidu():
        try:
            return _fetch_baidu(query, num, timeout)
        except Exception:
            return None

    def _bing():
        try:
            return _fetch_bing(query, num, timeout)
        except Exception:
            return None

    def _google_news():
        try:
            return _fetch_google_news(query, num, gn_timeout)
        except Exception:
            return None

    if baidu_first:
        # 定义类 / 中文：百度优先 → Bing → Google News
        results = _baidu()
        if results:
            return results, "baidu"
        _log("WARN", f"百度无结果/被反爬，回退 Bing: {query}")
        results = _bing()
        if results:
            return results, "bing"
        _log("WARN", f"Bing 无结果，回退 Google News: {query}")
        results = _google_news()
        if results:
            return results, "google_news"
    else:
        # 纯英文非定义：Google News 优先 → Bing → 百度
        results = _google_news()
        if results:
            return results, "google_news"
        _log("WARN", f"Google News 无结果，回退 Bing: {query}")
        results = _bing()
        if results:
            return results, "bing"
        _log("WARN", f"Bing 无结果，回退百度: {query}")
        results = _baidu()
        if results:
            return results, "baidu"

    raise RuntimeError("Google News、Bing 与百度均不可用（超时/无结果/被拦截）")


# ---------------------------------------------------------------------------
# 网页正文抓取（供 Agentic 联网证据补全正文，纯 HTTP + 正则，不引入新依赖）
# ---------------------------------------------------------------------------
def _extract_body_text(html_text: str) -> str:
    """从 HTML 提取正文纯文本（启发式）。

    顺序：去 script/style/noscript → 去注释 → 优先取 article/main 主体 →
    提取 <p> 段落 → 去所有标签 → HTML 反转义 → 压缩空白。
    """
    # 去脚本 / 样式 / noscript（含内容）
    html_text = re.sub(
        r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', html_text, flags=re.S | re.I
    )
    # 去注释
    html_text = re.sub(r'<!--.*?-->', ' ', html_text, flags=re.S)

    # 优先取 article / main 主体（正文通常集中在这些容器里）
    for tag in ('article', 'main'):
        m = re.search(
            r'<{0}[^>]*>(.*?)</{0}>'.format(tag), html_text, flags=re.S | re.I
        )
        if m:
            html_text = m.group(1)
            break

    # 提取段落文本；无段落则退化为全文本
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html_text, flags=re.S | re.I)
    text = '\n'.join(paras) if paras else html_text

    # 去剩余所有标签
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _html.unescape(text)

    # 压缩空白：行内空格归并、多余空行归并
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()


def fetch_page_content(url: str, timeout: float = None, max_chars: int = 2000) -> str:
    """抓取网页正文，返回纯文本（失败返回空串，由调用方降级用 title+snippet）。

    - timeout 未指定时从 config 的 mcp.web_search.timeout 读取。
    - max_chars 截断正文，控制下游 reranker / 合成 prompt 的体积。
    - 忽略系统代理（trust_env=False），与搜索抓取保持一致。
    """
    if not url:
        return ""
    if timeout is None:
        try:
            timeout = float(_get_web_search_config().get("timeout", 6.0))
        except Exception:
            timeout = 6.0

    try:
        import requests
        sess = requests.Session()
        sess.trust_env = False
        r = sess.get(
            url,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return ""
        body = _extract_body_text(r.text)
        return body[:max_chars] if body else ""
    except Exception:
        return ""
