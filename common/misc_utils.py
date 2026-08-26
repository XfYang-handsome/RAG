"""RAGFlow common.misc_utils 的轻量替代实现。"""
import asyncio


async def thread_pool_exec(fn, *args, **kwargs):
    """在线程池中执行同步函数，返回 awaitable。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def pip_install_torch(*args, **kwargs):
    """deepdoc 的 OCR 模块在需要 torch 时会调用它。

    本项目走轻量文本解析路径，不依赖 OCR/torch，因此留空实现。
    """
    return None
