"""RAGFlow rag.app.picture 的轻量替代实现。

仅被 deepdoc 的视觉解析器在运行时延迟导入，本项目不启用视觉解析，
提供可导入的空实现即可。
"""


def vision_llm_chunk(*args, **kwargs):
    raise NotImplementedError(
        "视觉解析（vision_llm_chunk）在本项目未启用，请使用文本解析路径。"
    )
