"""RAGFlow rag.prompts.generator 的轻量替代实现。

deepdoc 的视觉解析器（VisionParser 等）会调用这些 prompt 生成函数，
本项目走文本解析路径，仅需保证可导入、可调用。
"""


def vision_llm_describe_prompt(page=1):
    return (
        f"Describe the content of page {page} in detail, "
        "including all text, tables, and figures."
    )


def vision_llm_figure_describe_prompt(context=""):
    return "Describe this figure in detail." + (
        f"\nContext: {context}" if context else ""
    )


def vision_llm_figure_describe_prompt_with_context(context=""):
    return vision_llm_figure_describe_prompt(context)
