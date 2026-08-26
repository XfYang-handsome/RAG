#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from .docx_parser import RAGFlowDocxParser as DocxParser
from .epub_parser import RAGFlowEpubParser as EpubParser
from .excel_parser import RAGFlowExcelParser as ExcelParser
from .html_parser import RAGFlowHtmlParser as HtmlParser
from .json_parser import RAGFlowJsonParser as JsonParser
from .markdown_parser import MarkdownElementExtractor
from .markdown_parser import RAGFlowMarkdownParser as MarkdownParser

# pdf_parser 依赖 deepdoc.vision（onnxruntime / opencv / shapely 等）以及
# xgboost / scikit-learn 等重型库。这些库并非所有部署环境都具备，因此这里
# 做惰性降级：导入失败时仅让 PDF 解析器不可用，不影响其它轻量解析器。
try:
    from .pdf_parser import PlainParser
    from .pdf_parser import RAGFlowPdfParser as PdfParser
except Exception as _pdf_import_error:  # pragma: no cover
    import logging

    logging.warning(
        "deepdoc PDF parser unavailable (heavy deps missing): %s",
        _pdf_import_error,
    )
    PlainParser = None
    PdfParser = None

from .ppt_parser import RAGFlowPptParser as PptParser
from .txt_parser import RAGFlowTxtParser as TxtParser

__all__ = [
    "PdfParser",
    "PlainParser",
    "DocxParser",
    "EpubParser",
    "ExcelParser",
    "PptParser",
    "HtmlParser",
    "JsonParser",
    "MarkdownParser",
    "TxtParser",
    "MarkdownElementExtractor",
]
