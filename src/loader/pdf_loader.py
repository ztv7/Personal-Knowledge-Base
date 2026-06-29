import fitz
from typing import List, Tuple


def load_pdf(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """返回 [(页码, 文本内容), ...]"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages
