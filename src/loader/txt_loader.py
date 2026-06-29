import re
from typing import List, Tuple


def load_txt(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """纯文本按自然段落分页"""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("gbk", errors="replace")

    # 按空行分段
    paragraphs = re.split(r"\n\s*\n", text)
    # 过滤空白段
    pages = [
        (i + 1, p.strip())
        for i, p in enumerate(paragraphs)
        if p.strip()
    ]
    return pages or [(1, text)]
