import re
from typing import List, Tuple


def load_md(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Markdown 按 ## 标题分节"""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("gbk", errors="replace")

    # 按 ## 标题切分 (保留标题作为内容前缀)
    sections = re.split(r"\n(?=#{1,3}\s)", text)
    pages = [
        (i + 1, s.strip())
        for i, s in enumerate(sections)
        if s.strip()
    ]
    return pages or [(1, text)]
