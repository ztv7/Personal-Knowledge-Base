from typing import List, Tuple, Optional
from src.core.splitter import HybridSplitter
from src.core.embeddings import get_embedding_fn
from src.loader.base import validate_file
from src.loader.pdf_loader import load_pdf
from src.loader.txt_loader import load_txt
from src.loader.md_loader import load_md
from src.config import settings

LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_md,
}


def load_and_chunk(
    file_bytes: bytes,
    filename: str,
    use_semantic: bool = True,
) -> Tuple[List[str], List[str], List[dict]]:
    """
    统一入口：验证 → 加载 → 切分
    返回: (ids, documents, metadatas)
    """
    ext, safe_name = validate_file(filename, file_bytes)

    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"不支持的文件格式: {ext}")

    page_data = loader(file_bytes, safe_name)
    if not page_data:
        raise ValueError(f"文件解析失败，未提取到文本内容: {safe_name}")

    splitter = HybridSplitter()

    ids, documents, metadatas = [], [], []

    for page_num, text in page_data:
        if use_semantic:
            embed_fn = get_embedding_fn()
            chunks = splitter.split_with_semantic(text, embed_fn)
        else:
            chunks = splitter.split(text)

        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            chunk_id = f"{safe_name}_p{page_num}_c{chunk_idx + 1}"
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append({
                "source": safe_name,
                "page": page_num,
                "chunk_index": chunk_idx + 1,
            })

    return ids, documents, metadatas
