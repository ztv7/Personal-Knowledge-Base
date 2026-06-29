from typing import List, Dict, Any, Optional
from src.store.vector_store import VectorStore
from src.core.reranker import rerank
from src.config import settings


def retrieve(
    store: VectorStore,
    query: str,
    top_k: int = None,
    rerank_top_n: int = None,
    where: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    向量检索 + Rerank 重排序
    返回: [{text, metadata, relevance_score}, ...]
    """
    if top_k is None:
        top_k = settings.top_k
    if rerank_top_n is None:
        rerank_top_n = settings.rerank_top_n

    # 1. 向量检索
    result = store.query(query_text=query, top_k=top_k, where=where)

    docs_list = result.get("documents", [[]])[0]
    meta_list = result.get("metadatas", [[]])[0]

    if not docs_list:
        return []

    # 2. Rerank
    rerank_items = rerank(query=query, documents=docs_list, top_n=rerank_top_n)

    # 3. 过滤低相关性，组装结果
    threshold = settings.relevance_threshold
    output = []
    for idx, text, score in rerank_items:
        if score < threshold:
            continue
        meta = meta_list[idx] if idx < len(meta_list) else {}
        output.append({
            "text": text,
            "metadata": meta,
            "relevance_score": round(score, 4),
        })

    return output


def format_context(retrieved: List[Dict[str, Any]]) -> str:
    """将检索结果格式化为 LLM 上下文"""
    if not retrieved:
        return "未找到相关文档片段。"
    blocks = []
    for item in retrieved:
        meta = item["metadata"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "-")
        score = item["relevance_score"]
        text = item["text"]
        blocks.append(
            f"【文档：{source} 第{page}页 | 相关度：{score:.4f}】\n{text}"
        )
    return "\n\n".join(blocks)
