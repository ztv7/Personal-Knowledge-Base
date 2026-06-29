import requests
from typing import List, Tuple
from src.config import settings


def rerank(
    query: str, documents: List[str], top_n: int = None
) -> List[Tuple[int, str, float]]:
    """
    SiliconFlow BGE Reranker 重排序
    返回: [(原始下标, 文本, 相关性分数)]，按分数降序排列
    """
    if top_n is None:
        top_n = settings.rerank_top_n
    if not documents:
        return []

    headers = {
        "Authorization": f"Bearer {settings.siliconflow_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.siliconflow_rerank_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }

    try:
        resp = requests.post(
            settings.siliconflow_rerank_url,
            headers=headers,
            json=payload,
            timeout=settings.api_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        raise TimeoutError("Rerank API 请求超时，请检查网络或稍后重试")
    except requests.RequestException as e:
        raise RuntimeError(f"Rerank API 调用失败: {e}")

    results = []
    for item in data.get("results", []):
        idx = item["index"]
        score = item["relevance_score"]
        text = documents[idx]
        results.append((idx, text, score))

    # 按分数降序排序 (接口通常已排序，但确保一次)
    results.sort(key=lambda x: x[2], reverse=True)
    return results
