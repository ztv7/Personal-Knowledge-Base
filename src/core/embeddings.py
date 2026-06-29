from openai import OpenAI
from src.config import settings


def get_embedding_fn():
    """返回一个 embedding 函数: List[str] -> List[List[float]]"""
    client = OpenAI(
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
    )

    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(
            model=settings.siliconflow_embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    return embed
