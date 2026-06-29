from langchain_openai import ChatOpenAI
from src.config import settings


def get_llm(
    temperature: float = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """获取 LangChain ChatOpenAI 实例，连接 DeepSeek"""
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature if temperature is not None else settings.temperature,
        streaming=streaming,
        timeout=settings.api_timeout_seconds,
        max_retries=1,
    )
