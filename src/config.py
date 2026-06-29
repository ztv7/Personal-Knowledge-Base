import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- DeepSeek (LLM) ---
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(
        default="deepseek-v4-pro", alias="DEEPSEEK_MODEL"
    )

    # --- SiliconFlow (Embedding + Rerank) ---
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1", alias="SILICONFLOW_BASE_URL"
    )
    siliconflow_embedding_model: str = Field(
        default="Qwen/Qwen3-VL-Embedding-8B",
        alias="SILICONFLOW_EMBEDDING_MODEL",
    )
    siliconflow_rerank_url: str = Field(
        default="https://api.siliconflow.cn/v1/rerank",
        alias="SILICONFLOW_RERANK_URL",
    )
    siliconflow_rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="SILICONFLOW_RERANK_MODEL",
    )

    # --- ChromaDB ---
    chromadb_path: str = Field(
        default="chroma_db", alias="CHROMADB_CLIENT_PATH"
    )

    # --- Runtime overridable config ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.65
    top_k: int = 8
    rerank_top_n: int = 3
    temperature: float = 0.7
    relevance_threshold: float = 0.3

    # --- App ---
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20
    api_timeout_seconds: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# 全局单例，支持运行时修改
settings = Settings()


def update_runtime_config(**kwargs) -> None:
    """运行时覆盖配置，仅允许指定的可调参数"""
    allowed = {
        "chunk_size", "chunk_overlap", "similarity_threshold",
        "top_k", "rerank_top_n", "temperature", "relevance_threshold",
    }
    for k, v in kwargs.items():
        if k in allowed and hasattr(settings, k):
            setattr(settings, k, v)
