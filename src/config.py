import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL","https://api.siliconflow.cn/v1")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL","Qwen/Qwen3-VL-Embedding-8B")
SILICONFLOW_RERANK_URL = os.getenv("SILICONFLOW_RERANK_URL","https://api.siliconflow.cn/v1/rerank")
SILICONFLOW_RERANK_MODEL = os.getenv("SILICONFLOW_RERANK_MODEL","BAAI/bge-reranker-v2-m3")
CHROMADB_CLIENT_PATH = os.getenv("CHROMADB_CLIENT_PATH")
