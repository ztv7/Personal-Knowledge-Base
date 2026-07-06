# Personal Knowledge Base RAG

基于 RAG 技术的本地私有知识库智能问答系统。上传文档即可进行多轮对话问答，支持跨轮查询改写与单轮复合问题分解。

## 功能特性

- **多格式文档**：支持 PDF / TXT / Markdown 上传入库
- **混合切分**：递归切分 + Embedding 语义边界检测，语义断点处精准切分
- **两阶段检索**：ChromaDB 向量召回 + BGE-Reranker 重排序 + 相关性阈值过滤
- **双维度链式依赖处理**：
  - 跨轮依赖：多轮对话查询改写，自动补全上下文省略信息
  - 单轮多跳：复合问题自动分解为有序子问题链，按依赖顺序求解后汇总
- **SSE 流式输出**：OpenAI Chat Completions 兼容格式
- **可配置侧边栏**：chunk_size、top-k、temperature、threshold 实时调节
- **对话导出**：Markdown 格式导出对话历史

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | FastAPI + Uvicorn |
| 向量数据库 | ChromaDB |
| LLM | DeepSeek (OpenAI SDK) |
| Embedding | SiliconFlow Qwen3-VL-Embedding-8B |
| Rerank | SiliconFlow BGE-Reranker-v2-m3 |
| 文本切分 | LangChain + 自定义语义切分 |
| 前端 | 原生 HTML/CSS/JS (SPA) |

## 项目结构

```
├── main.py                     # FastAPI 入口
├── requirements.txt
├── src/
│   ├── config.py               # Pydantic Settings 配置
│   ├── api/                    # HTTP 路由、模型、异常处理
│   ├── core/                   # LLM / Embedding / Rerank / Splitter
│   ├── loader/                 # PDF / TXT / MD 文档加载
│   ├── engine/                 # RAG 引擎（查询改写、问题分解、检索、生成）
│   ├── store/                  # ChromaDB 向量存储
│   ├── chat/                   # 会话管理、Markdown 导出
│   └── static/                 # 前端 SPA
└── chroma_db/                  # 向量数据库持久化目录
```

## 快速开始

### 1. 环境配置

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
SILICONFLOW_RERANK_URL=https://api.siliconflow.cn/v1/rerank
SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3

CHROMADB_CLIENT_PATH=chroma_db
```

### 3. 启动

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问 `http://localhost:8000`

## API 端点

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | 对话（SSE 流式），OpenAI 兼容格式 |
| POST | `/v1/files` | 上传文档 |
| GET | `/v1/files` | 已入库文档列表 |
| DELETE | `/v1/files/{file_id}` | 删除文档 |
| GET | `/v1/config` | 获取当前配置 |
| PUT | `/v1/config` | 更新配置 |
| GET | `/v1/chat/history` | 获取对话历史 |
| GET | `/v1/chat/history/export` | 导出对话为 Markdown 文件 |

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| chunk_size | 500 | 文本切分大小 |
| chunk_overlap | 50 | 切分重叠长度 |
| similarity_threshold | 0.65 | 语义合并阈值 |
| top_k | 8 | 向量检索召回数 |
| rerank_top_n | 3 | Rerank 保留数 |
| temperature | 0.7 | LLM 生成温度 |
| relevance_threshold | 0.3 | 相关性过滤阈值 |
