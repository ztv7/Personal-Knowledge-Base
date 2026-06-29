import os
import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from src.api.models import (
    ChatCompletionRequest, ChatCompletionChunk, ChoiceDelta, DeltaMessage,
    FileInfo, FileListResponse, FileDeleteResponse,
    RuntimeConfig, ConfigResponse, HistoryResponse, HistoryMessage,
)
from src.api.errors import EmptyQueryError
from src.store.vector_store import VectorStore
from src.loader.dispatcher import load_and_chunk
from src.engine.rewriter import rewrite_query
from src.engine.decomposer import decompose_question
from src.engine.retriever import retrieve, format_context
from src.engine.generator import generate_stream, generate_sync
from src.chat.manager import ChatSessionManager
from src.chat.export import export_markdown
from src.config import settings, update_runtime_config

router = APIRouter(prefix="/v1")

# 全局状态
_vector_store: VectorStore = None
_chat_manager: ChatSessionManager = ChatSessionManager()


def get_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# ---------- Chat Completions (OpenAI format, SSE streaming) ----------

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    store = get_store()
    session_id = "default"

    # 提取最后一条 user 消息
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise EmptyQueryError("缺少用户消息")
    raw_question = user_messages[-1].content.strip()
    if not raw_question:
        raise EmptyQueryError("用户消息为空")

    # 转换 OpenAI messages 格式为内部 chat_history
    chat_history = _extract_chat_history(req.messages)

    temperature = req.temperature if req.temperature is not None else settings.temperature

    # 1. 跨轮查询改写
    rewrite_result = rewrite_query(raw_question, chat_history)
    search_query = rewrite_result["new_query"]

    # 2. 单轮内问题分解
    sub_questions = decompose_question(search_query)

    if not sub_questions:
        # 简单问题：直接检索 → 生成
        return StreamingResponse(
            _simple_qa_stream(
                store, raw_question, search_query, chat_history, temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # 复合问题：分解 → 逐个子问题求解 → 最终合成
    return StreamingResponse(
        _multi_hop_qa_stream(
            store, raw_question, sub_questions, chat_history, temperature
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _extract_chat_history(messages: list) -> list[dict]:
    """从 OpenAI messages 格式提取 user-assistant 对"""
    result = []
    temp_user = None
    for m in messages:
        if m.role == "user":
            temp_user = m.content
        elif m.role == "assistant" and temp_user is not None:
            result.append({"user": temp_user, "assistant": m.content})
            temp_user = None
    return result


async def _simple_qa_stream(
    store, raw_question, search_query, chat_history, temperature
):
    """简单问题流式回答生成器"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        # 匹配文档 source
        doc_names = store.doc_names
        where = None
        for name in doc_names or []:
            if name.lower() in search_query.lower():
                where = {"source": name}
                break

        # 检索
        retrieved = retrieve(store, search_query, where=where)
        context = format_context(retrieved)

        # 保存用户消息
        _chat_manager.add_message("default", "user", raw_question)

        # 流式生成
        full_response = ""
        for delta_text in generate_stream(
            raw_question, context, chat_history, temperature
        ):
            full_response += delta_text
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created,
                model=settings.deepseek_model,
                choices=[
                    ChoiceDelta(
                        delta=DeltaMessage(content=delta_text)
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # 保存助手消息
        _chat_manager.add_message("default", "assistant", full_response)

        # 发送结束信号
        finish_chunk = ChatCompletionChunk(
            id=chat_id,
            created=created,
            model=settings.deepseek_model,
            choices=[
                ChoiceDelta(
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except TimeoutError as e:
        error_chunk = ChatCompletionChunk(
            id=chat_id,
            created=created,
            model=settings.deepseek_model,
            choices=[
                ChoiceDelta(
                    delta=DeltaMessage(
                        content=f"\n\n[错误] {str(e)}"
                    ),
                    finish_reason="error",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


async def _multi_hop_qa_stream(
    store, raw_question, sub_questions, chat_history, temperature
):
    """多跳复合问题流式回答生成器"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        # 按依赖顺序求解每个子问题
        sub_answers = {}
        for sq in sub_questions:
            step = sq["step"]
            sub_q = sq["question"]
            depends = sq.get("depends_on", [])

            # 融合前置答案作为检索上下文
            enriched_query = sub_q
            for dep_step in depends:
                if dep_step in sub_answers:
                    enriched_query = f"{sub_answers[dep_step]}\n{sub_q}"

            # 检索
            retrieved = retrieve(store, enriched_query)
            context = format_context(retrieved)

            # 非流式生成子答案
            sub_ans = generate_sync(sub_q, context, temperature)
            sub_answers[step] = sub_ans

        # 汇总合成最终答案
        all_qa_pairs = "\n".join([
            f"子问题{sq['step']}：{sq['question']}\n子答案：{sub_answers.get(sq['step'], '')}"
            for sq in sub_questions
        ])

        _chat_manager.add_message("default", "user", raw_question)

        # 流式输出合成结果
        synthetic_prompt = (
            f"以下是对复合问题的子问题逐一解答，请汇总为一段连贯完整的回答：\n\n"
            f"{all_qa_pairs}\n\n原始问题：{raw_question}"
        )

        full_response = ""
        for delta_text in generate_stream(
            synthetic_prompt,
            "参考上述子答案合成",
            chat_history,
            temperature,
        ):
            full_response += delta_text
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created,
                model=settings.deepseek_model,
                choices=[
                    ChoiceDelta(
                        delta=DeltaMessage(content=delta_text)
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        _chat_manager.add_message("default", "assistant", full_response)

        finish_chunk = ChatCompletionChunk(
            id=chat_id,
            created=created,
            model=settings.deepseek_model,
            choices=[
                ChoiceDelta(
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except TimeoutError as e:
        error_chunk = ChatCompletionChunk(
            id=chat_id,
            created=created,
            model=settings.deepseek_model,
            choices=[
                ChoiceDelta(
                    delta=DeltaMessage(content=f"\n\n[错误] {str(e)}"),
                    finish_reason="error",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


# ---------- File Management ----------

@router.post("/files", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    store = get_store()
    content = await file.read()
    filename = file.filename or "unknown"

    from src.loader.base import FileValidationError
    try:
        ids, documents, metadatas = load_and_chunk(
            file_bytes=content, filename=filename
        )
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ids:
        raise HTTPException(status_code=400, detail="文件内容为空，无法入库")

    store.add(ids=ids, documents=documents, metadatas=metadatas)

    return FileInfo(
        id=ids[0].split("_p")[0],
        filename=filename,
        bytes=len(content),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.get("/files", response_model=FileListResponse)
async def list_files():
    store = get_store()
    files = [
        FileInfo(id=name, filename=name)
        for name in store.doc_names
    ]
    return FileListResponse(data=files)


@router.delete("/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(file_id: str):
    store = get_store()
    store.delete_by_source(file_id)
    return FileDeleteResponse(id=file_id)


# ---------- Config ----------

@router.get("/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(config=RuntimeConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        similarity_threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        temperature=settings.temperature,
        relevance_threshold=settings.relevance_threshold,
    ))


@router.put("/config", response_model=ConfigResponse)
async def update_config(config: RuntimeConfig):
    update_runtime_config(**config.model_dump())
    return ConfigResponse(config=RuntimeConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        similarity_threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        temperature=settings.temperature,
        relevance_threshold=settings.relevance_threshold,
    ))


# ---------- Chat History ----------

@router.get("/chat/history", response_model=HistoryResponse)
async def get_history(session_id: str = Query(default="default")):
    messages = _chat_manager.get_history(session_id)
    return HistoryResponse(messages=[
        HistoryMessage(role=m["role"], content=m["content"])
        for m in messages
    ])


@router.get("/chat/history/export")
async def export_history(session_id: str = Query(default="default")):
    messages = _chat_manager.get_history(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="无对话历史")
    md_content = export_markdown(
        messages, title=f"Knowledge Base Chat - {session_id}"
    )
    return StreamingResponse(
        iter([md_content]),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=chat_export_{session_id}.md"
        },
    )
