from openai import OpenAI
from chromadb import QueryResult
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SILICONFLOW_API_KEY, SILICONFLOW_RERANK_URL,SILICONFLOW_RERANK_MODEL
from typing import Dict, Generator, List
import json
import requests
from . import store
# 全局大模型客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# Rerank 重排序函数
def rerank(query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, str, float]]:
    """
    硅基流动 BGE Reranker 重排序
    返回：[(原始下标, 文本, 相关性分数)]，接口默认按分数降序返回
    """
    header = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SILICONFLOW_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    resp = requests.post(
        SILICONFLOW_RERANK_URL,
        headers=header,
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data["results"]
    # 直接遍历接口返回结果，接口天然按 score 从高到低，不额外排序
    output = []
    for item in results:
        ori_idx = item["index"]  # 这里就是 raw_texts / meta_list 的原始下标
        score = item["relevance_score"]
        text = documents[ori_idx]
        output.append((ori_idx, text, score))
    return output

# Function Calling 工具定义
EXTRACT_SOURCE_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_document_filter",
        "description": "根据用户问题，匹配对应的文档完整文件名，输出检索参数",
        "parameters": {
            "type": "object",
            "required": ["target_source", "search_query"],
            "properties": {
                "target_source": {
                    "type": "string",
                    "description": "匹配到的完整文档文件名，无匹配则返回空字符串''，必须严格使用给定文档列表里的全名"
                },
                "search_query": {
                    "type": "string",
                    "description": "向量库检索精简关键词"
                }
            }
        }
    }
}

REWRITE_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "rewrite_search_query",
        "description": "结合历史对话判断当前问题是否依赖上文，生成无歧义、独立完整的检索query",
        "parameters": {
            "type": "object",
            "required": ["is_depend_history", "new_query"],
            "properties": {
                "is_depend_history": {
                    "type": "boolean",
                    "description": "当前问题是否依赖上一轮对话上下文"
                },
                "new_query": {
                    "type": "string",
                    "description": "融合历史上下文后的独立检索问句，无依赖则等于原问题"
                }
            }
        }
    }
}

def rewrite_query_with_history(question: str, chat_history: list[dict]) -> dict:
    """
    chat_history: [{"user":"xxx","assistant":"xxx"},...]
    返回 {"is_depend_history":bool, "new_query":"重写后的检索词"}
    """
    history_text = ""
    for item in chat_history:
        history_text += f"用户：{item['user']}\n助手：{item['assistant']}\n"

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "你是检索问句改写器。结合历史对话判断当前问题是否缺少前置信息。\n"
                               "1. 若依赖上文，把隐含信息补全，生成一条完整独立、可直接用于向量检索的问句；\n"
                               "2. 若完全独立，new_query直接等于用户原问题；\n"
                               "仅调用工具输出结构化结果，不要回答问题。"
                },
                {"role": "user", "content": f"历史对话：\n{history_text}\n当前用户新问题：{question}"}
            ],
            tools=[REWRITE_QUERY_TOOL]
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"is_depend_history": False, "new_query": question}
        tool_args = json.loads(msg.tool_calls[0].function.arguments)
        return tool_args
    except Exception as e:
        print(f"query改写异常:{e}，使用原问题检索")
        return {"is_depend_history": False, "new_query": question}

def parse_user_question(question: str, all_doc_sources: List[str]) -> Dict[str, str]:
    """
    all_doc_sources：所有已入库完整文档名列表
    返回 {target_source: 完整文件名 / "", search_query: 检索词}
    """
    doc_list_text = "\n".join([f"- {name}" for name in all_doc_sources])
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": f"""你是文档检索匹配器。现有已入库文档列表：
{doc_list_text}
规则：
1. 根据用户问题语义，从上面列表中匹配对应的完整文档名；
2. 用户只说简称、主题、章节时，自动对应完整文件名；
3. 问题不指向任何文档，则target_source填空字符串；
4. search_query去除与文档相关的内容，保留其他所有内容；
5. 必须调用工具函数输出结构化数据，禁止直接回答问题。"""
                },
                {"role": "user", "content": f"用户问题：{question}"}
            ],
            tools=[EXTRACT_SOURCE_TOOL],
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"target_source": "", "search_query": question}
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return args
    except Exception as e:
        print(f"文档匹配解析异常：{e}，检索全部文档")
        return {"target_source": "", "search_query": question}

def generate_answer(
    db: store.Database,
    raw_question: str,
    chat_history: list[dict],
    recall_top_k: int = 8,
    rerank_top_n: int = 3
) -> Generator[str, None, None]:
    # 1. 结合历史重写检索问句
    rewrite_res = rewrite_query_with_history(raw_question, chat_history)
    search_query = rewrite_res["new_query"]
    print(f"\n【依赖历史：{rewrite_res['is_depend_history']}】重写检索问句：{search_query}")

    # 2. 获取全部文档名，匹配目标文档source
    all_doc_names = list(db.doc_name_map.keys())
    parse_info = parse_user_question(search_query, all_doc_names)
    target_source = parse_info["target_source"]
    filter_where = {"source": target_source} if target_source else {}
    print(f"【匹配文档】{target_source if target_source else '全部文档'}，召回TopK={recall_top_k}")

    # 3. 使用重写后的问句执行向量检索（核心改动：检索内移）
    query_result = db.query_question(
        query_text=search_query,
        where=filter_where,
        n_result=recall_top_k
    )

    docs_list = query_result["documents"][0]
    meta_list = query_result["metadatas"][0]
    raw_texts = docs_list

    # 4. 重写后的问句做Rerank打分
    rerank_items = rerank(query=search_query, documents=raw_texts, top_n=rerank_top_n)

    # 5. 组装上下文
    context_blocks = []
    for ori_idx, text, score in rerank_items:
        if 0 <= ori_idx < len(meta_list):
            meta = meta_list[ori_idx]
            source = meta["source"]
            page = meta["page"]
            context_blocks.append(f"【文档：{source} 第{page}页 | 相关度：{score:.4f}】\n{text}")
    context = "\n\n".join(context_blocks)

    # 6. 拼接历史对话送入LLM
    history_str = ""
    for h in chat_history:
        history_str += f"历史用户：{h['user']}\n历史回答：{h['assistant']}\n\n"

    messages = [
        {
            "role": "system",
            "content": "严格依据参考文档回答，标注文档名与页码，禁止编造，结合对话历史连贯作答。"
        },
        {
            "role": "user",
            "content": f"历史对话记录：\n{history_str}\n参考文档片段：\n{context}\n用户当前问题：{raw_question}"
        }
    ]

    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.7,
        stream=True
    )
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta
            yield delta
            print(delta, end="", flush=True)
    print("\n")