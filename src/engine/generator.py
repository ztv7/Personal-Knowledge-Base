from typing import Generator, List
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import get_llm

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个基于个人知识库的智能助手。严格依据以下参考文档回答问题：

规则：
1. 仅根据提供的文档片段回答，禁止编造任何信息
2. 回答时标注信息来源的文档名和页码，格式：(来源: xxx 第x页)
3. 如果文档片段不足以回答，请明确说明"根据已有文档无法回答此问题"
4. 如果涉及多个子问题，请逐一回答使其连贯自然
5. 结合对话历史上下文，使回答与之前的交流连贯"""),
    ("user", "对话历史：\n{history}\n\n参考文档片段：\n{context}\n\n用户问题：{question}"),
])


def generate_stream(
    question: str,
    context: str,
    chat_history: list[dict],
    temperature: float = None,
) -> Generator[str, None, None]:
    """流式生成回答"""
    history_text = ""
    for h in chat_history[-6:]:
        history_text += f"用户：{h['user']}\n助手：{h['assistant']}\n"

    if not history_text:
        history_text = "无历史对话"

    llm = get_llm(temperature=temperature, streaming=True)
    chain = QA_PROMPT | llm

    for chunk in chain.stream({
        "history": history_text,
        "context": context or "未找到相关文档片段。",
        "question": question,
    }):
        if chunk.content:
            yield chunk.content


def generate_sync(
    question: str,
    context: str,
    temperature: float = None,
) -> str:
    """非流式生成回答（用于子问题求解）"""
    llm = get_llm(temperature=temperature or 0.3, streaming=False)
    chain = QA_PROMPT | llm
    response = chain.invoke({
        "history": "无历史对话",
        "context": context,
        "question": question,
    })
    return response.content
