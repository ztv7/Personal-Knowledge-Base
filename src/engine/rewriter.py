import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import get_llm

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是检索问句改写器。结合历史对话判断当前问题是否缺少前置信息：
1. 若依赖上文，把隐含信息补全，生成一条完整独立、可直接用于向量检索的问句；
2. 若完全独立，new_query 直接等于原问题。

仅输出 JSON，格式: {{"is_depend_history": true/false, "new_query": "改写后的检索问句"}}"""),
    ("user", "历史对话：\n{history}\n\n当前用户新问题：{question}"),
])


def rewrite_query(question: str, chat_history: list[dict]) -> dict:
    """
    chat_history: [{"user":"xxx","assistant":"xxx"}, ...]
    返回 {"is_depend_history": bool, "new_query": str}
    """
    history_text = ""
    for h in chat_history[-6:]:  # 仅保留最近6轮
        history_text += f"用户：{h['user']}\n助手：{h['assistant']}\n"

    llm = get_llm(temperature=0)
    chain = REWRITE_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "history": history_text or "无历史对话",
            "question": question,
        })
        # 提取 JSON
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        result = json.loads(raw)
        return {
            "is_depend_history": result.get("is_depend_history", False),
            "new_query": result.get("new_query", question),
        }
    except Exception:
        return {"is_depend_history": False, "new_query": question}
