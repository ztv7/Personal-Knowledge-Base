import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import get_llm

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是问题分解器。判断用户问题是否为复合问题（包含多个可独立检索的子问题）。

规则：
1. 若问题是单一简单问题，返回空列表 []
2. 若问题包含多个子问题（用"；"、"并"、"和"、"还有"等连接，或有依赖关系），拆分为有序子问题列表
3. 子问题按依赖关系排列：不依赖任何答案的放前面，后续子问题可引用前面的编号
4. 每个子问题应为独立的、可直接检索的完整问句

仅输出 JSON 格式: [{"step": 1, "question": "子问题1", "depends_on": []}, ...]"""),
    ("user", "用户问题：{question}"),
])


def decompose_question(question: str) -> List[dict]:
    """
    分解复合问题为有序子问题列表
    返回: [{"step": 1, "question": "...", "depends_on": []}, ...]
    空列表表示单一问题，无需分解
    """
    # 快速启发式检测：包含分号、或者句号分隔的多个问句
    has_multi_clue = any(
        s in question for s in ["；", "并", "还有", "以及", "另外"]
    ) or question.count("？") > 1 or question.count("?") > 1

    if not has_multi_clue:
        return []

    llm = get_llm(temperature=0)
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"question": question})
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        result = json.loads(raw)
        if isinstance(result, list) and len(result) > 0:
            return result
        return []
    except Exception:
        return []


EXTRACT_ENTITIES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """从前置答案中提取与当前子问题相关的关键实体和核心事实。
仅输出 1-2 句精简摘要，不要复述完整内容，只保留对检索有帮助的关键词。
例如：答案提到"A是X用于Y"，而当前子问题是"A的特点"，则摘要为"已知：A是X技术，用于Y领域。"。"""),
    ("user", "前置答案：{prev_answer}\n当前子问题：{next_question}\n提取相关关键信息："),
])


def extract_key_context(prev_answer: str, next_question: str) -> str:
    """从前置答案中提取与当前子问题相关的关键实体，用于精简检索 query"""
    if not prev_answer.strip():
        return next_question

    llm = get_llm(temperature=0)
    chain = EXTRACT_ENTITIES_PROMPT | llm | StrOutputParser()

    try:
        summary = chain.invoke({
            "prev_answer": prev_answer[:2000],  # 截断过长答案
            "next_question": next_question,
        })
        return f"{summary.strip()}\n{next_question}"
    except Exception:
        return f"{prev_answer[:500]}\n{next_question}"  # 降级：截断而非全量
