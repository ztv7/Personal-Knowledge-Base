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
