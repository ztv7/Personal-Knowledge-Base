from typing import List
from datetime import datetime


def export_markdown(messages: List[dict], title: str = "Chat Export") -> str:
    """将对话历史导出为 Markdown 格式"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"",
        f"> 导出时间：{now}",
        f"> 共 {len(messages)} 条消息",
        f"",
        "---",
        "",
    ]

    for i, msg in enumerate(messages, 1):
        role_label = "**用户**" if msg["role"] == "user" else "**助手**"
        lines.append(f"### #{i} {role_label}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
