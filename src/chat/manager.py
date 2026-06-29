from typing import List, Dict
from collections import defaultdict


class ChatSessionManager:
    """内存会话管理"""

    def __init__(self):
        self._sessions: Dict[str, List[dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if content.strip():
            self._sessions[session_id].append({
                "role": role,
                "content": content,
            })

    def get_history(self, session_id: str) -> List[dict]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
