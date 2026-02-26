import threading
from collections import deque
from typing import Deque, Dict, List


class SessionStore:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._store: Dict[str, Deque[dict]] = {}

    def get(self, session_id: str) -> List[dict]:
        with self._lock:
            dq = self._store.get(session_id)
            if not dq:
                return []
            return list(dq)

    def append(self, session_id: str, role: str, content: str) -> None:
        item = {"role": role, "content": content}
        with self._lock:
            dq = self._store.get(session_id)
            if dq is None:
                dq = deque(maxlen=self.max_turns * 2)
                self._store[session_id] = dq
            dq.append(item)
