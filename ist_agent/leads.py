import json
import os
import re
import threading
import time
from typing import Dict, Optional


_ESCALATION_MSG = (
    "I will forward your specific query to the IST Admissions Office. "
    "Could you please provide your phone number so we can call you back with an official answer?"
)


class LeadManager:
    def __init__(self, lead_log_path: str, session_dir: str):
        self.lead_log_path = lead_log_path
        self.session_dir = session_dir

        os.makedirs(os.path.dirname(self.lead_log_path), exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._pending: Dict[str, str] = {}

    @property
    def escalation_message(self) -> str:
        return _ESCALATION_MSG

    def start_escalation(self, session_id: str, query: str) -> None:
        with self._lock:
            self._pending[session_id] = query

    def extract_phone(self, text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"\b(03\d{2})[- ]?(\d{7})\b", text)
        if not m:
            m = re.search(r"\b(\+?92)\s?3\d{2}[- ]?(\d{7})\b", text)
            if not m:
                return None
            cc = m.group(1)
            tail = m.group(2)
            if cc.startswith("+"):
                return f"+92{tail}"
            return f"92{tail}"
        return f"{m.group(1)}-{m.group(2)}"

    def finalize_lead(self, session_id: str, phone: str) -> bool:
        with self._lock:
            query = self._pending.pop(session_id, None)
        if not query:
            return False

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts} | {phone} | {query} | {session_id}\n"
        with self._lock:
            with open(self.lead_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        return True

    def save_session_record(self, session_id: str, record: dict) -> None:
        path = os.path.join(self.session_dir, f"{session_id}.json")
        with self._lock:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = []
            else:
                data = []
            data.append(record)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
