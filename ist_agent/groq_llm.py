import os
from typing import List, Tuple

from groq import Groq


_ESCALATION_MSG = (
    "I will forward your specific query to the IST Admissions Office. "
    "Could you please provide your phone number so we can call you back with an official answer?"
)


_SYSTEM = (
    "You are IST Admissions Voice Agent."
    "\nRules:" 
    "\n- Use ONLY the provided CONTEXT."
    "\n- If the answer is not explicitly in CONTEXT and it is not a simple yes/no from CONTEXT, output exactly: ESCALATE"
    "\n- Never repeat or paraphrase the user question. Start with the answer."
    "\n- Be concise: 1-2 sentences, max 4 only for complex fee structures."
    "\n- No apologies. No uncertainty phrases."
)


def _build_context(chunks: List[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        src = c.get("source", "")
        txt = c.get("text", "")
        parts.append(f"[SOURCE: {src}] {txt}")
    return "\n\n".join(parts)


def _is_refusal(text: str) -> bool:
    t = (text or "").lower()
    bad = [
        "i can't",
        "i cannot",
        "i’m unable",
        "i am unable",
        "as an ai",
        "i don't have access",
        "technical issue",
        "error",
    ]
    return any(b in t for b in bad)


def grounded_answer(query: str, context_chunks: List[dict], history: List[dict]) -> Tuple[str, bool]:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_CHAT_MODEL", "llama3-70b-8192")

    ctx = _build_context(context_chunks)
    messages = [{"role": "system", "content": _SYSTEM}]

    if history:
        messages.extend(history[-10:])

    user_msg = f"CONTEXT:\n{ctx}\n\nUSER_QUERY:\n{query}\n\nAnswer with either ESCALATE or the final answer.".strip()
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return _ESCALATION_MSG, True

    if text == "ESCALATE":
        return _ESCALATION_MSG, True

    if _is_refusal(text):
        return _ESCALATION_MSG, True

    if not ctx.strip():
        return _ESCALATION_MSG, True

    return text, False
