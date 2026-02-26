import os
import time
import uuid
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from ist_agent.rag import RagEngine
from ist_agent.session_store import SessionStore
from ist_agent.leads import LeadManager
from ist_agent.groq_llm import grounded_answer
from ist_agent.livekit_tokens import create_livekit_token

load_dotenv()

app = Flask(__name__, static_folder="web", static_url_path="")

rag = RagEngine(
    data_dir=os.path.join(os.path.dirname(__file__), "data"),
    persist_dir=os.path.join(os.path.dirname(__file__), "chroma_db"),
)
sessions = SessionStore(max_turns=12)
leads = LeadManager(
    lead_log_path=os.path.join(os.path.dirname(__file__), "logs", "lead_logs.txt"),
    session_dir=os.path.join(os.path.dirname(__file__), "logs", "sessions"),
)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/config")
def config():
    return jsonify({"livekit_url": os.getenv("LIVEKIT_URL", "").strip()})


@app.get("/livekit/token")
def livekit_token():
    identity = request.args.get("identity") or f"web-{uuid.uuid4().hex[:12]}"
    room = request.args.get("room") or os.getenv("LIVEKIT_ROOM", "ist-admissions")
    token = create_livekit_token(identity=identity, room=room)
    return jsonify({"identity": identity, "room": room, "token": token})


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True, silent=False)
    session_id = payload.get("session_id") or uuid.uuid4().hex
    user_text = (payload.get("text") or "").strip()

    if not user_text:
        return jsonify({"session_id": session_id, "answer": "Please ask a question about IST admissions.", "escalate": False})

    history = sessions.get(session_id)
    chunks = rag.hybrid_search(user_text)

    answer, should_escalate = grounded_answer(
        query=user_text,
        context_chunks=chunks,
        history=history,
    )

    sessions.append(session_id, role="user", content=user_text)
    sessions.append(session_id, role="assistant", content=answer)

    if should_escalate:
        leads.start_escalation(session_id=session_id, query=user_text)

    leads.save_session_record(
        session_id=session_id,
        record={
            "ts": time.time(),
            "query": user_text,
            "answer": answer,
            "escalate": should_escalate,
        },
    )

    return jsonify({"session_id": session_id, "answer": answer, "escalate": should_escalate})


@app.post("/api/lead")
def api_lead():
    payload = request.get_json(force=True, silent=False)
    session_id = payload.get("session_id")
    text = (payload.get("text") or "").strip()

    if not session_id:
        return jsonify({"ok": False, "error": "missing session_id"}), 400

    extracted = leads.extract_phone(text)
    if not extracted:
        return jsonify({"ok": True, "saved": False})

    saved = leads.finalize_lead(session_id=session_id, phone=extracted)
    return jsonify({"ok": True, "saved": saved, "phone": extracted})


@app.get("/<path:path>")
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
