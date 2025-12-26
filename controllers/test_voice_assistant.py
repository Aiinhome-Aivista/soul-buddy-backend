import json
import os
import json
import uuid
import pymysql
import re
import unicodedata
import asyncio
from datetime import datetime
from flask import request, jsonify, send_from_directory, current_app

import edge_tts

from controllers.chat import rag_chat_controller
from model.llm_client import call_llm
from database.config import BASE_URL, MYSQL_CONFIG


# ===================== CONFIG =====================
AUDIO_DIR = os.path.join(os.getcwd(), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


# ===================== DB =====================
def get_connection():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        cursorclass=pymysql.cursors.DictCursor
    )

# ===================== GREETING HELPERS =====================


def is_first_session(user_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) AS cnt
        FROM conversation_history
        WHERE user_id = %s
    """, (user_id,))

    count = cur.fetchone()["cnt"]
    conn.close()

    return count == 0


def fetch_user_context_for_greeting(user_id: str) -> dict:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            u.full_name,
            u.age,
            u.work,
            u.emotional_state,
            u.relationship,
            w.ai_profile
        FROM users u
        LEFT JOIN wellbeing_ai_results w
            ON u.user_id = w.user_id
        WHERE u.user_id = %s
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return {}

    context = {
        "full_name": row.get("full_name"),
        "age": row.get("age"),
        "work": row.get("work"),
        "emotional_state": row.get("emotional_state"),
        "relationship": row.get("relationship")
    }

    if row.get("ai_profile"):
        context["wellbeing_profile"] = json.loads(row["ai_profile"])

    return context


def build_first_time_greeting(user_id: str) -> str:
    context = fetch_user_context_for_greeting(user_id)

    if not context:
        return "Hi, I’m really glad you’re here. I’m here to support you."

    prompt = f"""
You are a calm, empathetic wellness voice assistant.

Using the user information below, generate a short greeting
for a FIRST-TIME conversation.

Context:
{json.dumps(context, indent=2)}

Rules:
- 2–3 short sentences
- Friendly, warm, human tone
- Spoken language
- No advice
- Do NOT mention AI or data
"""

    greeting = call_llm(prompt)
    return clean_text_for_voice(greeting)


def build_returning_user_greeting(user_id: str) -> str:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT model_response
        FROM conversation_history
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if row and row.get("model_response"):
        return "Welcome back. I’m here with you. Let’s continue."

    return "Welcome back. I’m here with you."

# ===================== TEXT CLEANER =====================
def clean_text_for_voice(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    # remove emojis
    text = re.sub(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "]+",
        "",
        text
    )

    # remove markdown & symbols
    text = re.sub(r"[*_`>#\-]", " ", text)

    # remove newlines
    text = text.replace("\n", " ")

    # collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ===================== SESSION =====================
def get_or_create_session_id(user_id: str) -> str:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT session_id
        FROM conversation_history
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()

    if row:
        session_id = row["session_id"]
    else:
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cur.execute("""
            INSERT INTO conversation_history
            (session_id, user_id, user_input, model_response)
            VALUES (%s, %s, %s, %s)
        """, (session_id, user_id, "[SESSION STARTED]", "[SESSION CREATED]"))
        conn.commit()

    conn.close()
    return session_id

def is_first_message_of_session(session_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) AS cnt
        FROM conversation_history
        WHERE session_id = %s
    """, (session_id,))

    count = cur.fetchone()["cnt"]
    conn.close()

    # Only the SESSION START row exists
    return count == 1

# ===================== VOICE (PLAIN TEXT – STABLE) =====================
def generate_voice(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Empty text passed to TTS")

    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)

    async def _generate():
        tts = edge_tts.Communicate(
            text,
            voice="en-US-JennyNeural",
            rate="-20%"
        )
        await tts.save(filepath)

    try:
        asyncio.run(_generate())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()

    if not os.path.exists(filepath) or os.path.getsize(filepath) < 2000:
        raise RuntimeError("Audio generation failed or silent")

    return filename


# ===================== MAIN HANDLER =====================
def handle_voice_ask():
    """
    User sends ONLY:
    {
        "user_id": "u1",
        "text": "im feeling depressed what can i do?"
    }
    """

    data = request.json or {}
    user_id = data.get("user_id")
    user_input = data.get("text")

    if not user_input:
        user_input = ""

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    session_id = get_or_create_session_id(user_id)
# ---------- GREETING (ONLY ON FIRST MESSAGE OF SESSION) ----------
    greeting_text = ""

    if is_first_message_of_session(session_id):
        if is_first_session(user_id):
            greeting_text = build_first_time_greeting(user_id)
        else:
            greeting_text = build_returning_user_greeting(user_id)
    # ---------- RAG FIRST ----------
    rag_answer = None
    try:
        with current_app.test_request_context(
            "/rag_chat",
            json={"session_id": session_id, "query": user_input}
        ):
            rag_resp = rag_chat_controller()
            rag_json = rag_resp.get_json()
            rag_answer = rag_json.get("answer")
    except Exception:
        rag_answer = None

    if rag_answer and rag_answer.strip():
        reply = clean_text_for_voice(rag_answer)
        final_reply = f"{greeting_text} {reply}".strip()
        source = "rag_database"
    else:
        prompt = f"""
You are a calm, supportive voice assistant.

User says: {user_input if user_input else "No message yet, user just joined."}

Respond gently and clearly in simple spoken language.
"""
        raw_reply = call_llm(prompt)
        reply = clean_text_for_voice(raw_reply)
        final_reply = f"{greeting_text} {reply}".strip()
        source = "mistral_fallback"

    # ---------- SAVE ----------
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversation_history
        (session_id, user_id, user_input, model_response)
        VALUES (%s, %s, %s, %s)
    """, (session_id, user_id, user_input, final_reply))
    conn.commit()
    conn.close()

    # ---------- VOICE ----------
    audio_file = generate_voice(final_reply)

    return jsonify({
        "session_id": session_id,
        "text_response": final_reply,
        "source": source,
        "audio_url": f"{BASE_URL}/audio/{audio_file}"
    })


# ===================== AUDIO SERVE =====================
def serve_audio_file(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/wav")
