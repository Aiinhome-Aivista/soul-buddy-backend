import json
import os
import uuid
import pymysql
import re
import unicodedata
import asyncio
from datetime import datetime, timedelta
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


# ===================== PLAN HELPERS =====================


def get_plan_details(plan_name):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT plan_code,
               plan_name,
               minutes_per_day,
               validity_days,
               is_trial,
               plan_level
        FROM subscription_plans
        WHERE plan_name = %s
          AND is_active = 1
        LIMIT 1
    """, (plan_name,))

    row = cur.fetchone()
    conn.close()
    return row

# ===================== SUBSCRIPTION HELPERS =====================
# def get_active_subscription(user_id):
#     conn = get_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         SELECT plan_name, start_date, end_date
#         FROM subscriptions
#         WHERE user_id = %s
#           AND status = 'active'
#           AND NOW() BETWEEN start_date AND end_date
#         LIMIT 1
#     """, (user_id,))

#     row = cur.fetchone()
#     conn.close()
#     return row

def get_active_subscription(user_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT plan_name, start_date, end_date
        FROM subscriptions
        WHERE user_id = %s
          AND status = 'active'
        ORDER BY start_date DESC
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()
    conn.close()
    return row

# ===================== VOICE WINDOW =====================
def get_today_voice_window(user_id, daily_limit_seconds):
    """
    Voice window starts from FIRST conversation today.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT created_at
        FROM conversation_history
        WHERE user_id = %s
          AND DATE(created_at) = CURDATE()
          AND user_input NOT IN ('[SESSION STARTED]')
        ORDER BY created_at ASC
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    start_time = row["created_at"]
    end_time = start_time + timedelta(seconds=daily_limit_seconds)
    remaining = int((end_time - datetime.now()).total_seconds())

    return {
        "start_time": start_time,
        "end_time": end_time,
        "remaining_seconds": max(0, remaining)
    }


# ===================== UTIL =====================
def extract_json_from_response(resp):
    if isinstance(resp, tuple):
        resp = resp[0]
    return resp.get_json() if resp else {}


def clean_text_for_voice(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[*_`>#\-]", " ", text)
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


# ===================== VOICE =====================
def generate_voice(text: str) -> str:
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)

    async def _gen():
        tts = edge_tts.Communicate(
            text,
            voice="en-US-JennyNeural",
            rate="-20%"
        )
        await tts.save(filepath)

    try:
        asyncio.run(_gen())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_gen())
        loop.close()

    return filename


# ===================== MAIN HANDLER =====================
def handle_voice_ask():
    data = request.json or {}
    user_id = data.get("user_id")
    user_input = data.get("text", "").strip()

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    session_id = get_or_create_session_id(user_id)

    # ---- Subscription ----
    subscription = get_active_subscription(user_id)
    if not subscription:
        msg = "You don’t have an active voice subscription."
        audio = generate_voice(msg)
        return jsonify({
            "session_id": session_id,
            "text_response": msg,
            "audio_url": f"{BASE_URL}/audio/{audio}"
        })

    plan = get_plan_details(subscription["plan_name"])
    if not plan:
        msg = "Your subscription plan is invalid."
        audio = generate_voice(msg)
        return jsonify({
            "session_id": session_id,
            "text_response": msg,
            "audio_url": f"{BASE_URL}/audio/{audio}"
        })

    # ---- Validity ----
    if datetime.now() > subscription["end_date"]:
        msg = "Your subscription has expired."
        audio = generate_voice(msg)
        return jsonify({
            "session_id": session_id,
            "text_response": msg,
            "audio_url": f"{BASE_URL}/audio/{audio}"
        })

    # ---- Daily Voice Limit ----
    if plan["minutes_per_day"] is not None:
        daily_limit_seconds = plan["minutes_per_day"] * 60
        window = get_today_voice_window(user_id, daily_limit_seconds)

        if window and window["remaining_seconds"] <= 0:
            msg = "Your daily voice limit is over. Please come back tomorrow."
            audio = generate_voice(msg)
            return jsonify({
                "session_id": session_id,
                "text_response": msg,
                "audio_url": f"{BASE_URL}/audio/{audio}"
            })

    # ---- RAG ----
    try:
        with current_app.test_request_context(
            "/rag_chat",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "query": user_input
            }
        ):
            rag_resp = rag_chat_controller()
            rag_json = extract_json_from_response(rag_resp)
            answer = rag_json.get("answer", "")
    except Exception as e:
        print(f"RAG error: {e}")
        answer = ""

    final_reply = clean_text_for_voice(answer)

    # ---- Fallback ----
    if not final_reply:
        fallback = call_llm(f"""
You are a calm, supportive wellness voice assistant.

User says: {user_input}

Respond in 1–2 warm spoken sentences.
""")
        final_reply = clean_text_for_voice(fallback)

    audio = generate_voice(final_reply)

    return jsonify({
        "session_id": session_id,
        "text_response": final_reply,
        "audio_url": f"{BASE_URL}/audio/{audio}"
    })


# ===================== AUDIO SERVE =====================
def serve_audio_file(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/wav")
