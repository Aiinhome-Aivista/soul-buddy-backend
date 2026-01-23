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
import random
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

# ===================== PEP_TALK_MESSAGES =====================
PEP_TALK_MESSAGES = [
    # Gentle check-ins
    "Hey {name}, I’m right here with you.",
    "I’m still here, {name}. Take your time.",
    "Just checking in, {name}.",
    "I’m here whenever you’re ready, {name}.",
    "No rush at all, {name}. I’m here.",

    # Reassuring presence
    "You’re not alone right now, {name}.",
    "I’m staying with you, {name}.",
    "It’s okay to pause, {name}.",
    "I’m right beside you, {name}.",
    "You’re doing just fine, {name}.",

    # Calm grounding
    "Take a breath, {name}. I’m here.",
    "This moment is okay, {name}.",
    "You’re safe here, {name}.",
    "We can go slow, {name}.",
    "Nothing to fix right now, {name}.",

    # Warm encouragement (no pressure)
    "I’m listening whenever you want to continue, {name}.",
    "I’m here for you, {name}.",
    "It’s alright to take a moment, {name}.",
    "You can take your time here, {name}.",
    "I’ve got you, {name}.",

    # Soft emotional support
    "You don’t have to say anything yet, {name}.",
    "This space is just for you, {name}.",
    "I’m holding space for you, {name}.",
    "You’re allowed to pause, {name}.",
    "I’m still with you, {name}.",

    # Simple human warmth
    "Hey {name}, I’m here.",
    "I’m right here, {name}.",
    "Still here with you, {name}.",
    "I haven’t gone anywhere, {name}.",
    "Just here with you, {name}."
]


# ===================== SMALL WORD HELPERS =====================
def fetch_last_topic(user_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT user_input, model_response
        FROM conversation_history
        WHERE user_id = %s
          AND user_input NOT IN ('[LOGIN]', '[SESSION STARTED]')
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()
    return row

def generate_supportive_ack(user_id: str) -> str:
    last = fetch_last_topic(user_id)

    context = {}
    if last:
        context = {
            "last_user_message": last["user_input"],
            "last_assistant_response": last["model_response"]
        }

    prompt = f"""
You are a calm, empathetic wellness voice assistant.

User gave a small acknowledgement (like hmm / okay).

Conversation context:
{json.dumps(context, indent=2)}

Rules:
- Respond with EXACTLY 1 or 2 short sentences
- Supportive, warm tone
- No questions
- No advice
- No repetition
- Spoken, human language
- Output ONLY the response
"""

    reply = call_llm(prompt)
    return clean_text_for_voice(reply)


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

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    #  REMOVE EMOJIS
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"          # misc symbols
        "\u2700-\u27BF"          # dingbats
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # Remove markdown / symbols
    text = re.sub(r"[*_`>#\-]", " ", text)

    # Remove newlines
    text = text.replace("\n", " ")

    # Collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

SMALL_ACK_WORDS = {
    "hmm", "hm", "hmmm",
    "ok", "okay", "okk",
    "acha", "achaa", "accha",
    "haan", "haanji", "hmm okay",
    "yes","yeah"
}

def is_small_acknowledgement(text: str) -> bool:
    if not text:
        return False

    normalized = text.lower().strip()

    # exact match words
    if normalized in SMALL_ACK_WORDS:
        return True

    # short filler phrases only
    filler_patterns = [
        r"^(hmm+)$",
        r"^(ok+|okay+)$",
        r"^(acha+|accha+)$",
        r"^(haan+|haanji)$",
        r"^(yes|yeah)$"
    ]

    return any(re.match(p, normalized) for p in filler_patterns)


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

    # Check if row exists AND if session_id is actually a valid string
    if row and row["session_id"]:
        session_id = row["session_id"]
    else:
        # If no history OR the last history had a NULL session_id, create a new one
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

# ===================== GREETING HELPERS =====================

def is_first_session(user_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) AS cnt
        FROM conversation_history
        WHERE user_id = %s
          AND user_input NOT IN ('[SESSION STARTED]', '[LOGIN]')
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
        try:
            context["wellbeing_profile"] = json.loads(row["ai_profile"])
        except Exception:
            pass

    return context


def build_first_time_greeting(user_id: str) -> str:
    context = fetch_user_context_for_greeting(user_id)

    if not context:
        return "Hi, I’m really glad you’re here. I’m here to support you."

    prompt = f"""
You are a calm, empathetic wellness voice assistant.

Generate EXACTLY ONE short spoken greeting sentence
for a FIRST-TIME conversation.

Context:
{json.dumps(context, indent=2)}

Rules:
- ONE response only (no alternatives, no options)
- 1–2 short sentences maximum
- Warm, human, spoken tone
- No advice
- No explanations
- No quotation marks
- Do NOT mention AI, data, or preferences
- Output ONLY the greeting text
"""

    greeting = call_llm(prompt)
    return clean_text_for_voice(greeting)


def build_returning_user_greeting(user_id: str) -> str:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_input, model_response
        FROM conversation_history
        WHERE user_id = %s
          AND user_input NOT IN ('[SESSION STARTED]', '[LOGIN]', '')
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return "Welcome back. I’m really glad you’re here."

    context = {
        "last_user_message": row.get("user_input"),
        "last_assistant_response": row.get("model_response")
    }

    prompt = f"""
You are a calm wellness voice assistant.

Create a welcome-back greeting tied to the user's last topic.

Context:
{json.dumps(context, indent=2)}

Rules:
- ONE response only (no alternatives, no optional text)
- 1–2 short sentences maximum
- Mention the topic briefly
- Warm, human, spoken tone
- No questions
- No advice
- No quotation marks
- No parentheses
- No explanations
- Output ONLY the greeting text
"""

    greeting = call_llm(prompt)
    return clean_text_for_voice(greeting)


def handle_voice_ask():
    data = request.json or {}
    user_id = data.get("user_id")
    user_input = data.get("text", "")
    user_input = user_input.strip() if user_input else ""
    
    # 1. Identify if this is a Login/Start event (Empty text)
    is_login_only = user_input == ""

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    session_id = get_or_create_session_id(user_id)
    if user_input == "__NO_RESPONSE__":
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT full_name FROM users WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        conn.close()

        name = row["full_name"].split()[0] if row and row.get("full_name") else "there"

        msg = random.choice(PEP_TALK_MESSAGES).format(name=name)
        audio = generate_voice(msg)

        return jsonify({
            "session_id": session_id,
            "text_response": msg,
            "audio_url": f"{BASE_URL}/audio/{audio}"
        })

    # ===================== SUBSCRIPTION CHECKS =====================
    # (We keep these to ensure they have valid access before greeting)
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

    if datetime.now() > subscription["end_date"]:
        msg = "Your subscription has expired."
        audio = generate_voice(msg)
        return jsonify({
            "session_id": session_id,
            "text_response": msg,
            "audio_url": f"{BASE_URL}/audio/{audio}"
        })

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

    # ===================== CORE LOGIC START =====================
    
    final_reply = ""

    # 2. LOGIC SPLIT: Is this a Greeting or a Chat?
    if is_login_only:
        # === GREETING LOGIC ===
        if is_first_session(user_id):
            final_reply = build_first_time_greeting(user_id)
        else:
            final_reply = build_returning_user_greeting(user_id)
            
    else:
        # === SMALL ACK HANDLING ===
        if is_small_acknowledgement(user_input):
            final_reply = generate_supportive_ack(user_id)

        else:
            # === RAG CHAT LOGIC ===
            try:
                with current_app.test_request_context(
                    "/rag_chat",
                    json={
                        "session_id": session_id,
                        "user_id": user_id,
                        "query": user_input,
                        "is_voice": True
                    }
                ):
                    rag_resp = rag_chat_controller()
                    rag_json = extract_json_from_response(rag_resp)
                    answer = rag_json.get("answer", "")
                    final_reply = clean_text_for_voice(answer)
            except Exception as e:
                print(f"RAG error: {e}")
                final_reply = ""

            if not final_reply:
                fallback = call_llm(f"""
                You are a calm, supportive wellness voice assistant.
                User says: {user_input}
                Respond in 1–2 warm spoken sentences.
                """)
                final_reply = clean_text_for_voice(fallback)


    # ===================== SAVE HISTORY =====================
    # We save history even for greetings so the next turn has context
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # If input was empty (login), we record it as [LOGIN] in DB for clarity
        db_user_input = user_input if user_input else "[LOGIN]"
        
        cur.execute("""
            INSERT INTO conversation_history 
            (session_id, user_id, user_input, model_response)
            VALUES (%s, %s, %s, %s)
        """, (session_id, user_id, db_user_input, final_reply))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}")

    # ===================== GENERATE RESPONSE =====================
    audio = generate_voice(final_reply)
    
    return jsonify({
        "session_id": session_id,
        "text_response": final_reply,
        "audio_url": f"{BASE_URL}/audio/{audio}"
    })

# ===================== AUDIO SERVE =====================
def serve_audio_file(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/wav")
