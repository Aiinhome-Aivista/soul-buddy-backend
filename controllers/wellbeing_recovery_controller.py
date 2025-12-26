import json
from flask import request, jsonify
from openai import OpenAI
from database.config import MISTRAL_API_KEY, MYSQL_CONFIG
from datetime import datetime

from database.db_handler import fetch_ai_profile, save_recovery_plan


# ==============================
# MISTRAL CLIENT SETUP
# ==============================

client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

MODEL_NAME = "mistral-small"


# ==============================
# PROMPT BUILDER
# ==============================

current_date = datetime.utcnow().strftime("%B %Y")


def build_recovery_prompt(ai_profile, current_date):
    return f"""
You are an advanced well-being recovery planning AI with expertise in psychology, habit formation, emotional regulation, and long-term personal growth.

IMPORTANT CONTEXT (STRICT):
- Today's date is {current_date}
- All timelines must be future-oriented
- NEVER output past or current dates
- If uncertain, choose a conservative future recovery timeframe

TASK:
Analyze the user's psychological profile and generate a realistic, supportive, and structured recovery plan that promotes stability, growth, and sustainable well-being.

PROFILE DATA:
{json.dumps(ai_profile, indent=2)}

OUTPUT REQUIREMENTS:
Return ONLY valid JSON.
The JSON structure, keys, and data types MUST match the schema below EXACTLY.
Do NOT add, remove, or rename any fields.
Do NOT include commentary, markdown, or explanations outside the JSON object.

REQUIRED JSON FORMAT:
{{
  "goal_plan": ["string", "string"],
  "daily_routine": ["string", "string"],
  "activities": ["string", "string"],
  "growth_strengths": ["string", "string"],
  "recommended_content": ["string", "string"],
  "recovery_time_needed": "Month Year (must be AFTER {current_date})",
  "reasoning": "2–3 sentences"
}}

CONTENT GUIDELINES:
- goal_plan: Clear, achievable recovery goals aligned with the profile’s emotional and cognitive needs
- daily_routine: Practical daily habits that support consistency, grounding, and resilience
- activities: Low-pressure, restorative, or growth-oriented actions suitable for recovery
- growth_strengths: Strengths the user can realistically cultivate during recovery
- recommended_content: Educational or inspirational content types (not links) that support healing
- recovery_time_needed: A realistic future month and year (no ranges, no ambiguity)
- reasoning: Brief justification tying the recovery plan to the profile’s needs and challenges

RULES (MANDATORY):
- Be realistic and compassionate, not idealistic
- Avoid clinical diagnosis language
- Avoid absolute claims or guarantees
- Ensure internal consistency across all fields
- Output must be valid JSON and parseable without errors
"""


# ==============================
# SAFE JSON EXTRACTION
# ==============================


def extract_json(text):
    if not text or not text.strip():
        raise ValueError("Empty response from model")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            return json.loads(text[start:end])

        raise ValueError("Model response is not valid JSON")


# ==============================
# API CONTROLLER
# ==============================

def get_recovery_plan_controller():
    try:
        data = request.json
        user_id = data.get("user_id")

        # Validate input
        if not user_id:
            return jsonify({
                "status": "failed",
                "statusCode": 400,
                "message": "user_id is required",
                "data": {}
            }), 400

        # Fetch saved AI profile
        ai_profile = fetch_ai_profile(user_id)

        if not ai_profile:
            return jsonify({
                "status": "failed",
                "statusCode": 404,
                "message": "No AI profile found for this user",
                "data": {}
            }), 404

        # Build prompt
        prompt = build_recovery_prompt(ai_profile, current_date)

        # Call Mistral
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a practical mental wellness coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content

        # Extract valid JSON
        recovery_plan = extract_json(raw_output)

        save_recovery_plan(user_id, recovery_plan)

        # Success response
        return jsonify({
            "status": "success",
            "statusCode": 200,
            "message": "Recovery plan generated successfully",
            "data": recovery_plan
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to generate recovery plan",
            "error": str(e),
            "data": {}
        }), 500
