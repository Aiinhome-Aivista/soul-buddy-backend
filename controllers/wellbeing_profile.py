import json
from flask import request, jsonify
from openai import OpenAI

from database.config import MISTRAL_API_KEY, MYSQL_CONFIG
from database.db_handler import fetch_user_answers
from database.db_handler import save_ai_profile


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

def build_prompt(answers):
    return f"""
You are a trained psychological counsellor and well-being assessment AI.

Your role is to gently interpret the user's responses using a psychology-informed lens.
This is a reflective well-being assessment, NOT a medical or clinical diagnosis.

IMPORTANT RULES (STRICT):
- Respond with ONLY valid JSON
- Do NOT include markdown
- Do NOT include backticks
- Do NOT include explanations outside the JSON
- The JSON structure and keys MUST match exactly

ANALYSIS FRAMEWORK:
Base your reasoning on these psychological dimensions:
- Thoughts (beliefs, worries, self-talk, interpretations)
- Feelings (emotional states, mood patterns, intensity)
- Behaviour (actions, avoidance, habits, coping responses)

INTERPRETATION GUIDELINES:
- Infer patterns carefully and conservatively from the answers
- Avoid clinical labels or diagnostic language
- Use compassionate, non-judgmental wording
- If information is unclear, choose the most reasonable neutral interpretation
- Focus on impact on daily functioning and well-being

USER ANSWERS:
{json.dumps(answers, indent=2)}

RETURN JSON EXACTLY IN THIS STRUCTURE:

{{
  "negative_effects_level": "Low | Normal | Medium | High",
  "main_difficulty": "Brief description of the core psychological challenge",
  "trigger": "Primary situation, pattern, or experience contributing to the difficulty",
  "energy_level": "Low | Normal | High",
  "challenging_period": "Recent months | Whole life",
  "explanation": "Single empathetic sentence written in a calm, supportive counselling tone"
}}

EXPLANATION RULES (MANDATORY):
- The explanation MUST be written based on the selected negative_effects_level

If negative_effects_level = Low:
- Emphasize limited impact on daily life
- Normalize the experience and reflect general coping

If negative_effects_level = Medium:
- Acknowledge noticeable emotional strain
- Reflect occasional interference with daily functioning

If negative_effects_level = High:
- Convey strong empathy
- Reflect significant emotional burden, increased worry, reduced energy, or difficulty maintaining well-being

ADDITIONAL EXPLANATION CONSTRAINTS:
- Exactly 1 sentence
- Validates the user’s experience
- Reflects emotional understanding
- Does NOT give advice, strategies, or solutions
- Avoids alarming, clinical, or diagnostic language
"""


# ==============================
# SAFE JSON EXTRACTION
# ==============================

def extract_json(text):
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

def get_wellbeing_profile_controller():
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

        # Fetch answers from DB
        user_answers = fetch_user_answers(user_id)

        if not user_answers:
            return jsonify({
                "status": "failed",
                "statusCode": 404,
                "message": "No responses found for this user",
                "data": {}
            }), 404

        # Build prompt & call Mistral
        prompt = build_prompt(user_answers)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                    "content": "You are a calm, empathetic well-being expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        profile = extract_json(response.choices[0].message.content)

        # SAVE AI RESULT
        save_ai_profile(user_id, profile)

        # Success response
        return jsonify({
            "status": "success",
            "statusCode": 200,
            "message": "Well-being profile generated successfully",
            "data": profile
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to generate well-being profile",
            "error": str(e),
            "data": {}
        }), 500
