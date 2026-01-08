import json
from flask import jsonify
from database.db_connection import get_db_connection

def get_user_sessions_controller(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            SELECT
                session_id,
                MIN(created_at) AS session_start,
                MAX(created_at) AS last_activity
            FROM conversation_history
            WHERE user_id = %s
            GROUP BY session_id
            ORDER BY last_activity DESC
        """

        cursor.execute(query, (user_id,))
        sessions = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "message": "User sessions fetched successfully",
            "data": sessions
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to fetch user sessions",
            "error": str(e)
        }), 500



# =========================================
# API 2: User wellbeing summary (GRAPH API)
# =========================================
ENERGY_LEVEL_MAP = {
    "Low": {
        "score": 30,
        "status": "Needs Attention"
    },
    "Normal": {
        "score": 70,
        "status": "Stable"
    },
    "High": {
        "score": 90,
        "status": "Good"
    }
}

NEGATIVE_EFFECTS_MAP = {
    "Low": {
        "score": 20,
        "status": "Minimal"
    },
    "Medium": {
        "score": 60,
        "status": "Moderate"
    },
    "High": {
        "score": 85,
        "status": "Severe"
    }
}

def user_wellbeing_summary_controller(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()   # remove dictionary=True

        query = """
            SELECT ai_profile
            FROM wellbeing_ai_results
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if not row:
            return jsonify({
                "status": "error",
                "statusCode": 404,
                "message": "No wellbeing data found"
            })

        # row[0] because cursor() returns tuple
        ai_profile = json.loads(row[0])

        energy_label = ai_profile.get("energy_level", "Normal")
        negative_label = ai_profile.get("negative_effects_level", "Low")

        response = {
            "status": "success",
            "statusCode": 200,
            "data": {
                "energy_level": {
                    "label": energy_label,
                    "score": ENERGY_LEVEL_MAP.get(energy_label, {}).get("score", 0),
                    "status": ENERGY_LEVEL_MAP.get(energy_label, {}).get("status", "Unknown")
                },
                "negative_effects": {
                    "label": negative_label,
                    "score": NEGATIVE_EFFECTS_MAP.get(negative_label, {}).get("score", 0),
                    "status": NEGATIVE_EFFECTS_MAP.get(negative_label, {}).get("status", "Unknown")
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to fetch wellbeing summary",
            "error": str(e)
        }), 500
