import json
from flask import request, jsonify
from database.db_connection import get_db_connection


# =========================================
# 1️⃣ LIST USERS WITH SESSIONS (ADMIN VIEW)
# =========================================

def list_users_with_sessions_controller():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                ch.user_id,
                ch.session_id,
                MAX(ch.created_at) AS last_activity
            FROM conversation_history ch
            GROUP BY ch.user_id, ch.session_id
            ORDER BY last_activity DESC
        """)

        data = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to fetch sessions",
            "error": str(e)
        }), 500


# =========================================
# 2️⃣ GET FULL CONVERSATION BY SESSION
# =========================================

def get_session_conversation_controller(session_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                user_input,
                model_response,
                created_at
            FROM conversation_history
            WHERE session_id = %s
            ORDER BY created_at ASC
        """, (session_id,))

        data = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to fetch conversation",
            "error": str(e)
        }), 500


# =========================================
# 3️⃣ SAVE EXPERT INSIGHT
# =========================================

def save_expert_insight_controller():
    try:
        data = request.get_json()

        required_fields = ["user_id", "insight_text"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "failed",
                    "statusCode": 400,
                    "message": f"{field} is required"
                }), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expert_insights
            (
                expert_id,
                user_id,
                session_id,
                insight_text,
                tags
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (
            1,  # DEFAULT SYSTEM EXPERT
            data["user_id"],
            data.get("session_id"),
            data["insight_text"],
            json.dumps(data.get("tags", []))
        ))



        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "message": "Expert insight saved successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Failed to save expert insight",
            "error": str(e)
        }), 500
