import mysql.connector
import pymysql
from flask import jsonify, request
from datetime import datetime
from database.config import MYSQL_CONFIG


def get_user_details():
    """
    Fetch single user details + subscription info
    """

    data = request.json or {}
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({
            "success": False,
            "message": "user_id is required"
        }), 400

    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # ---------------- USER DETAILS ----------------
        user_query = """
            SELECT 
                user_id,
                full_name,
                email,
                age,
                gender,
                work,
                health,
                emotional_state,
                relationship,
                created_at,
                updated_at
            FROM users
            WHERE user_id = %s
        """
        cursor.execute(user_query, (user_id,))
        user_data = cursor.fetchone()

        if not user_data:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

        # ---------------- SUBSCRIPTION DETAILS ----------------
        sub_query = """
            SELECT
                plan_name,
                status,
                start_date,
                end_date
            FROM subscriptions
            WHERE user_id = %s
            ORDER BY start_date DESC
            LIMIT 1
        """
        cursor.execute(sub_query, (user_id,))
        subscription = cursor.fetchone()

        subscription_data = None

        if subscription:
            end_date = subscription.get("end_date")
            today = datetime.now()

            if end_date:
                validity_days_left = max(
                    0,
                    (end_date - today).days
                )
            else:
                validity_days_left = None

            subscription_data = {
                "plan_name": subscription.get("plan_name"),
                "status": subscription.get("status"),
                "start_date": subscription.get("start_date"),
                "end_date": subscription.get("end_date"),
                "validity_days_left": validity_days_left
            }

        # ---------------- FINAL RESPONSE ----------------
        return jsonify({
            "success": True,
            "data": {
                "user": user_data,
                "subscription": subscription_data
            }
        })

    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return jsonify({
            "success": False,
            "message": str(err)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()



def get_connection():
    return pymysql.connect(**MYSQL_CONFIG)

# ================= SUBSCRIPTION DETAILS (PLAIN GET) =================
def get_user_subscription_controller():
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = """
            SELECT
                u.user_id,
                u.full_name,
                s.plan_name,
                DATE(s.start_date) AS start_date,
                DATE(s.end_date) AS end_date,
                GREATEST(DATEDIFF(s.end_date, CURDATE()), 0) AS remaining_days
            FROM subscriptions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.status = 'active'
            ORDER BY s.created_at DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            return jsonify({
                "status": "failed",
                "message": "No active subscriptions found",
                "count": 0,
                "data": [],
                "statusCode": 404
            }), 404

        subscriptions = []

        for r in rows:
            subscriptions.append({
                "user_id": r["user_id"],
                "full_name": r["full_name"],
                "subscription": {
                    "plan_name": r["plan_name"],
                    "start_date": str(r["start_date"]),
                    "end_date": str(r["end_date"]),
                    "remaining_days": r["remaining_days"]
                }
            })

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "count": len(subscriptions),
            "data": subscriptions
        }), 200

    except Exception as e:
        return jsonify({
            "status": "failed",
            "message": "Internal server error",
            "error": str(e),
            "statusCode": 500
        }), 500
