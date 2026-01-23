from flask import request, jsonify
import pymysql
from database.config import MYSQL_CONFIG

def get_connection():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        cursorclass=pymysql.cursors.DictCursor
    )

def get_transaction_history(user_id):
    status = request.args.get("status")  # optional filter

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        conn = get_connection()
        with conn.cursor() as cursor:

            sql = """
                SELECT 
                    plan_name,
                    amount,
                    status,
                    start_date,
                    payment_method,
                    transaction_id
                FROM subscriptions
                WHERE user_id = %s
            """
            params = [user_id]

            if status:
                sql += " AND status = %s"
                params.append(status)

            sql += " ORDER BY start_date DESC"

            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

        conn.close()

        return jsonify({
            "status": "success",
            "user_id": user_id,
            "count": len(rows),
            "transactions": rows
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500