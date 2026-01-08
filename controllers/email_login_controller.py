import uuid
import mysql.connector
from flask import request, jsonify
from werkzeug.security import check_password_hash

def email_login_controller(get_connection_func):
    data = request.json
    email = data.get("email")
    password = data.get("password")

    # --- Input Validation ---
    if not email or not password:
        return jsonify({
            "status": "failed",
            "statusCode": 400,
            "message": "Email and password are required"
        }), 400

    conn = None
    cursor = None
    try:
        conn = get_connection_func()
        if not conn:
            return jsonify({"status": "error", "statusCode": 500, "message": "Database connection failed"}), 500

        cursor = conn.cursor(dictionary=True)

        # 1. FETCH USER BY EMAIL
        # We need the password hash to verify credentials
        query = "SELECT user_id, full_name, password FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        # 2. VERIFY USER EXISTS AND PASSWORD MATCHES
        if not user:
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Invalid email or password"
            }), 401
            
        # Check if user has a password set (Social login users might have NULL password)
        if not user['password']:
             return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Account exists but has no password set. Please login via Google/Facebook."
            }), 401

        if not check_password_hash(user['password'], password):
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Invalid email or password"
            }), 401

        # --- Login Successful ---
        db_user_id = user["user_id"]
        db_full_name = user["full_name"]

        # 3. SESSION MANAGEMENT (Same logic as your social login)
        cursor.execute("""
            SELECT session_id
            FROM session_log
            WHERE user_id = %s
            ORDER BY created_at DESC LIMIT 1
        """, (db_user_id,))
        session_row = cursor.fetchone()

        if session_row:
            session_id = session_row["session_id"]
        else:
            session_id = str(uuid.uuid4())
            cursor.execute("INSERT INTO session_log (user_id, session_id, created_at) VALUES (%s, %s, NOW())", (db_user_id, session_id))

        # Update last_login_at
        cursor.execute("UPDATE session_log SET last_login_at = NOW() WHERE user_id = %s", (db_user_id,))
        conn.commit()

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "message": "Login successful",
            "user_id": db_user_id,
            "full_name": db_full_name,
            "login_type": "email",
            "session_id": session_id
        }), 200

    except mysql.connector.Error as e:
        print(f"Database error during email login: {e}")
        return jsonify({"status": "error", "statusCode": 500, "message": f"Database error: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error during email login: {e}")
        return jsonify({"status": "error", "statusCode": 500, "message": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()