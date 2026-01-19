import uuid
import mysql.connector
from flask import request, jsonify
from werkzeug.security import check_password_hash
from datetime import datetime
from helper.captcha_helper import verify_captcha
from datetime import datetime


def email_login_controller(get_connection_func):
    data = request.json
    email = data.get("email")
    password = data.get("password")
    captcha_id = data.get("captchaId")
    captcha_value = data.get("captchaValue")

    # --- Input Validation (SINGLE BLOCK) ---
    if not email or not password or not captcha_id or not captcha_value:
        return jsonify({
            "status": "failed",
            "statusCode": 400,
            "message": "Email, password and captcha are required",
            "refreshCaptcha": True
        }), 400

    conn = None
    cursor = None
    try:
        conn = get_connection_func()
        if not conn:
            return jsonify({
                "status": "error",
                "statusCode": 500,
                "message": "Database connection failed"
            }), 500

        cursor = conn.cursor(dictionary=True)

        # --- CAPTCHA VALIDATION ---
        cursor.execute("""
            SELECT captcha_hash, expires_at
            FROM captcha_store
            WHERE id = %s
        """, (captcha_id,))
        captcha_row = cursor.fetchone()

        if (
            not captcha_row or
            captcha_row["expires_at"] < datetime.utcnow() or
            not verify_captcha(captcha_value, captcha_row["captcha_hash"])
        ):
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Invalid or expired captcha",
                "refreshCaptcha": True
            }), 401


        # --- FETCH USER ---
        cursor.execute(
            "SELECT user_id, full_name, password FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()

        if not user:
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Invalid email or password",
                "refreshCaptcha": True
            }), 401

        if not user["password"]:
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Account exists but has no password set. Please login via Google/Facebook.",
                "refreshCaptcha": True
            }), 401

        if not check_password_hash(user["password"], password):
            return jsonify({
                "status": "failed",
                "statusCode": 401,
                "message": "Invalid email or password",
                "refreshCaptcha": True
            }), 401

        # --- LOGIN SUCCESS ---
        # One-time captcha cleanup
        cursor.execute("DELETE FROM captcha_store WHERE id = %s", (captcha_id,))

        db_user_id = user["user_id"]
        db_full_name = user["full_name"]

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
            cursor.execute("""
                INSERT INTO session_log (user_id, session_id, created_at)
                VALUES (%s, %s, NOW())
            """, (db_user_id, session_id))

        cursor.execute(
            "UPDATE session_log SET last_login_at = NOW() WHERE user_id = %s",
            (db_user_id,)
        )
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
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": "Database error"
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
