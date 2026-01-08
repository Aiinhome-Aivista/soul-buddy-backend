import random
from flask import request, jsonify
from datetime import datetime, timedelta

# Email sender
from helper.email_sender import send_otp_email



def request_password_reset_controller(get_connection_func):
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({
            "status": "failed",
            "message": "Email is required"
        }), 400

    conn = None
    cursor = None

    try:
        conn = get_connection_func()
        cursor = conn.cursor(dictionary=True)

        # Check if user exists
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        # ALWAYS return same message (security)
        if not user:
            return jsonify({
                "status": "success",
                "message": "If this email exists, an OTP has been sent."
            }), 200

        # Generate OTP
        otp = str(random.randint(100000, 999999))
        expires_at = datetime.now() + timedelta(minutes=10)

        # Remove old OTP
        cursor.execute("DELETE FROM password_resets WHERE email = %s", (email,))

        # Save new OTP
        cursor.execute(
            """
            INSERT INTO password_resets (email, otp_code, expires_at)
            VALUES (%s, %s, %s)
            """,
            (email, otp, expires_at)
        )
        conn.commit()

        # Send OTP email
        send_otp_email(email, otp)

        return jsonify({
            "status": "success",
            "message": "If this email exists, an OTP has been sent."
        }), 200

    except Exception as e:
        print("Forgot password error:", e)
        return jsonify({
            "status": "error",
            "message": "Something went wrong"
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
