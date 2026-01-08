from flask import request, jsonify
from werkzeug.security import generate_password_hash
from datetime import datetime


def reset_password_with_otp_controller(get_connection_func):
    data = request.json

    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("new_password")

    if not email or not otp or not new_password:
        return jsonify({
            "status": "failed",
            "message": "Email, OTP, and new password are required"
        }), 400

    conn = None
    cursor = None

    try:
        conn = get_connection_func()
        cursor = conn.cursor(dictionary=True)

        # Fetch OTP record
        cursor.execute(
            """
            SELECT otp_code, expires_at
            FROM password_resets
            WHERE email = %s
            """,
            (email,)
        )
        record = cursor.fetchone()

        if not record:
            return jsonify({
                "status": "failed",
                "message": "Invalid or expired OTP"
            }), 400

        if record["otp_code"] != otp:
            return jsonify({
                "status": "failed",
                "message": "Invalid OTP"
            }), 400

        if record["expires_at"] < datetime.now():
            return jsonify({
                "status": "failed",
                "message": "OTP has expired. Please request a new one."
            }), 400

        # Hash new password
        hashed_password = generate_password_hash(new_password)

        # Update user password
        cursor.execute(
            "UPDATE users SET password = %s WHERE email = %s",
            (hashed_password, email)
        )

        # Delete OTP (single-use)
        cursor.execute(
            "DELETE FROM password_resets WHERE email = %s",
            (email,)
        )

        conn.commit()

        return jsonify({
            "status": "success",
            "message": "Password has been reset successfully. You can now login."
        }), 200

    except Exception as e:
        print("Reset password error:", e)
        return jsonify({
            "status": "error",
            "message": "Something went wrong"
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
