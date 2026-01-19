
# import uuid
# import mysql.connector
# from flask import request, jsonify
# from werkzeug.security import generate_password_hash

# def signup_controller(get_db_connection_func):
#     data = request.json

#     # Extract Data
#     full_name = data.get('full_name')
#     email = data.get('email')
#     raw_password = data.get('password')  # OPTIONAL
#     age = data.get('age')
#     gender = data.get('gender')
#     work = data.get('work')
#     health = data.get('health')
#     emotional_state = data.get('emotional_state')
#     relationship = data.get('relationship')

#     # Required fields
#     if not full_name or not email:
#         return jsonify({"error": "Full name and email are required"}), 400

#     # Hash password ONLY if provided
#     hashed_password = None
#     if raw_password:
#         hashed_password = generate_password_hash(raw_password)

#     conn = get_db_connection_func()
#     if not conn:
#         return jsonify({"error": "Database connection failed"}), 500

#     cursor = conn.cursor()

#     try:
#         user_id = str(uuid.uuid4())

#         query = """
#             INSERT INTO users 
#             (user_id, full_name, email, password, age, gender, work, health, emotional_state, relationship)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """

#         values = (
#             user_id,
#             full_name,
#             email,
#             hashed_password,  # NULL if no password
#             age,
#             gender,
#             work,
#             health,
#             emotional_state,
#             relationship
#         )

#         cursor.execute(query, values)
#         conn.commit()

#         return jsonify({
#             "message": "User registered successfully",
#             "user_id": user_id
#         }), 201

#     except mysql.connector.Error as err:
#         if err.errno == 1062:
#             return jsonify({"error": "Email already exists"}), 409
#         return jsonify({"error": f"Database operation failed: {str(err)}"}), 500

#     finally:
#         if cursor:
#             cursor.close()
#         if conn:
#             conn.close()


import uuid
import mysql.connector
from flask import request, jsonify
from werkzeug.security import generate_password_hash
from datetime import datetime
from helper.captcha_helper import verify_captcha

def signup_controller(get_db_connection_func):
    data = request.json

    # Extract Data
    full_name = data.get('full_name')
    email = data.get('email')
    raw_password = data.get('password')  # OPTIONAL
    age = data.get('age')
    gender = data.get('gender')
    work = data.get('work')
    health = data.get('health')
    emotional_state = data.get('emotional_state')
    relationship = data.get('relationship')
    # CAPTCHA
    captcha_id = data.get("captchaId")
    captcha_value = data.get("captchaValue")

    # Required fields
    if not full_name or not email:
        return jsonify({"error": "Full name and email are required"}), 400
     # CAPTCHA VALIDATION (INPUT)
    if not captcha_id or not captcha_value:
        return jsonify({
            "error": "Captcha is required",
            "refreshCaptcha": True
        }), 400

    conn = get_db_connection_func()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor(dictionary=True)

    try:
        # CAPTCHA VALIDATION (DB)
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
                "error": "Invalid or expired captcha",
                "refreshCaptcha": True
            }), 401

        # 🔥 One-time use captcha
        cursor.execute("DELETE FROM captcha_store WHERE id = %s", (captcha_id,))

        # 🔎 1. Check if email already exists
        cursor.execute(
            "SELECT user_id FROM users WHERE email = %s",
            (email,)
        )
        existing_user = cursor.fetchone()

        if existing_user:
            return jsonify({
                "error": "Email has already been registered. Please try with another email ID.",
                "refreshCaptcha": True
            }), 409


        # 🔐 2. Hash password only if provided
        hashed_password = None
        if raw_password:
            hashed_password = generate_password_hash(raw_password)

        # 🆔 3. Create user
        user_id = str(uuid.uuid4())

        insert_query = """
            INSERT INTO users
            (user_id, full_name, email, password, age, gender, work, health, emotional_state, relationship)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            user_id,
            full_name,
            email,
            hashed_password,
            age,
            gender,
            work,
            health,
            emotional_state,
            relationship
        ))

        conn.commit()

        return jsonify({
            "message": "User registered successfully",
            "user_id": user_id
        }), 201

    except mysql.connector.Error as err:
        return jsonify({
            "error": f"Database operation failed: {str(err)}"
        }), 500

    finally:
        cursor.close()
        conn.close()
