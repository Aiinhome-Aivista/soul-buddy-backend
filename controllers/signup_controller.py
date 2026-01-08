# import uuid
# import mysql.connector
# from flask import request, jsonify

# def signup_controller(get_db_connection_func):
#     data = request.json
    
#     # Extract Data
#     full_name = data.get('full_name')
#     email = data.get('email')
#     password = data.get('password')
#     age = data.get('age')
#     gender = data.get('gender')
#     work = data.get('work')
#     health = data.get('health')
#     emotional_state = data.get('emotional_state')
#     relationship = data.get('relationship')

#     if not all([full_name, email]):
#         return jsonify({"error": "Full name, email are required"}), 400

#     conn = get_db_connection_func() # Call the helper function provided by app.py
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
#         values = (user_id, full_name, email, password, age, gender, work, health, emotional_state, relationship)
        
#         cursor.execute(query, values)
#         conn.commit()

#         return jsonify({"message": "User registered successfully", "user_id": user_id}), 201

#     except mysql.connector.Error as err:
#         if err.errno == 1062: 
#             return jsonify({"error": "Email already exists"}), 409
#         return jsonify({"error": f"Database operation failed: {str(err)}"}), 500
#     finally:
#         if cursor: cursor.close()
#         if conn: conn.close()



import uuid
import mysql.connector
from flask import request, jsonify
from werkzeug.security import generate_password_hash

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

    # Required fields
    if not full_name or not email:
        return jsonify({"error": "Full name and email are required"}), 400

    # Hash password ONLY if provided
    hashed_password = None
    if raw_password:
        hashed_password = generate_password_hash(raw_password)

    conn = get_db_connection_func()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()

    try:
        user_id = str(uuid.uuid4())

        query = """
            INSERT INTO users 
            (user_id, full_name, email, password, age, gender, work, health, emotional_state, relationship)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            user_id,
            full_name,
            email,
            hashed_password,  # NULL if no password
            age,
            gender,
            work,
            health,
            emotional_state,
            relationship
        )

        cursor.execute(query, values)
        conn.commit()

        return jsonify({
            "message": "User registered successfully",
            "user_id": user_id
        }), 201

    except mysql.connector.Error as err:
        if err.errno == 1062:
            return jsonify({"error": "Email already exists"}), 409
        return jsonify({"error": f"Database operation failed: {str(err)}"}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
