import mysql.connector
from flask import jsonify, request
from database.config import MYSQL_CONFIG

def get_user_details():
    """
    Fetch single user details by user_id
    """

    data = request.json
    user_id = data.get('user_id')

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

        query = """
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

        cursor.execute(query, (user_id,))
        user_data = cursor.fetchone()

        if user_data:
            return jsonify({
                "success": True,
                "data": user_data
            })
        else:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

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