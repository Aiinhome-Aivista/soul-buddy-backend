import json
import mysql.connector
from flask import request, jsonify
from database.config import MYSQL_CONFIG

def get_users_controller():
    """
    Fetches the list of all users from the database (ai_soulbuddy.users).
    This allows the application to view user details like email, age, and emotional state.
    """
    try:
        # Establish connection using the config dictionary
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        
        # Use dictionary=True so the results are returned as JSON-ready objects
        cursor = conn.cursor(dictionary=True)

        # SQL to select all users
        sql = "SELECT * FROM ai_soulbuddy.users"
        
        cursor.execute(sql)
        users = cursor.fetchall()

        cursor.close()
        conn.close()

        # Return the list of users
        return jsonify({
            "status": "success", 
            "count": len(users),
            "data": users
        })

    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500