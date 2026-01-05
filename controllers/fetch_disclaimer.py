import mysql.connector
from flask import jsonify
from database.config import MYSQL_CONFIG


def fetch_disclaimer():
    """
    Fetch active disclaimers sections
    """
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT id, title, content, last_updated
            FROM disclaimers
            WHERE is_active = 1
            ORDER BY id ASC;
        """

        cursor.execute(query)
        results = cursor.fetchall()

        return jsonify({
            "success": True,
            "data": results
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
