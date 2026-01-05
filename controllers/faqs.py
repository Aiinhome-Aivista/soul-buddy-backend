import mysql.connector
from flask import jsonify
from database.config import MYSQL_CONFIG


def fetch_grouped_faqs():
    """
    Fetch active FAQs grouped by category (UI-friendly format)
    """
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT category, question, answer
            FROM faqs
            WHERE is_active = 1
            ORDER BY category ASC, id ASC;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        grouped = {}

        for row in rows:
            category = row["category"]

            if category not in grouped:
                grouped[category] = []

            grouped[category].append({
                "question": row["question"],
                "answer": row["answer"]
            })

        # Convert to UI-friendly array
        response_data = []
        for category, faqs in grouped.items():
            response_data.append({
                "category": category,
                "faqs": faqs
            })

        return jsonify({
            "success": True,
            "data": response_data
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
