import mysql.connector
from flask import request, jsonify
from datetime import datetime, timedelta


def subscription_controller(get_db_connection_func):
    data = request.json
    
    user_id = data.get('user_id')
    amount = data.get('amount')
    transaction_id = data.get('transaction_id')
    
    if not all([user_id, amount, transaction_id]):
        return jsonify({"error": "Missing user_id, amount, or transaction_id"}), 400

    conn = get_db_connection_func() # Call the helper function provided by app.py
    if not conn: 
        return jsonify({"error": "Database connection failed"}), 500
    
    cursor = conn.cursor()

    try:
        # 1. Check if user exists first
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
        if not cursor.fetchone():
            return jsonify({"error": "User ID not found"}), 404

        # 2. Insert subscription record
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30) # 30 Day Plan

        query = """
            INSERT INTO subscriptions 
            (user_id, amount, status, start_date, end_date, transaction_id) 
            VALUES (%s, %s, 'active', %s, %s, %s)
        """
        cursor.execute(query, (user_id, amount, start_date, end_date, transaction_id))
        conn.commit()

        return jsonify({"message": "Subscription active", "status": "success"}), 201

    except mysql.connector.Error as err:
        return jsonify({"error": f"Database operation failed: {str(err)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()