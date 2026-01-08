import uuid
from flask import request, jsonify
from datetime import datetime

def submit_response(get_connection_func):
    """
    Handles POST request to submit a list of user responses.
    Aligned with DB schema: response_id is INT AUTO_INCREMENT.
    """
    data = request.json
    
    user_id = data.get('user_id')
    responses = data.get('responses') 
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    if not responses or not isinstance(responses, list):
        return jsonify({"error": "A list of responses is required"}), 400

    conn = get_connection_func()
    if not conn: 
        return jsonify({"error": "Database error"}), 500
    
    cursor = conn.cursor()
    
    try:
        # Verify User ID exists
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
        if not cursor.fetchone():
            return jsonify({"error": "User ID not found"}), 404

        # 1. Prepare Data: Removed response_id so DB can AUTO_INCREMENT
        response_records = []
        now = datetime.now()
        
        for response in responses:
            question_id = response.get('question_id')
            answer_value = response.get('answer_value')
            answer_text = response.get('answer_text') 

            if question_id is None or answer_value is None:
                conn.rollback()
                return jsonify({"error": "Missing question_id or answer_value"}), 400

            if question_id != 13:
                answer_text = None

            response_records.append((
                user_id,
                question_id,
                answer_value,
                answer_text,
                now
            ))

            
        # 2. Updated Query: Removed response_id from columns and values
        query = """
                INSERT INTO user_responses 
                (user_id, question_id, answer_value, answer_text, response_timestamp)
                VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.executemany(query, response_records)
        conn.commit()
        
        return jsonify({
            "message": f"Successfully submitted {len(response_records)} responses",
            "status": "success"
        }), 201

    except Exception as err:
        conn.rollback()
        print(f"Error during response submission: {err}") 
        return jsonify({"error": f"Database error: {str(err)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()