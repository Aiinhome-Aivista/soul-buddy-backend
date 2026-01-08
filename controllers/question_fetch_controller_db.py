import mysql.connector
from flask import jsonify

COLUMN_NAMES = [
    'question_id', 'question_text','question_description', 'question_type',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'other_options', 'more_options'
]
OPTION_KEYS = ['option_1', 'option_2', 'option_3', 'option_4', 'option_5']


def question_fetch_controller_db(get_connection_func):
    """
    Fetches all questions from the 'questions' table, handles tuple results,
    and formats them for the API response.
    """
    conn = None
    cursor = None
    questions_list = []
    
    try:
        
        conn = get_connection_func()
        cursor = conn.cursor() 
        
        # 2. Execute query to fetch all questions
        query = f"SELECT {', '.join(COLUMN_NAMES)} FROM questions ORDER BY question_id ASC"
        cursor.execute(query)
        db_questions_tuples = cursor.fetchall()
        
        # 3. Process records: Manually map tuples to dictionaries
        for q_tuple in db_questions_tuples:
            # Manually map tuple to a dictionary using the predefined column names
            q = dict(zip(COLUMN_NAMES, q_tuple))
            
            # Determine the options array based on question_type
            # ----------------------------------------------------------------------------------
            # FIX: Include 'Multiple Select' in the types that require option compilation
            if q['question_type'] in ['Single Select', 'Agreement Scale', 'Multiple Select','Single Select with Text']:
            # ----------------------------------------------------------------------------------
                options = []
                
                # Compile options from the pre-defined option keys
                for option_key in OPTION_KEYS:
                    # Use .get() defensively as column might not be present or value might be None
                    value = q.get(option_key) 
                    # Only append the option if it's not None and not an empty string
                    if value is not None and value != '':
                        options.append(value)
                
                # This seems to be a fallback logic for 'Yes/No' questions
                if not options:
                     options = ["Yes", "No"] 
                
            else:
                # Text Input or other types have no discrete options
                options = None

            questions_list.append({
                "id": q['question_id'],
                "text": q['question_text'],
                 "description": q['question_description'],
                "type": q['question_type'],
                "options": options,
            })

        return jsonify(questions_list), 200

    except mysql.connector.Error as e:
        # ... (error handling remains the same) ...
        print(f"Database error during question fetch: {e}")
        return jsonify({
            "status": "error",
            "message": f"Database error fetching questions: {str(e)}"
        }), 500
    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"Unexpected error during question fetch: {e}")
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()