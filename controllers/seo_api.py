# File: seo_api.py
from flask import request, jsonify
# 1. REMOVE: from database.config import MYSQL_CONFIG
# 2. ADD: Import the actual connection function
from database.db_connection import get_db_connection

# --- 1. INSERT (Create) ---
def create_seo_entry():
    data = request.get_json()
    keyword = data.get('target_keyword')
    title = data.get('seo_title')
    desc = data.get('meta_description')
    page_url = data.get('page_url')

    if not keyword or not page_url:
        return jsonify({"error": "target_keyword and page_url are required"}), 400


    # 3. FIX: Call the function, not the config dict
    conn = get_db_connection()
    
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO seo_metadata 
        (target_keyword, page_url, seo_title, meta_description)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (keyword, page_url, title, desc))

        conn.commit()
        new_id = cursor.lastrowid
        return jsonify({"success": True, "message": "SEO data added", "id": new_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


# --- 2. GET (Read) ---
def get_seo_data(seo_id=None):
    # FIX here as well
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor() 
        if seo_id:
            query = "SELECT * FROM seo_metadata WHERE id = %s"
            cursor.execute(query, (seo_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({"error": "Not found"}), 404
        else:
            query = "SELECT * FROM seo_metadata ORDER BY id DESC"
            cursor.execute(query)
            result = cursor.fetchall()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


# --- 3. UPDATE ---
def update_seo_entry(seo_id):
    data = request.get_json()
    keyword = data.get('target_keyword')
    title = data.get('seo_title')
    desc = data.get('meta_description')
    page_url = data.get('page_url')

    if not keyword or not page_url:
        return jsonify({"error": "target_keyword and page_url are required"}), 400


    # FIX here as well
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500


    try:
        cursor = conn.cursor()
        check_query = "SELECT id FROM seo_metadata WHERE id = %s"
        cursor.execute(check_query, (seo_id,))
        if not cursor.fetchone():
            return jsonify({"error": "SEO entry not found"}), 404

        query = """
        UPDATE seo_metadata
        SET target_keyword=%s,
            page_url=%s,
            seo_title=%s,
            meta_description=%s
        WHERE id=%s
        """
        cursor.execute(query, (keyword, page_url, title, desc, seo_id))

        conn.commit()
        return jsonify({"success": True, "message": "Updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


# --- 4. DELETE ---
def delete_seo_entry(seo_id):
    # FIX here as well
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor()
        query = "DELETE FROM seo_metadata WHERE id = %s"
        cursor.execute(query, (seo_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({"error": "Entry not found"}), 404
        return jsonify({"success": True, "message": "Deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
