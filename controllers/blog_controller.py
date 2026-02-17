import os
import pymysql
import logging
import json
from flask import request, jsonify
from database.config import MYSQL_CONFIG
from werkzeug.utils import secure_filename
from database.config import BASE_URL

# Image save korar path setup
UPLOAD_FOLDER = 'static/blog_images'
CONTENT_IMAGES_FOLDER = 'static/blog_images/content'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(CONTENT_IMAGES_FOLDER):
    os.makedirs(CONTENT_IMAGES_FOLDER)

def get_connection():
    return pymysql.connect(**MYSQL_CONFIG)

# --- Tags Cleaner Helper ---
def clean_tags_input(tags_input):
    if not tags_input:
        return ""
    
    tags_str = str(tags_input).strip()
    
    if tags_str in ['[]', '[""]', "['']", '', 'null', 'None']:
        return ""
    
    if tags_str.startswith('[') and tags_str.endswith(']'):
        try:
            tags_list = json.loads(tags_str)
            tags_list = [str(tag).strip() for tag in tags_list if tag and str(tag).strip()]
            if not tags_list:
                return ""
            return ','.join(tags_list)
        except (json.JSONDecodeError, ValueError):
            tags_str = tags_str.strip('[]').replace('"', '').replace("'", '')
            if not tags_str:
                return ""
    
    tags_list = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    if not tags_list:
        return ""
    
    return ','.join(tags_list)

def parse_tags_output(tags_str):
    if not tags_str:
        return []
    
    if tags_str.startswith('['):
        try:
            parsed = json.loads(tags_str)
            return [str(tag).strip() for tag in parsed if tag and str(tag).strip()]
        except:
            pass
    
    tags_list = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    return tags_list

# --- Role Check Helper ---
def check_admin_access(author_name):
    if not author_name:
        return False, "Author name is required"
    
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT role FROM staff_users WHERE full_name = %s", (author_name,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return False, "Author not found in staff list"
    
    if user['role'] in ['admin', 'super_admin']:
        return True, "Success"
    
    return False, "Access denied. Only Admin or Super Admin can perform this action."

# ================= CATEGORY CONTROLLERS =================

def get_all_categories_controller():
    """Get all categories with their subcategories"""
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM categories ORDER BY name ASC")
        categories = cursor.fetchall()
        conn.close()
        
        # Parse subcategories JSON
        for category in categories:
            if category.get('subcategories'):
                try:
                    category['subcategories'] = json.loads(category['subcategories'])
                except:
                    category['subcategories'] = []
            else:
                category['subcategories'] = []
        
        return jsonify({"status": "success", "data": categories}), 200
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

def create_category_controller():
    """Create a new category with optional subcategories"""
    data = request.get_json()
    author_name = data.get("author_name") 
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    name = data.get("name")
    subcategories = data.get("subcategories", [])
    
    # Convert subcategories to JSON string
    subcategories_json = json.dumps(subcategories) if subcategories else None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO categories (name, subcategories, created_by) VALUES (%s, %s, %s)", 
            (name, subcategories_json, author_name)
        )
        conn.commit()
        category_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            "status": "success", 
            "message": "Category created",
            "category_id": category_id
        }), 201
    except pymysql.IntegrityError:
        return jsonify({"status": "failed", "message": "Category already exists"}), 409

def update_category_controller(cat_id):
    """Update category name and/or subcategories"""
    data = request.get_json()
    author_name = data.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    new_name = data.get("name")
    subcategories = data.get("subcategories")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build update query dynamically
    updates = []
    params = []
    
    if new_name:
        updates.append("name = %s")
        params.append(new_name)
    
    if subcategories is not None:
        updates.append("subcategories = %s")
        params.append(json.dumps(subcategories) if subcategories else None)
    
    if not updates:
        return jsonify({"status": "failed", "message": "No fields to update"}), 400
    
    params.append(cat_id)
    query = f"UPDATE categories SET {', '.join(updates)} WHERE id = %s"
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": "Category updated"}), 200


def delete_category_controller(cat_id):
    """
    Delete category and all its blogs
    Cascade: Deletes all blogs in this category
    """
    author_name = request.args.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    conn = get_connection()
    cursor = conn.cursor()
    
    # First delete all blogs in this category
    cursor.execute("DELETE FROM blogs WHERE category_id = %s", (cat_id,))
    
    # Then delete the category
    cursor.execute("DELETE FROM categories WHERE id = %s", (cat_id,))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success", 
        "message": "Category and all its blogs deleted successfully"
    }), 200


def add_subcategory_controller(cat_id):
    """Multiple and Single subcategory add support"""
    data = request.get_json()
    author_name = data.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403
    
    new_input = data.get("subcategory_name")
    if not new_input:
        return jsonify({"status": "failed", "message": "subcategory_name is required"}), 400
    
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT subcategories FROM categories WHERE id = %s", (cat_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"status": "failed", "message": "Category not found"}), 404
    
    # Existing list parse kora
    try:
        current_subcats = json.loads(result['subcategories']) if result['subcategories'] else []
        if not isinstance(current_subcats, list): current_subcats = []
    except:
        current_subcats = []

    # Jodi input list hoy, tobe extend koro; string hole append koro
    if isinstance(new_input, list):
        for item in new_input:
            if item.strip() and item.strip() not in current_subcats:
                current_subcats.append(item.strip())
    else:
        if new_input.strip() and new_input.strip() not in current_subcats:
            current_subcats.append(new_input.strip())
    
    cursor.execute(
        "UPDATE categories SET subcategories = %s WHERE id = %s",
        (json.dumps(current_subcats), cat_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success",
        "message": "Subcategories added",
        "subcategories": current_subcats
    }), 200

def get_subcategories_by_category_name():
    """
    Fetch subcategories based on a provided category_name.
    Payload (JSON): { "category_name": "Health" }
    """
    try:
        data = request.get_json()
        category_name = data.get("category_name", "").strip()

        if not category_name:
            return jsonify({
                "status": "failed", 
                "message": "category_name is required in payload"
            }), 400

        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # Category name diye subcategories khunje ber kora
        query = "SELECT subcategories FROM categories WHERE name = %s"
        cursor.execute(query, (category_name,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({
                "status": "failed", 
                "message": f"Category '{category_name}' not found"
            }), 404

        # Subcategories JSON string ke Python list-e convert kora
        subcategories = []
        if result['subcategories']:
            try:
                subcategories = json.loads(result['subcategories'])
            except:
                subcategories = []

        return jsonify({
            "status": "success",
            "category_name": category_name,
            "count": len(subcategories),
            "subcategories": subcategories
        }), 200

    except Exception as e:
        logging.error(f"Error fetching subcategories: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def update_subcategory_controller(cat_id):
    """Update a specific subcategory name"""
    data = request.get_json()
    author_name = data.get("author_name")

    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    old_name = data.get("old_name")
    new_name = data.get("new_name")

    if not old_name or not new_name:
        return jsonify({"status": "failed", "message": "Both old_name and new_name are required"}), 400

    # Normalize input
    old_clean = old_name.strip().lower()
    new_clean = new_name.strip()

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT subcategories FROM categories WHERE id = %s", (cat_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return jsonify({"status": "failed", "message": "Category not found"}), 404

    try:
        current_subcats = json.loads(result['subcategories']) if result['subcategories'] else []
    except json.JSONDecodeError:
        conn.close()
        return jsonify({"status": "failed", "message": "Invalid subcategory data format"}), 500

    if not isinstance(current_subcats, list):
        conn.close()
        return jsonify({"status": "failed", "message": "Subcategories format invalid"}), 500

    # Normalize DB values for comparison
    normalized_subcats = [x.strip().lower() for x in current_subcats]

    # Check if old exists
    if old_clean not in normalized_subcats:
        conn.close()
        return jsonify({"status": "failed", "message": "Subcategory not found"}), 404

    # Prevent duplicate new name
    if new_clean.lower() in normalized_subcats and new_clean.lower() != old_clean:
        conn.close()
        return jsonify({"status": "failed", "message": "Subcategory already exists"}), 400

    # Update value
    index = normalized_subcats.index(old_clean)
    current_subcats[index] = new_clean.strip()

    cursor.execute(
        "UPDATE categories SET subcategories = %s WHERE id = %s",
        (json.dumps(current_subcats), cat_id)
    )

    conn.commit()
    conn.close()

    return jsonify({
        "status": "success",
        "message": "Subcategory updated successfully",
        "subcategories": current_subcats
    }), 200

def delete_subcategory_controller(cat_id):
    """Delete a specific subcategory (does NOT delete blogs)"""
    data = request.get_json()
    author_name = data.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403
    
    subcategory_name = data.get("subcategory_name")
    
    if not subcategory_name:
        return jsonify({"status": "failed", "message": "Subcategory name is required"}), 400
    
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    cursor.execute("SELECT subcategories FROM categories WHERE id = %s", (cat_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return jsonify({"status": "failed", "message": "Category not found"}), 404
    
    current_subcats = []
    if result['subcategories']:
        try:
            current_subcats = json.loads(result['subcategories'])
        except:
            current_subcats = []
    
    if subcategory_name not in current_subcats:
        conn.close()
        return jsonify({"status": "failed", "message": "Subcategory not found"}), 404
    
    # Remove subcategory
    current_subcats.remove(subcategory_name)
    
    cursor.execute(
        "UPDATE categories SET subcategories = %s WHERE id = %s",
        (json.dumps(current_subcats) if current_subcats else None, cat_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success",
        "message": "Subcategory deleted",
        "subcategories": current_subcats
    }), 200

# ================= BLOG CONTROLLERS (WITH SUBCATEGORY FIELD ADDED) =================

def get_all_blogs_controller():
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = """
            SELECT b.*, c.name as category_name 
            FROM blogs b 
            LEFT JOIN categories c ON b.category_id = c.id 
            ORDER BY b.is_pinned DESC, b.created_at DESC
        """
        cursor.execute(query)
        blogs = cursor.fetchall()
        conn.close()
        
        for blog in blogs:
            blog['tags'] = parse_tags_output(blog.get('tags'))
            
            # Parse content_images JSON
            if blog.get('content_images'):
                try:
                    blog['content_images'] = json.loads(blog['content_images'])
                except:
                    blog['content_images'] = []
            else:
                blog['content_images'] = []
        
        return jsonify({"status": "success", "data": blogs}), 200
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

def create_blog_controller():
    author_name = request.form.get("author_name") 
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    title = request.form.get("title")
    content_preview = request.form.get("content_preview")
    category_id = request.form.get("category_id")
    subcategory = request.form.get("subcategory", "")  # NEW: Subcategory field (optional)
    is_pinned = request.form.get("is_pinned", 0)
    is_post = request.form.get("is_post", 0)
    
    tags_input = request.form.get("tags", "")
    tags = clean_tags_input(tags_input)

    # Main image
    image_file = request.files.get('image')
    image_db_path = None

    if image_file:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(file_path)
        image_db_path = f"{UPLOAD_FOLDER}/{filename}"

    # Content images (multiple)
    content_images = []
    content_image_files = request.files.getlist('content_images')
    
    for img_file in content_image_files:
        if img_file:
            filename = secure_filename(img_file.filename)
            file_path = os.path.join(CONTENT_IMAGES_FOLDER, filename)
            img_file.save(file_path)
            content_images.append(f"{CONTENT_IMAGES_FOLDER}/{filename}")
    
    if not content_images:
        content_images_input = request.form.get("content_images")
        if content_images_input:
            try:
                content_images = json.loads(content_images_input)
            except:
                content_images = [content_images_input]

    content_images_json = json.dumps(content_images) if content_images else None

    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO blogs (title, author_name, content_preview, category_id, subcategory, image_url, content_images, is_pinned, is_post, tags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (title, author_name, content_preview, category_id, subcategory, image_db_path, content_images_json, is_pinned, is_post, tags))
        blog_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success", 
            "message": "Blog posted successfully",
            "blog_id": blog_id
        }), 201
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

def update_blog_controller(blog_id):
    author_name = request.form.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    title           = request.form.get("title")
    content_preview = request.form.get("content_preview")
    category_id     = request.form.get("category_id")
    subcategory     = request.form.get("subcategory", "")   # FIX: subcategory add
    is_pinned       = request.form.get("is_pinned", 0)
    is_post         = request.form.get("is_post", 0)

    # FIX: tags add
    tags_input = request.form.get("tags", "")
    tags = clean_tags_input(tags_input)

    # FIX: Main image handle
    image_file    = request.files.get("image")
    image_db_path = None
    if image_file and image_file.filename:
        filename      = secure_filename(image_file.filename)
        file_path     = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(file_path)
        image_db_path = f"{UPLOAD_FOLDER}/{filename}"

    content_images      = []
    content_image_files = request.files.getlist("content_images")

    for f in content_image_files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            f.save(os.path.join(CONTENT_IMAGES_FOLDER, filename))
            content_images.append(f"{CONTENT_IMAGES_FOLDER}/{filename}")

    if not content_images:
        images_text = request.form.get("content_images")
        if images_text:
            try:
                content_images = json.loads(images_text)  
            except Exception:
                content_images = [images_text]

    try:
        conn   = get_connection()
        cursor = conn.cursor()

        if image_db_path:
            query = """
                UPDATE blogs
                SET title=%s, content_preview=%s, category_id=%s, subcategory=%s,
                    image_url=%s, content_images=%s,
                    is_pinned=%s, is_post=%s, tags=%s
                WHERE id=%s
            """
            params = (
                title, content_preview, category_id, subcategory,
                image_db_path, json.dumps(content_images),
                is_pinned, is_post, tags,
                blog_id
            )
        else:
            query = """
                UPDATE blogs
                SET title=%s, content_preview=%s, category_id=%s, subcategory=%s,
                    content_images=%s,
                    is_pinned=%s, is_post=%s, tags=%s
                WHERE id=%s
            """
            params = (
                title, content_preview, category_id, subcategory,
                json.dumps(content_images),
                is_pinned, is_post, tags,
                blog_id
            )

        cursor.execute(query, params)
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Blog updated successfully"}), 200

    except pymysql.err.IntegrityError:
        return jsonify({"status": "failed", "message": "Invalid Category ID. Please check if the category exists."}), 400
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

def delete_blog_controller(blog_id):
    author_name = request.args.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blogs WHERE id = %s", (blog_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Blog deleted successfully"}), 200

def get_published_blogs_controller():
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = """
            SELECT b.*, c.name as category_name 
            FROM blogs b 
            LEFT JOIN categories c ON b.category_id = c.id 
            WHERE b.is_post = 1
            ORDER BY b.is_pinned DESC, b.created_at DESC
        """
        cursor.execute(query)
        blogs = cursor.fetchall()
        conn.close()

        for blog in blogs:
            blog['tags'] = parse_tags_output(blog.get('tags'))
            
            # Parse content_images
            if blog.get('content_images'):
                try:
                    blog['content_images'] = json.loads(blog['content_images'])
                except:
                    blog['content_images'] = []
            else:
                blog['content_images'] = []

        return jsonify({
            "status": "success", 
            "count": len(blogs),
            "data": blogs
        }), 200

    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

def get_all_tags_controller():
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT DISTINCT tags FROM blogs WHERE tags IS NOT NULL AND tags != '' AND is_post = 1")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({
                "status": "success",
                "message": "No tags available",
                "count": 0,
                "data": []
            }), 200
        
        unique_tags = set()
        for row in rows:
            tags_str = row.get('tags', '')
            if not tags_str:
                continue
            
            tag_list = parse_tags_output(tags_str)
            
            for tag in tag_list:
                tag_clean = tag.strip().lower()
                if tag_clean and tag_clean != '[]' and not tag_clean.startswith('['):
                    unique_tags.add(tag_clean)
        
        if not unique_tags:
            return jsonify({
                "status": "success",
                "message": "No valid tags available",
                "count": 0,
                "data": []
            }), 200
        
        formatted_tags = [
            {
                "tag": tag,
                "slug": tag.replace(' ', '-').replace('_', '-')
            } 
            for tag in sorted(unique_tags)
        ]

        return jsonify({
            "status": "success",
            "count": len(formatted_tags),
            "data": formatted_tags
        }), 200

    except Exception as e:
        logging.error(f"Error in get_all_tags_controller: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Failed to fetch tags",
            "count": 0,
            "data": []
        }), 500

def get_filtered_blogs_controller():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "failed", "message": "Request body is required"}), 400
        
        category_name = data.get("category_name", "").strip()
        tags_input = data.get("tags", [])
        
        # Tags processing
        if isinstance(tags_input, str):
            tags_list = [tag.strip().lower() for tag in tags_input.split(',') if tag.strip()]
        elif isinstance(tags_input, list):
            tags_list = [str(tag).strip().lower() for tag in tags_input if str(tag).strip()]
        else:
            tags_list = []
            
        if not category_name and not tags_list:
            return jsonify({"status": "failed", "message": "At least one filter is required"}), 400

        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # SQL parts builder
        conditions = []
        params = []
        matched_by = []

        # 1. Category Condition
        if category_name:
            conditions.append("c.name = %s")
            params.append(category_name)
            matched_by.append("category")

        # 2. Tags Condition (Using OR for each tag)
        if tags_list:
            tag_query_parts = []
            for tag in tags_list:
                tag_query_parts.append("(b.tags LIKE %s OR b.tags LIKE %s OR b.tags LIKE %s OR b.tags = %s)")
                params.extend([f"{tag},%", f"%,{tag},%", f"%,{tag}", tag])
            
            # Combine all tags with OR
            conditions.append(f"({' OR '.join(tag_query_parts)})")
            matched_by.append("tags")

        # Dynamic Query Building (OR condition between Category and Tags)
        query = f"""
            SELECT DISTINCT b.*, c.name as category_name 
            FROM blogs b 
            LEFT JOIN categories c ON b.category_id = c.id 
            WHERE b.is_post = 1 
            AND ({' OR '.join(conditions)})
            ORDER BY b.is_pinned DESC, b.created_at DESC
        """

        cursor.execute(query, params)
        blogs = cursor.fetchall()
        conn.close()

        # Data Cleaning and Parsing
        unique_blogs = []
        seen_ids = set()
        
        for blog in blogs:
            if blog['id'] not in seen_ids:
                seen_ids.add(blog['id'])
                blog['tags'] = parse_tags_output(blog.get('tags'))
                
                # Parse content_images
                if blog.get('content_images'):
                    try:
                        blog['content_images'] = json.loads(blog['content_images'])
                    except:
                        blog['content_images'] = []
                else:
                    blog['content_images'] = []
                
                unique_blogs.append(blog)

        return jsonify({
            "status": "success",
            "filters": {"category_name": category_name or None, "tags": tags_list},
            "matched_by": matched_by,
            "count": len(unique_blogs),
            "data": unique_blogs
        }), 200

    except Exception as e:
        logging.error(f"Filter Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def get_all_subcategories_list_controller():
    """
    Database-er sob category-r subcategories eksathe fetch korar API
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # Sudhu category name ebong subcategories column fetch kora
        query = "SELECT name, subcategories FROM categories ORDER BY name ASC"
        cursor.execute(query)
        categories = cursor.fetchall()
        conn.close()

        result_list = []

        for cat in categories:
            # Subcategories string ke JSON list-e convert kora
            sub_list = []
            if cat['subcategories']:
                try:
                    sub_list = json.loads(cat['subcategories'])
                    # Jodi nested list thake (vuler karone), seta flatten kora
                    if sub_list and isinstance(sub_list[0], list):
                        sub_list = [item for sub in sub_list for item in (sub if isinstance(sub, list) else [sub])]
                except:
                    sub_list = []
            
            # Response object toiri
            result_list.append({
                "category_name": cat['name'],
                "subcategories": sub_list,
                "total_subcategories": len(sub_list)
            })

        return jsonify({
            "status": "success",
            "count": len(result_list),
            "data": result_list
        }), 200

    except Exception as e:
        logging.error(f"Error in get_all_subcategories: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= CONTENT IMAGES LIBRARY CONTROLLERS (NEW) =================

def upload_content_image_controller():
    """
    Upload content image to library
    POST multipart/form-data
    Fields: author_name, image (file)
    Returns: Full URL of uploaded image with ID
    """
    author_name = request.form.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403
    
    image_file = request.files.get('image')
    
    if not image_file:
        return jsonify({"status": "failed", "message": "Image file is required"}), 400
    
    try:
        # Save image to content folder
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(CONTENT_IMAGES_FOLDER, filename)
        image_file.save(file_path)
        
        # Create relative path for database
        image_relative_path = f"{CONTENT_IMAGES_FOLDER}/{filename}"
        
        # Insert into database
        conn = get_connection()
        cursor = conn.cursor()
        query = "INSERT INTO content_images_library (author_name, image_url) VALUES (%s, %s)"
        cursor.execute(query, (author_name, image_relative_path))
        conn.commit()
        
        image_id = cursor.lastrowid
        conn.close()
        
        # Build full URL (change to your server URL)
        # Example: http://122.163.121.176:3004/static/blog_images/content/image.jpg
        full_url = f"{BASE_URL.rstrip('/')}/{image_relative_path}"
        
        return jsonify({
            "status": "success",
            "message": "Image uploaded successfully",
            "data": {
                "id": image_id,
                "author_name": author_name,
                "image_url": full_url,
                "image_path": image_relative_path
            }
        }), 201
        
    except Exception as e:
        logging.error(f"Error uploading image: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500


def get_content_images_controller():
    """
    Get all content images or filter by author_name
    GET /api/content-images?author_name=Admin (optional)
    """
    try:
        author_name = request.args.get("author_name")
        
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        if author_name:
            # Filter by author
            query = "SELECT * FROM content_images_library WHERE author_name = %s ORDER BY created_at DESC"
            cursor.execute(query, (author_name,))
        else:
            # Get all
            query = "SELECT * FROM content_images_library ORDER BY created_at DESC"
            cursor.execute(query)
        
        images = cursor.fetchall()
        conn.close()
        
        # Convert to full URLs
        for img in images:
            # Convert relative path to full URL
            full_url = f"{BASE_URL.rstrip('/')}/{img['image_url']}"
            img['image_url'] = full_url
        
        return jsonify({
            "status": "success",
            "count": len(images),
            "data": images
        }), 200
        
    except Exception as e:
        logging.error(f"Error fetching images: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500


def update_content_image_controller(image_id):
    """
    Update content image (replace image or update author)
    PUT /api/content-images/<id>
    multipart/form-data: author_name, image (optional)
    """
    author_name = request.form.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403
    
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # Check if image exists
        cursor.execute("SELECT * FROM content_images_library WHERE id = %s", (image_id,))
        existing = cursor.fetchone()
        
        if not existing:
            conn.close()
            return jsonify({"status": "failed", "message": "Image not found"}), 404
        
        image_file = request.files.get('image')
        
        if image_file:
            # Replace image
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(CONTENT_IMAGES_FOLDER, filename)
            image_file.save(file_path)
            
            image_relative_path = f"{CONTENT_IMAGES_FOLDER}/{filename}"
            
            # Update both author and image
            query = "UPDATE content_images_library SET author_name = %s, image_url = %s WHERE id = %s"
            cursor.execute(query, (author_name, image_relative_path, image_id))
        else:
            # Update only author name
            query = "UPDATE content_images_library SET author_name = %s WHERE id = %s"
            cursor.execute(query, (author_name, image_id))
        
        conn.commit()
        
        # Fetch updated record
        cursor.execute("SELECT * FROM content_images_library WHERE id = %s", (image_id,))
        updated = cursor.fetchone()
        conn.close()
        
        # Convert to full URL
        full_url = f"{BASE_URL.rstrip('/')}/{image_relative_path.lstrip('/')}"
        updated['image_url'] = full_url
        
        return jsonify({
            "status": "success",
            "message": "Image updated successfully",
            "data": updated
        }), 200
        
    except Exception as e:
        logging.error(f"Error updating image: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500


def delete_content_image_controller(image_id):
    """
    Delete content image from library
    DELETE /api/content-images/<id>?author_name=Admin
    """
    author_name = request.args.get("author_name")
    
    is_allowed, msg = check_admin_access(author_name)
    if not is_allowed:
        return jsonify({"status": "failed", "message": msg}), 403
    
    try:
        conn = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # Check if image exists
        cursor.execute("SELECT * FROM content_images_library WHERE id = %s", (image_id,))
        existing = cursor.fetchone()
        
        if not existing:
            conn.close()
            return jsonify({"status": "failed", "message": "Image not found"}), 404
        
        # Delete from database
        cursor.execute("DELETE FROM content_images_library WHERE id = %s", (image_id,))
        conn.commit()
        conn.close()
        
        # Optionally delete physical file
        try:
            file_path = existing['image_url']
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.warning(f"Could not delete physical file: {str(e)}")
        
        return jsonify({
            "status": "success",
            "message": "Image deleted successfully"
        }), 200
        
    except Exception as e:
        logging.error(f"Error deleting image: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500

        