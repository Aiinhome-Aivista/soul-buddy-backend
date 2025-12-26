import re
import io
import uuid
import json
import pypdf
import logging
import chromadb
import itertools
import numpy as np
import pandas as pd
import mysql.connector
from rapidfuzz import fuzz
from flask import request, jsonify
from sqlalchemy import create_engine
from model.llm_client import call_llm
from database.config import ACTIVE_LLM, MYSQL_CONFIG
from sentence_transformers import SentenceTransformer

# Logging
logging.basicConfig(level=logging.INFO)

# Build SQLAlchemy URL
MYSQL_URL = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password'].replace('@', '%40')}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)
engine = create_engine(MYSQL_URL, pool_pre_ping=True)

sessions = {}

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------ ChromaDB Init ------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# ------------------------ Helpers ------------------------
def detect_relationships(dfs):
    relationships = []
    for (t1, df1), (t2, df2) in itertools.combinations(dfs.items(), 2):
        for c1 in df1.columns:
            for c2 in df2.columns:
                sim = fuzz.ratio(c1.lower(), c2.lower())
                if sim > 75:
                    relationships.append({
                        "table1": t1, "column1": c1,
                        "table2": t2, "column2": c2,
                        "similarity": sim
                    })
    return relationships

def store_df_mysql(df, table_name):
    safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)
    df.to_sql(safe_name, con=engine, if_exists='replace', index=False)
    return safe_name

def extract_json_block(text: str) -> dict | None:
    match_fence = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match_fence:
        try:
            return json.loads(match_fence.group(1))
        except:
            pass
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None

def call_llm_unified(prompt: str) -> str:
    try:
        response = call_llm(prompt)
        return str(response).strip()
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return f"[LLM API Error] {str(e)}"

# ------------------------ ChromaDB Vector Store ------------------------
def push_to_vector_db(uploaded_files, nodes, edges, session_id):
    collection = chroma_client.get_or_create_collection(
        name=f"session_{session_id}",
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    embeddings = []
    ids = []

    # 1. File names
    for f in uploaded_files:
        text = f.filename
        emb = embedding_model.encode(text).tolist()
        documents.append(text)
        embeddings.append(emb)
        ids.append(str(uuid.uuid4()))

    # 2. Node schemas
    for n in nodes:
        cols = n['props'].get('columns', [])
        text = f"Node {n['label']} — columns: {', '.join(cols)}"
        emb = embedding_model.encode(text).tolist()
        documents.append(text)
        embeddings.append(emb)
        ids.append(str(uuid.uuid4()))

    # 3. Edge relationships
    for e in edges:
        if not all(k in e for k in ("table1", "column1", "table2", "column2")):
            continue
        text = (
            f"Edge {e['table1']}.{e['column1']} ≈ "
            f"{e['table2']}.{e['column2']} "
            f"(similarity: {e.get('similarity', 0)})"
        )
        emb = embedding_model.encode(text).tolist()
        documents.append(text)
        embeddings.append(emb)
        ids.append(str(uuid.uuid4()))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )

    # Persistence for older Chroma versions
    try:
        chroma_client.persist()
    except AttributeError:
        pass

# ------------------------ Insights ------------------------
def generate_insights(session, more=False):
    schema_context = ""
    for t in session["tables"]:
        try:
            df = pd.read_sql(f"SELECT * FROM `{t}` LIMIT 5", engine)
            schema_context += f"\nTable {t} (sample 5 rows):\n{df.to_markdown(index=False)}\n"
        except:
            continue

    rel_context = "\n".join([
        f"{r['table1']}.{r['column1']} ≈ {r['table2']}.{r['column2']}"
        for r in session["relationships"]
    ])

    insight_type = "more diverse and deeper" if more else "basic and useful"

    prompt = f"""
You are a data analysis assistant.

Tables:
{schema_context}

Relationships:
{rel_context}

Generate {insight_type} insights. Bullet points only.
"""
    return call_llm_unified(prompt)

# ------------------------ Upload Controller (Admin Only Logic) ------------------------
def upload_files_controller():
    # Note: In a real app, you would add an authentication check here 
    # to ensure only 'admin' can hit this endpoint.
    
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    session_name = request.form.get("session_name") or request.json.get("session_name")
    if not session_name:
        return jsonify({"error": "Session name is required"}), 400

    session_id = str(uuid.uuid4())
    dfs = {}

    for file in files:
        name = file.filename.rsplit('.', 1)[0]
        content = file.read()
        df = None 

        file_extension = file.filename.split('.')[-1].lower()

        if file_extension == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif file_extension == "xlsx":
            df = pd.read_excel(io.BytesIO(content))
        elif file_extension == "json":
            df = pd.read_json(io.BytesIO(content))
        elif file_extension == "pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(content))
                full_text = []
                for page in reader.pages:
                    full_text.append(page.extract_text() or "")
                document_text = "\n".join(full_text)
                if document_text.strip():
                    df = pd.DataFrame([{"document_text": document_text}])
                else:
                    return jsonify({"error": f"PDF file {file.filename} empty."}), 400
            except Exception as e:
                return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 400
        else:
            return jsonify({"error": f"Unsupported file: {file.filename}"}), 400

        # Store to MySQL
        table_name = f"{name}_{uuid.uuid4().hex[:8]}"
        safe_table_name = store_df_mysql(df, table_name)
        dfs[safe_table_name] = df

    relationships = detect_relationships(dfs)

    sessions[session_id] = {
        "tables": list(dfs.keys()),
        "relationships": relationships,
        "chat_history": [],
        "last_insights": None
    }

    # Save session metadata
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        sql = """
        INSERT INTO session_tracking (session_name, session_id, tables, relationships)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE tables=VALUES(tables), relationships=VALUES(relationships)
        """
        cursor.execute(sql, (session_name, session_id,
                             json.dumps(list(dfs.keys())),
                             json.dumps(relationships)))
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass

    # Push to ChromaDB
    push_to_vector_db(
        uploaded_files=files,
        nodes=[{"label": t, "props": {"columns": list(df.columns)}} for t, df in dfs.items()],
        edges=relationships,
        session_id=session_id
    )

    return jsonify({
        "session_id": session_id,
        "tables": list(dfs.keys()),
        "relationships": relationships
    })

def is_message_safe(user_message: str) -> bool:
    prompt = f"""
You are a safety classifier.
Evaluate the user's message and classify it as either: SAFE or UNSAFE.

Return ONLY ONE WORD: SAFE or UNSAFE.
User message: "{user_message}"
"""
    result = call_llm_unified(prompt).strip().upper()
    return result == "SAFE"

# ------------------------ NEW USER CONTEXT HELPERS ------------------------
def get_user_profile(user_id):
    """
    Fetches the user's direct profile from the 'users' table AND 
    their survey responses to build a complete personality profile.
    """
    if not user_id:
        return "Unknown User Profile"
    
    profile_summary = "=== USER PERSONALIZATION DATA ===\n"
    
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 1. Fetch Basic Info from 'users' table
        user_query = """
            SELECT full_name, age, gender, work, health, emotional_state, relationship 
            FROM users 
            WHERE user_id = %s
        """
        cursor.execute(user_query, (user_id,))
        user_row = cursor.fetchone()
        
        if user_row:
            profile_summary += "--- Basic Information ---\n"
            profile_summary += f"Name: {user_row.get('full_name', 'N/A')}\n"
            profile_summary += f"Age: {user_row.get('age', 'N/A')}\n"
            profile_summary += f"Work/Job: {user_row.get('work', 'N/A')}\n"
            profile_summary += f"Current Emotional State: {user_row.get('emotional_state', 'N/A')}\n"
            profile_summary += f"Relationship Status: {user_row.get('relationship', 'N/A')}\n"
            profile_summary += f"Health Perception: {user_row.get('health', 'N/A')}\n"
        else:
            profile_summary += "Basic Info: User ID not found in 'users' table.\n"

        # 2. Fetch Detailed Survey Answers (from 'user_responses')
        survey_query = """
            SELECT q.question_text, ur.answer_value 
            FROM user_responses ur
            JOIN ai_soulbuddy.questions q ON ur.question_id = q.question_id
            WHERE ur.user_id = %s
        """
        cursor.execute(survey_query, (user_id,))
        survey_rows = cursor.fetchall()
        
        if survey_rows:
            profile_summary += "\n--- Detailed Survey Responses ---\n"
            for row in survey_rows:
                profile_summary += f"- {row['question_text']}: {row['answer_value']}\n"
            
        cursor.close()
        conn.close()
        return profile_summary

    except Exception as e:
        logging.error(f"Error fetching user profile: {e}")
        return "Error fetching profile."

def get_long_term_history(user_id, limit=5):
    """
    Fetches past conversations from the permanent history table.
    """
    if not user_id:
        return ""
        
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Assuming table has columns: user_input, model_response, created_at
        # Use reversed order to show oldest -> newest
        query = """
            SELECT user_input, model_response 
            FROM conversation_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """
        cursor.execute(query, (user_id, limit))
        rows = cursor.fetchall()
        
        if not rows:
            return ""
            
        history_text = "Previous Conversation History (Context):\n"
        for row in reversed(rows):
            history_text += f"User: {row['user_input']}\nAI: {row['model_response']}\n---\n"
            
        cursor.close()
        conn.close()
        return history_text
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return ""

# ------------------------ RAG Chat Controller ------------------------
def rag_chat_controller():
    data = request.get_json()
    session_id = data.get("session_id")
    user_id = data.get("user_id")  # Expect user_id from frontend now
    user_message = data.get("query", "").strip()

    # 1. Safety Gate
    if not is_message_safe(user_message):
        return jsonify({"answer": "Sorry, I can't help with that topic."}), 200

    if not session_id:
        return jsonify({"error": "Session ID missing"}), 400

    # 2. Fetch User Context (Profile + History)
    user_profile_context = get_user_profile(user_id)
    user_history_context = get_long_term_history(user_id)
    
    full_user_context = f"""
    {user_profile_context}
    
    {user_history_context}
    """

    # 3. Load Session (Restore from MySQL if not in memory)
    if session_id not in sessions:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM session_tracking WHERE session_id=%s", (session_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return jsonify({"error": "Session not found"}), 404

        sessions[session_id] = {
            "tables": json.loads(row["tables"]),
            "relationships": json.loads(row["relationships"]),
            "chat_history": []
        }

    session = sessions[session_id]
    
    if not user_message:
        return jsonify({"answer": generate_insights(session)})

    # ============================================================
    # LOGIC: VectorDB -> (If Hit) Enhanced Answer -> (If Miss) General Answer
    # ============================================================

    # A. Search VectorDB
    collection = chroma_client.get_or_create_collection(name=f"session_{session_id}")
    query_emb = embedding_model.encode(user_message).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=5)

    RELEVANCE_THRESHOLD = 0.55
    is_relevant = False
    context_text = ""
    context_docs = []

    if results.get("documents") and results.get("distances"):
        top_distance = results["distances"][0][0]
        logging.info(f"[VECTOR_DB] Top distance: {top_distance}")

        if top_distance < RELEVANCE_THRESHOLD:
            is_relevant = True
            context_docs = results["documents"][0]
            context_text = "\n".join(context_docs)
        else:
            print("[VECTOR_DB] No relevant answer found")
    else:
        print("[VECTOR_DB] No results returned from Vector DB")


    # B. Execute Logic Paths
    if is_relevant:
        # --- PATH 1: Answer from Dataset (Personalized) ---
        
        # Prepare Schema info in case it's a SQL question
        schema_context = ""
        for t in session["tables"]:
            try:
                # Lightweight fetch for columns
                df = pd.read_sql(f"SELECT * FROM `{t}` LIMIT 0", engine)
                schema_context += f"Table {t}: {', '.join(df.columns)}\n"
            except:
                pass

        rel_context = "\n".join([f"{r['table1']}.{r['column1']} ≈ {r['table2']}.{r['column2']}" for r in session["relationships"]])

        llm_prompt = f"""
You are "SoulBuddy", an intelligent and empathetic AI assistant.

{full_user_context}

CONTEXT FROM UPLOADED DATASETS:
{context_text}

AVAILABLE TABLES (Schema):
{schema_context}

RELATIONSHIPS:
{rel_context}

USER QUESTION: {user_message}

INSTRUCTIONS:
1. Answer the question using the Dataset Context.
2. PERSONALIZATION: Tailor your tone and advice based on the "User Profile Traits" above (e.g., if they are tired, be gentle; if distracted, be concise).
3. REFERENCE: If the User's History shows they asked about this before, acknowledge it.
4. If the user asks for specific data rows/stats, generate a SQL query inside a JSON block: {{"tool": "run_sql_query", "query": "SELECT ..."}}
5. If you use the context, provide an ENHANCED, professional answer.

Respond with the answer or the JSON block.
"""
        llm_response = call_llm_unified(llm_prompt)
        tool_call = extract_json_block(llm_response)

        sql_query = None
        final_answer = ""

        if tool_call and tool_call.get("tool") == "run_sql_query":
            sql_query = tool_call.get("query")
            try:
                df = pd.read_sql(sql_query, engine)
                if df.empty:
                    final_answer = "I checked the uploaded datasets, but found no matching records."
                else:
                    records = df.head(5).to_dict(orient="records")
                    # Enhance the SQL result
                    enhance_prompt = f"""
The user asked: "{user_message}"
SQL Results: {records}

User Profile: {user_profile_context}

Please provide a natural, enhanced answer summarizing these results, matching the user's vibe/profile.
"""
                    final_answer = call_llm_unified(enhance_prompt)
            except Exception as e:
                final_answer = f"Error querying dataset: {str(e)}"
        else:
            final_answer = llm_response.strip()

        session["chat_history"].append({"user": user_message, "ai": final_answer})
        
        # Log to permanent history
        if user_id:
            try:
                conn = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = conn.cursor()
                sql_log = "INSERT INTO conversation_history (user_id, user_input, model_response) VALUES (%s, %s, %s)"
                cursor.execute(sql_log, (user_id, user_message, final_answer))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                logging.error(f"Failed to log to permanent DB: {e}")

        return jsonify({
            "answer": final_answer,
            "source": "dataset_enhanced",
            "context_used": context_docs
        })

    else:
        print("[LLM] Using Mistral API")

        mistral_prompt = f"""
You are "SoulBuddy", a helpful and empathetic AI assistant.

{full_user_context}

The user asked: "{user_message}"

We checked the uploaded datasets but found NO relevant information.
Please answer the user's question using your own general knowledge.
Do NOT mention the dataset check.
STRICTLY personalize the advice based on the User Profile above.
(e.g., If the user is young (18-24), use relatable examples. If they often procrastinate, give actionable, small steps).
"""
        final_answer = call_llm_unified(mistral_prompt).strip()

        session["chat_history"].append({
            "user": user_message,
            "ai": final_answer
        })

        # Log to permanent history
        if user_id:
            try:
                conn = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = conn.cursor()
                sql_log = "INSERT INTO conversation_history (user_id, user_input, model_response) VALUES (%s, %s, %s)"
                cursor.execute(sql_log, (user_id, user_message, final_answer))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                logging.error(f"Failed to log to permanent DB: {e}")

        return jsonify({
            "answer": final_answer,
            "source": "mistral_general",
            "context_used": []
        })