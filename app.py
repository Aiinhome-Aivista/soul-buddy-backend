import os
import mysql.connector # Needed for direct connection
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from controllers.faqs import fetch_grouped_faqs
from controllers.tracker import get_tracker_data
from concurrent.futures import ThreadPoolExecutor
from controllers.insights import insights_controller
from werkzeug.security import generate_password_hash # Needed for signup
from controllers.submit_response import submit_response
from controllers.analyze_files import analyze_controller
from controllers.user_profile import get_users_controller
from controllers.view_info import view_analyze_controller
from controllers.login_controller import login_controller
from controllers.fetch_disclaimer import fetch_disclaimer
from controllers.signup_controller import signup_controller
from controllers.delete_history import delete_session_controller
from controllers.session_controller import fetch_successful_sessions
from controllers.subscription_controller import subscription_controller
from controllers.terms_and_conditions import fetch_terms_and_conditions
from controllers.user_input_analyze import user_input_analyze_controller
from controllers.wellbeing_profile import get_wellbeing_profile_controller
from controllers.chat import upload_files_controller , rag_chat_controller
from controllers.get_analyze_summary import get_analyze_summary_controller
from controllers.voice_assistant_app import handle_voice_ask, serve_audio_file
from controllers.uload_file_count_tablename import upload_files_count_controller
from controllers.question_fetch_controller_db import question_fetch_controller_db
from controllers.wellbeing_recovery_controller import get_recovery_plan_controller
from controllers.visualization import send_from_directory, upload_and_process_arangodb
from controllers.active_session_controller import update_active_sessions_controller
from controllers.user_details_controller import get_user_details
from controllers.expert_controller import (
    list_users_with_sessions_controller,
    get_session_conversation_controller,
    save_expert_insight_controller
)
from controllers.user_sessions import get_user_sessions_controller,user_wellbeing_summary_controller
from database.config import BASE_URL, MYSQL_CONFIG, MAX_SAMPLE_VALUES, GRAPH_FOLDER, UPLOAD_FOLDER, TEMP_UPLOAD_FOLDER
# --- Database & LLM Services ---
from pyvis.network import Network
from database.llm_service import LLMService
from database.database_service import DatabaseService
from controllers.orchestrator_controller import process_books
from controllers.mail import handle_contact_controller
from controllers.email_login_controller import email_login_controller
from controllers.forgot_password_request_controller import request_password_reset_controller
from controllers.forgot_password_verify_controller import reset_password_with_otp_controller
load_dotenv()

app = Flask(__name__)

CORS(app)

# Create necessary folders if not exist
for folder in [GRAPH_FOLDER, UPLOAD_FOLDER, TEMP_UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)


MYSQL_URI = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{quote_plus(MYSQL_CONFIG['password'])}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

engine = create_engine(MYSQL_URI, pool_recycle=3600, pool_pre_ping=True)

# Helper function to create a new mysql.connector connection for the controllers
def get_db_connection():
    """Establishes and returns a raw mysql.connector connection."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database']
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL DB: {err}")
        return None
# ------------------- Flask App -------------------
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload


UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")  
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

  
# Global thread pool
executor = ThreadPoolExecutor(max_workers=8)  # tune based on CPU cores


# ---------------------------------- API Endpoints ---------------------------------

# Insight
@app.route("/insight", methods=["POST"])
def insights():
    return insights_controller()

# Tracker
@app.route('/tracker', methods=['GET'])
def get_tracker_data_():
    return get_tracker_data()

# Analyze Files
@app.route("/analyze_files", methods=["POST"])
def analyze():
    return analyze_controller()

# Upload
# @app.route("/upload", methods=["POST"])
# def upload_process():
#     return upload_and_process_arangodb()

@app.route('/upload', methods=['POST'])
def process_books_route():
    return process_books()

# # Chat
# @app.route("/chat", methods=["POST"])
# def chat():
#     return chat_controller()

# RAG Chat (vector database)
@app.route("/rag_chat", methods=["POST"])
def rag_chat_route():
    return rag_chat_controller()

# View Info
@app.route("/view_info", methods=["GET"])
def view_analyze():
    return view_analyze_controller() 


@app.route("/graphs/<filename>")
def send_graph(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

 

@app.route("/upload_files", methods=["POST"])
def upload_files():
    return upload_files_controller()


@app.route("/uploadfile_details", methods=["POST"])
def get_analyze_summary_():
    return get_analyze_summary_controller()

@app.route("/upload_files_count", methods=["POST"])
def upload_files_count():
    return upload_files_count_controller()

@app.route("/delete_session/<string:session_name>", methods=["DELETE"])
def delete_session(session_name):
    return delete_session_controller(session_name)

## User Input analyze
@app.route("/user_input_analyze", methods=["POST"])
def user_input_analyze():
    return user_input_analyze_controller()

# --- NEW ROUTES: SIGNUP AND SUBSCRIPTION ---
@app.route('/signup', methods=['POST'])
def signup_route():
    # Pass the helper function to the controller
    return signup_controller(get_db_connection)

@app.route('/subscribe', methods=['POST'])
def subscribe_route():
    # Pass the helper function to the controller
    return subscription_controller(get_db_connection)
 
# --- NEW ROUTE: LOGIN ---

@app.route('/login', methods=['POST'])
def login_route():
    # Pass the original pymysql connection function (get_connection) to the controller
    return login_controller(get_db_connection)

# --- NEW ROUTE: FETCH QUESTIONS FROM DB ---
@app.route("/fetch_questions", methods=["GET"])
def fetch_questions():
    """Endpoint to fetch the list of onboarding questions from the database."""
    # Use the get_connection function that connects to the database with the 'questions' table
    return question_fetch_controller_db(get_db_connection)

# --- ROUTE 3: SUBMIT RESPONSE ---
@app.route('/submit_response', methods=['POST'])
def submit_response_route():
    return submit_response(get_db_connection) 

# profiling routes
@app.route('/wellbeing-profile', methods=['POST'])
def wellbeing_profile():
    return get_wellbeing_profile_controller()


@app.route('/wellbeing-recovery-plan', methods=['POST'])
def wellbeing_recovery_plan():
    return get_recovery_plan_controller()
 

@app.route("/voice-ask", methods=["POST"])
def voice_ask():
    return handle_voice_ask()

@app.route("/audio/<filename>")
def serve_audio(filename):
    return serve_audio_file(filename)

# 🆕 NEW ROUTE HERE
@app.route('/successful_sessions', methods=['GET'])
def get_successful_sessions_route():
    return fetch_successful_sessions()


@app.route('/disclaimer', methods=['GET'])
def disclaimer():
    return fetch_disclaimer()


@app.route('/terms-and-conditions', methods=['GET'])
def terms_and_conditions():
    return fetch_terms_and_conditions()


@app.route('/faqs', methods=['GET'])
def faqs():
    return fetch_grouped_faqs()

@app.route('/update_active_sessions', methods=['POST'])
def update_active_sessions_route():
    return update_active_sessions_controller()

@app.route('/users', methods=['GET'])
def fetch_users():
    return get_users_controller()

# =======================
# EXPERT / ADMIN ROUTES
# =======================

@app.route("/sessions", methods=["GET"])
def admin_users_sessions():
    return list_users_with_sessions_controller()


@app.route("/session/<session_id>", methods=["GET"])
def admin_session_conversation(session_id):
    return get_session_conversation_controller(session_id)


@app.route("/expert-insight", methods=["POST"])
def admin_expert_insight():
    return save_expert_insight_controller()


@app.route('/user_details', methods=['POST'])
def user_details():
    return get_user_details()


@app.route("/user-sessions/<user_id>", methods=["GET"])
def get_user_sessions(user_id):
    return get_user_sessions_controller(user_id)

@app.route("/user-summary/<user_id>", methods=["GET"])
def user_wellbeing_summary(user_id):
    return user_wellbeing_summary_controller(user_id)

@app.route('/contact', methods=['POST'])
def handle_contact():
    return handle_contact_controller()

@app.route('/login_email', methods=['POST'])
def login_email_route():
    return email_login_controller(get_db_connection)

# @app.route('/forgot_password', methods=['POST'])
# def forgot_password_route():
#     return forgot_password_controller(get_db_connection)

# Step 1: Request OTP
@app.route('/forgot_password_request', methods=['POST'])
def request_reset_route():
    return request_password_reset_controller(get_db_connection)

# Step 2: Verify OTP & Change Password
# Step 2: Verify OTP & Change Password
@app.route('/forgot_password_reset', methods=['POST'])
def reset_password_route():
    return reset_password_with_otp_controller(get_db_connection)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3004, debug=True)