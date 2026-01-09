import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from neo4j import GraphDatabase, basic_auth
from pymongo import MongoClient

load_dotenv()

# Local Ollama API (for mistral_local)
MISTRAL_API_URL  = "http://localhost:11434/api/generate"

# API Keys
# GEMINI_API_KEY = "AIzaSyATdX-0CN2CMfJWkhVVju11ahdKmTwZxxM"
GEMINI_API_KEY = "AIzaSyC8vJObb0iCn0fAHXxK7DYxJaqj-DD7Bxo"
MISTRAL_API_KEY = "IotlgX9OC7gWRj0WqHuT5xdhT1LNkNne"
MISTRAL_MODEL="mistral-small-latest"

# Choose between: "gemini", "mistral_cloud", "mistral_local"
ACTIVE_LLM = "mistral_cloud"
MODEL_NAME = "gemini-2.5-flash"  # for Gemini


# ============ Gemini Configuration ============
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "IotlgX9OC7gWRj0WqHuT5xdhT1LNkNne")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC8vJObb0iCn0fAHXxK7DYxJaqj-DD7Bxo")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# ============ Folder Configuration ============
GRAPH_FOLDER = os.getenv("GRAPH_FOLDER", "graphs")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
TEMP_UPLOAD_FOLDER = os.getenv("TEMP_UPLOAD_FOLDER", "uploads")

# ============ MySQL Configuration ============
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "116.193.134.6"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "lmysqluser"),
    "password": os.getenv("MYSQL_PASSWORD", "lenovo@429"),
    # "database": os.getenv("MYSQL_DATABASE", "NEW_DPT_V2")
    "database": os.getenv("MYSQL_DATABASE", "ai_soulbuddy")
}
# --- SQLAlchemy Engine ---
MYSQL_URI = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{quote_plus(MYSQL_CONFIG['password'])}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
)

engine = create_engine(MYSQL_URI, pool_recycle=3600, pool_pre_ping=True)


# ============ Base URL ============
BASE_URL = os.getenv("BASE_URL", "https://aivista.co.in")

# ============ Misc Settings ============
MAX_SAMPLE_VALUES = int(os.getenv("MAX_SAMPLE_VALUES", "100"))

#=======================NEO4J_URI==================
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://287472f2.databases.neo4j.io")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASS = os.getenv("NEO4J_PASS", "Rnpyic2loh-N10dJKWZdnaloP3AJiYPjb7HI1eODTfs")

ARANGO_HOST = os.getenv("ARANGO_HOST", "https://eaffa1ddb656.arangodb.cloud:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "t8aNkAzuMmoI0Ew8xx5i")
ARANGO_DB = os.getenv("ARANGO_DB", "graph_ai")


# ============ MongoDB ATLAS Configuration (Direct URI) ============

# *** WARNING: THIS STRING SHOULD ONLY BE A TEMPLATE OR A NON-PRODUCTION DEFAULT ***
# The MONGO_ATLAS_URI should ideally be provided via environment variables 
# to keep credentials secure.

# Default values for user, password, cluster, and database name
# MONGO_USER_DEFAULT = "5ynt4x3rr0rss_db_user"
# MONGO_PASS_DEFAULT = "GuDlppCH7zBDAUFi"
# MONGO_CLUSTER_DEFAULT = "@cluster0.gcg8xdx.mongodb.net"
MONGO_DATABASE_DEFAULT = "Soulbuddy_V2"

MONGO_ATLAS_URI ="mongodb+srv://5ynt4x3rr0rss_db_user:GuDlppCH7zBDAUFi@cluster0.gcg8xdx.mongodb.net/Soulbuddy_V2?retryWrites=true&w=majority"

# The connection URI is the variable itself
MONGO_URI = MONGO_ATLAS_URI
MONGO_DATABASE_NAME = MONGO_DATABASE_DEFAULT
# try:
#     driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))
#     with driver.session() as session:
#         result = session.run("RETURN 'Connected!' AS message")
#         print(result.single()["message"])
# except Exception as e:
#     print(f" Connection failed: {e}")

# --- gmail Configuration ---
GMAIL_USER = 'saikatofficial1998@gmail.com'
GMAIL_APP_PASSWORD = 'gkzlglukauqwflnd' 
RECEIVER_EMAIL = 'saikatofficial1998@gmail.com' 