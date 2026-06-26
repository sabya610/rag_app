import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Use local offline embedding model path
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        os.path.join(BASE_DIR, "..", "models", "embedding", "all-MiniLM-L6-v2")
    )

    PDF_FOLDER = os.path.join(BASE_DIR, "pdf_kb_files")
    PGVECTOR_DIM = 384
    MAX_RESULTS = 50
    ALLOWED_EXTENSIONS = {'pdf'}

    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASS = os.getenv('DB_PASS', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'ragdb')

    DB_URI = os.getenv(
        "DB_URI",
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Path to local llama model file
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "..", "models", "llama-2-7b-chat.Q4_K_M.gguf")
    )

    # Slack Integration
    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN', '')
    SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET', '')
    SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN', '')
    
    # Slack import configuration
    SLACK_IMPORT_LIMIT = int(os.getenv('SLACK_IMPORT_LIMIT', '100'))  # Messages per channel
    SLACK_IMPORT_DAYS = int(os.getenv('SLACK_IMPORT_DAYS', '30'))  # Days back to import
