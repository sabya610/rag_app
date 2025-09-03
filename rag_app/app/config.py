import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import socket

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print("The Base dir path")

class Config:

    PDF_FOLDER = os.path.join(BASE_DIR, "pdf_kb_files")
    PGVECTOR_DIM = 384
    MAX_RESULTS = 50
    ALLOWED_EXTENSIONS = {'pdf'}

    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASS = os.getenv('DB_PASS', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'ragdb')


    # Validate hostname; fallback to localhost if not resolvable
    try:
        socket.gethostbyname(DB_HOST)
    except socket.error:
        DB_HOST = "localhost"


    DB_URI = os.getenv(
        "DB_URI",
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    
    # Use local offline embedding model path
    _embedding_model_env = os.getenv("EMBEDDING_MODEL")
    _embedding_model_default = os.path.join(
        BASE_DIR, "models", "embedding", "all-MiniLM-L6-v2"
    )

    if _embedding_model_env and os.path.exists(_embedding_model_env):
        EMBEDDING_MODEL = _embedding_model_env
    else:
        EMBEDDING_MODEL = _embedding_model_default

    _model_path_env = os.getenv("MODEL_PATH")
    _model_path_default = os.path.join(
        BASE_DIR, "..", "..","models", "llama-2-7b-chat.Q4_K_M.gguf"
    )

    # Path to local llama model file
    #MODEL_PATH = os.getenv(
    #    "MODEL_PATH",
    #    os.path.join(BASE_DIR, "..", "models", "llama-2-7b-chat.Q4_K_M.gguf")
    #)
    
    if _model_path_env and os.path.exists(_model_path_env):
        MODEL_PATH = _model_path_env
    else:
        MODEL_PATH = _model_path_default

    