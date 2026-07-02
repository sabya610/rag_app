import os  # noqa: prodline_v3
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load .env.local first if it exists (for Windows dev), then .env as fallback
_base = Path(__file__).resolve().parent.parent
if (_base / ".env.local").is_file():
    load_dotenv(_base / ".env.local")
load_dotenv()  # loads .env (won't override already-set vars)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Use local offline embedding model path
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        os.path.join(BASE_DIR, "..", "models", "embedding", "all-MiniLM-L6-v2")
    )

    PDF_FOLDER = os.path.join(BASE_DIR, "pdf_kb_files")
    PGVECTOR_DIM = 384
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
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
        os.path.join(BASE_DIR, "..", "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    )

    # ---- Salesforce (SFDC) Knowledge Article Integration ----
    # Auth Mode 1: Session ID (from browser DevTools cookie)
    SF_URL = os.getenv("SF_URL", "https://hp.my.salesforce.com")
    SF_SID = os.getenv("SF_SID", "")
    SF_SID_FILE = os.getenv("SF_SID_FILE", "")

    # Auth Mode 2: OAuth2 password flow
    SFDC_CLIENT_ID = os.getenv("SFDC_CLIENT_ID", "")
    SFDC_CLIENT_SECRET = os.getenv("SFDC_CLIENT_SECRET", "")
    SFDC_USERNAME = os.getenv("SFDC_USERNAME", "")
    SFDC_PASSWORD = os.getenv("SFDC_PASSWORD", "")
    SFDC_SECURITY_TOKEN = os.getenv("SFDC_SECURITY_TOKEN", "")
    SFDC_LOGIN_URL = os.getenv("SFDC_LOGIN_URL", "https://login.salesforce.com")

    # Feature toggle
    SFDC_ENABLED = os.getenv("SFDC_ENABLED", "true").lower() in ("1", "true", "yes")
    SFDC_SEARCH_LIMIT = int(os.getenv("SFDC_SEARCH_LIMIT", "10"))

    # Product filter — restrict KA queries to HPE Ezmeral articles only
    SFDC_PRODUCT_QUEUE = os.getenv("SFDC_PRODUCT_QUEUE", "HPE Ezmeral")
    SFDC_PRODUCT_LINE = os.getenv("SFDC_PRODUCT_LINE", "CONT PLT SW (RM)")

    # ---- Slack Bot Integration ----
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")          # xoxb-...
    SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "") # from Slack App Basic Info
    # Comma-separated channel names to listen in (without #)
    # e.g. "aie-eng-helpdesk,hpe-private-cloud-ai"
    SLACK_CHANNELS = os.getenv("SLACK_CHANNELS", "aie-eng-helpdesk")
    SLACK_SOURCE_MODE = os.getenv("SLACK_SOURCE_MODE", "both")   # sfdc | pdf | both
    SLACK_PRODUCT_LINE = os.getenv("SLACK_PRODUCT_LINE", "")
