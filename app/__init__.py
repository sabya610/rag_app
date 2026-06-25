#gunicorn -w 4 "app:create_app()"

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import Config
from app.models import db
from dotenv import load_dotenv
from app.routes.rag_routes import rag_bp
from sqlalchemy import event
import os
#from sentence_transformers import SentenceTransformer
#from llama_cpp import Llama
from app.utils import (
    extract_text_from_pdfs_single, split_text,
    retrieve_relevant_chunks_pg, load_embeddings_to_pg,
    llm_answer, overlapping_chunks
)
from app.models import KBChunk, QAHist, SFDCArticle, db
from app.config import Config
from app.services.populate_db import populatedb


# Globals (loaded only once)
embedder = None
llama = None

def load_models():
    global embedder, llama
    #if embedder is None:
    print("[LOADING] Loading embedder (offline)")
    from sentence_transformers import SentenceTransformer  # lazy import

    model_path = Config.EMBEDDING_MODEL
    print(f"[INFO] Loading model from: {model_path}")

    embedder = SentenceTransformer(model_path)  # always local
    print("[OK] Embedder loaded (offline).")


    #if llama is None:
    print("[LOADING] Loading llama.cpp model")
    from llama_cpp import Llama  # lazy import

    llama = Llama(
        model_path=Config.MODEL_PATH,
        n_ctx=8192,
        chat_format="llama-3",
        verbose=False,
        use_mlock=True,
        use_mmap=True,
    )
    print("[OK] LLaMA loaded.")

    return embedder, llama

def create_app():

    base_dir = os.path.abspath(os.path.dirname(__file__))  # this is rag_app/app
    templates_dir = os.path.join(base_dir, "..", "templates")  # go one level up
    app = Flask(__name__,template_folder="templates")

    print("\n Flask template folder: \n", app.template_folder)

    app.config.from_object(Config)

    app.config["SQLALCHEMY_DATABASE_URI"] =  Config.DB_URI

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    """Initialize database and enforce UTF-8 encoding"""
    db.init_app(app)

    with app.app_context():
        #embedder, llama = load_models()
        #app.embedder = embedder
        #app.llama = llama
        if os.getenv("CI", "0") != "1":
           embedder, llama = load_models()
           app.embedder = embedder
           app.llama = llama
        else:
            print("[CI MODE] Skipping model loading")
        
        db.create_all()

        # Migrate: add missing columns to existing tables
        try:
            from sqlalchemy import text, inspect
            inspector = inspect(db.engine)
            if "qa_history" in inspector.get_table_names():
                existing = [c["name"] for c in inspector.get_columns("qa_history")]
                if "source" not in existing:
                    db.session.execute(text(
                        "ALTER TABLE qa_history ADD COLUMN source VARCHAR DEFAULT 'pdf'"
                    ))
                    db.session.commit()
                    print("[MIGRATE] Added 'source' column to qa_history table.")
        except Exception as e:
            print(f"[MIGRATE] Migration check skipped: {e}")

        #Populate DB with embeddings from PDFs.
        populatedb()

        # Initialize SFDC client if configured
        if Config.SFDC_ENABLED:
            from app.services.sfdc_client import get_sfdc_client
            sfdc = get_sfdc_client()
            if sfdc:
                app.sfdc_client = sfdc
                print("[OK] SFDC client initialized.")
            else:
                app.sfdc_client = None
                print("[WARN] SFDC credentials not configured. SFDC features disabled.")
        else:
            app.sfdc_client = None
            print("[INFO] SFDC integration disabled via SFDC_ENABLED=false.")

        @event.listens_for(db.engine, "connect")
        def set_utf8_encoding(dbapi_connection, connection_record):
            cur = dbapi_connection.cursor()
            try:
                cur.execute("SET client_encoding TO 'UTF8'")
            except Exception as e:
                print(f"[WARN] Could not set UTF-8 encoding: {e}")
            finally:
                cur.close()

    # Register Blueprints

    app.register_blueprint(rag_bp)


    return app
