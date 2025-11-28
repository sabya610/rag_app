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
from app.models import KBChunk, QAHist, db
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
        n_ctx=4096,
        temperature=0.2,   # lower temp for more extractive behavior in fallback
        top_p=0.9,
        repeat_penalty=1.2,
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

        #Populate DB with embeddings from PDFs.
        populatedb()

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
