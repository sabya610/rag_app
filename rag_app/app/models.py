from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from pgvector.sqlalchemy import Vector
from app.config import Config
#from sentence_transformers import SentenceTransformer
#from llama_cpp import Llama



db = SQLAlchemy()

class KBChunk(db.Model):
    __tablename__ = "kb_chunks"
    id = db.Column(db.Integer, primary_key=True)
    chunk_id = db.Column(db.String)
    text = db.Column(db.Text, nullable=False)
    embedding = db.Column(Vector(Config.PGVECTOR_DIM), nullable=False)
    source_file = db.Column(db.String, nullable=False)


class QAHist(db.Model):
    __tablename__ = "qa_history"
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)



# Globals (loaded only once)
embedder = None
llama = None
"""
def load_models():
    global embedder, llama
    if embedder is None:
        print("[LOADING] Loading embedder")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("[OK] Embedder loaded.")

    if llama is None:
        print("[LOADING] Loading llama.cpp model")
        from llama_cpp import Llama 
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
"""