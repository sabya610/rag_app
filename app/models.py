from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from pgvector.sqlalchemy import Vector
from app.config import Config
from datetime import datetime
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


class SlackMessage(db.Model):
    __tablename__ = "slack_messages"
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.String, unique=True, nullable=False)  # Slack ts
    channel_id = db.Column(db.String, nullable=False)
    channel_name = db.Column(db.String, nullable=False)
    user_id = db.Column(db.String, nullable=False)
    user_name = db.Column(db.String, nullable=False)
    text = db.Column(db.Text, nullable=False)
    embedding = db.Column(Vector(Config.PGVECTOR_DIM), nullable=False)
    thread_id = db.Column(db.String, nullable=True)  # Parent message ts if this is a reply
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    imported_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'message_id': self.message_id,
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'text': self.text,
            'thread_id': self.thread_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class SlackThread(db.Model):
    __tablename__ = "slack_threads"
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.String, unique=True, nullable=False)
    channel_id = db.Column(db.String, nullable=False)
    channel_name = db.Column(db.String, nullable=False)
    root_user_id = db.Column(db.String, nullable=False)
    root_user_name = db.Column(db.String, nullable=False)
    root_text = db.Column(db.Text, nullable=False)
    thread_embedding = db.Column(Vector(Config.PGVECTOR_DIM), nullable=False)
    reply_count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    imported_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'thread_id': self.thread_id,
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'root_user_id': self.root_user_id,
            'root_user_name': self.root_user_name,
            'root_text': self.root_text,
            'reply_count': self.reply_count,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


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